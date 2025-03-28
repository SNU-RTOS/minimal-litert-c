#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

void softmax(const float *logits, std::vector<float> &probs, int size)
{
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f)
    {
        for (int i = 0; i < size; ++i)
        {
            probs[i] /= sum;
        }
    }
}

void preprocess_image(const std::string &image_path, cv::Mat &output_rgb)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
        throw std::runtime_error("Failed to load image: " + image_path);

    int h = image.rows, w = image.cols;
    float scale = 256.0f / std::min(h, w);
    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int x = (new_w - 224) / 2;
    int y = (new_h - 224) / 2;
    cv::Rect crop(x, y, 224, 224);

    cv::Mat cropped = resized(crop);
    cv::cvtColor(cropped, output_rgb, cv::COLOR_BGR2RGB);
}

void fill_input_tensor(const cv::Mat &rgb, TfLiteTensor *input_tensor, tflite::Interpreter *interpreter)
{
    const int height = 224, width = 224, channels = 3;

    float *input = interpreter->typed_input_tensor<float>(0);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    for (int i = 0; i < height * width; ++i)
    {
        const cv::Vec3b &pix = rgb.at<cv::Vec3b>(i / width, i % width);
        for (int c = 0; c < channels; ++c)
            input[i * channels + c] = (pix[c] / 255.0f - mean[c]) / std[c];
    }
}

std::vector<int> get_topK_indices(const std::vector<float> &data, int k)
{
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&data](int a, int b)
        { return data[a] > data[b]; });
    indices.resize(k);
    return indices;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to build interpreter" << std::endl;
        return 1;
    }

    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    cv::Mat rgb_image;

    preprocess_image(argv[2], rgb_image);
    fill_input_tensor(rgb_image, input_tensor, interpreter.get());

    // Inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    // Post process
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    int num_classes = output_tensor->dims->data[1];
    float *logits = interpreter->typed_output_tensor<float>(0);

    std::vector<float> probs(num_classes);
    softmax(logits, probs, num_classes);

    auto top_k_indices = get_topK_indices(probs, 5);
    std::cout << "Top 5 predictions:" << std::endl;
    for (int idx : top_k_indices)
    {
        std::cout << "Class " << idx << ": " << probs[idx] << std::endl;
    }

    return 0;
}
