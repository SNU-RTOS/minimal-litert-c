#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

void preprocessImage(const std::string& image_path, cv::Mat& output_rgb)
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

void fillInputTensor(const cv::Mat& rgb, TfLiteTensor* input_tensor, tflite::Interpreter* interpreter)
{
    const int height = 224, width = 224, channels = 3;
    bool is_quant = (input_tensor->type == kTfLiteUInt8);

    if (is_quant)
    {
        uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
        std::memcpy(input, rgb.data, height * width * channels);
    }
    else
    {
        float* input = interpreter->typed_input_tensor<float>(0);
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};

        for (int i = 0; i < height * width; ++i)
        {
            const cv::Vec3b& pix = rgb.at<cv::Vec3b>(i / width, i % width);
            for (int c = 0; c < channels; ++c)
                input[i * channels + c] = (pix[c] / 255.0f - mean[c]) / std[c];
        }
    }
}

void softmaxStable(const float* input, std::vector<float>& output, int size)
{
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    if (sum > 0.0f)
    {
        for (int i = 0; i < size; ++i)
        {
            output[i] /= sum;
        }
    }
}

std::vector<int> topKIndices(const std::vector<float>& data, int k)
{
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&data](int a, int b) { return data[a] > data[b]; });
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

    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    cv::Mat rgb_image;

    try {
        preprocessImage(argv[2], rgb_image);
        fillInputTensor(rgb_image, input_tensor, interpreter.get());
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    int num_classes = output_tensor->dims->data[1];

    if (output_tensor->type == kTfLiteFloat32)
    {
        float* output_data = interpreter->typed_output_tensor<float>(0);
        std::vector<float> probs(num_classes);
        softmaxStable(output_data, probs, num_classes);

        auto top_k_indices = topKIndices(probs, 5);
        std::cout << "Top 5 predictions:" << std::endl;
        for (int idx : top_k_indices)
        {
            std::cout << "Class " << idx << ": " << probs[idx] << std::endl;
        }
    }
    else if (output_tensor->type == kTfLiteUInt8)
    {
        uint8_t* data = interpreter->typed_output_tensor<uint8_t>(0);
        float scale = output_tensor->params.scale;
        int zero = output_tensor->params.zero_point;

        std::vector<float> scores(num_classes);
        for (int i = 0; i < num_classes; ++i)
            scores[i] = (data[i] - zero) * scale;

        auto top_k_indices = topKIndices(scores, 5);
        std::cout << "Top 5 predictions:" << std::endl;
        for (int idx : top_k_indices)
        {
            std::cout << "Class " << idx << ": " << scores[idx] << std::endl;
        }
    }

    return 0;
}
