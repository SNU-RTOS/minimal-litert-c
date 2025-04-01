// xnn-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp> //opencv

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "util.hpp"

void softmax_bool(const float *logits, std::vector<float> &probs, int size, bool apply_softmax);
cv::Mat preprocess_image_resnet_keras_application_caffe(cv::Mat &image, int target_width, int target_height);

int main(int argc, char *argv[])
{
    std::cout << "====== main_cpu ====" << std::endl;

    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> <label_json_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string label_path = argv[3];

    /* Load model */
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    /* Build interpreter */
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    /* Apply XNNPACK delegate */
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    bool delegate_applied = false;
    if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
    {
        delegate_applied = true;
    }
    else
    {
        std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
    }

    /* Allocate Tensor */
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }

    util::print_model_summary(interpreter.get(), delegate_applied);

    /* Load input image */
    cv::Mat origin_image = cv::imread(image_path);
    if (origin_image.empty())
        throw std::runtime_error("Failed to load image: " + image_path);

    /* Preprocessing */
    // Get input tensor info
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    int input_channels = input_tensor->dims->data[3];

    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;

    // Preprocess input data
    cv::Mat preprocessed_image = preprocess_image_resnet_keras_application_caffe(origin_image, input_height, input_width);

    // Copy HWC float32 cv::Mat to TFLite input tensor
    float *input_tensor_buffer = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor_buffer, preprocessed_image.ptr<float>(),
                preprocessed_image.total() * preprocessed_image.elemSize());

    /* Inference */
    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    /* PostProcessing */
    // Get output tensor
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    std::cout << "[INFO] Output shape : ";
    util::print_tensor_shape(output_tensor);
    std::cout << std::endl;

    float *logits = interpreter->typed_output_tensor<float>(0);
    int num_classes = output_tensor->dims->data[1];

    std::vector<float> probs(num_classes);

    softmax_bool(logits, probs, num_classes, false);

    // /* Print Results */
    // Load class label mapping
    auto label_map = util::load_class_labels(label_path);

    // Print Top-5 results
    std::cout << "\n[INFO] Top 5 predictions:" << std::endl;
    auto top_k_indices = util::get_topK_indices(probs, 5);
    for (int idx : top_k_indices)
    {
        std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
        std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
    }

    /* Deallocate delegate */
    if (xnn_delegate)
    {
        TfLiteXNNPackDelegateDelete(xnn_delegate);
    }
    return 0;
}



cv::Mat preprocess_image_resnet_keras_application_caffe(cv::Mat &image, int target_height, int target_width)
{
    // Resize & convert to float32
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3);

    // Per-channel normalization (mean/std in RGB order)
    // const float mean[3] = {103.939f, 116.779f, 123.68f}; // BGR 순서
    const float mean[3] = {0.406f, 0.456f, 0.485f}; // BGR 순서

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c)
    {
        channels[c] = (channels[c] - (mean[c] * 255));
    }

    cv::Mat normalized;
    cv::merge(channels, normalized); // back to CV_32FC3, RGB order

    // Debug: check range
    double min_val, max_val;
    cv::minMaxLoc(normalized.reshape(1), &min_val, &max_val);
    std::cout << "Normalized RGB image range: [" << min_val << ", " << max_val << "]" << std::endl;

    return normalized;
}

// Apply softmax to logits
void softmax_bool(const float *logits, std::vector<float> &probs, int size, bool apply_softmax = true)
{
    probs.resize(size);

    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    if (!apply_softmax)
    {
        for (int i = 0; i < size; ++i)
        {
            probs[i] = logits[i];
        }
        return;
    }

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