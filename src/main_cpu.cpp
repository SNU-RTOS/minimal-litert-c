// main_cpu.cpp
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

void softmax(const float *logits, std::vector<float> &probs, int size);
cv::Mat preprocess_image(cv::Mat &image, int target_width, int target_height);

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

    /* Build interpreter */

    /* Apply XNNPACK delegate */

    /* Allocate Tensor */

    /* Load input image */

    /* Preprocessing */

    /* Inference */

    /* PostProcessing */

    /* Print Results */

    /* Deallocate delegate */

    return 0;
}

// Preprocess: load, resize, center crop, RGB → float32 + normalize
cv::Mat preprocess_image(cv::Mat &image, int target_height, int target_width)
{
    int h = image.rows, w = image.cols;
    float scale = 256.0f / std::min(h, w);
    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int x = (new_w - target_width) / 2;
    int y = (new_h - target_height) / 2;
    cv::Rect crop(x, y, target_width, target_height);

    cv::Mat cropped = resized(crop);
    cv::Mat rgb_image;
    cv::cvtColor(cropped, rgb_image, cv::COLOR_BGR2RGB);

    // Normalize to float32
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    for (int c = 0; c < 3; ++c)
        channels[c] = (channels[c] - mean[c]) / std[c];
    cv::merge(channels, float_image);

    return float_image;
}

// Apply softmax to logits
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