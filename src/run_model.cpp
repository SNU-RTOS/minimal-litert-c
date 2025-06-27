#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>  // For std::accumulate
#include "model_utils.h"  // preprocess_image, print_top_predictions

using Clock = std::chrono::high_resolution_clock;

// new
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.tflite> <image1> [<image2> ...]" << std::endl;
        return 1;
    }
    const char* model_path = argv[1];

    // Collect image paths
    std::vector<std::string> images;
    for (int i = 2; i < argc; ++i) {
        images.emplace_back(argv[i]);
    }

    // Load TFLite model
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter" << std::endl;
        return 1;
    }
    interpreter->SetNumThreads(1);
    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);
    if (interpreter->ModifyGraphWithDelegate(gpu) != kTfLiteOk) {
        std::cerr << "Failed to apply GPU delegate" << std::endl;
        return 1;
    }
    interpreter->AllocateTensors();
    std::cout << "Starting inference on " << images.size() << " images." << std::endl;
    // Timing accumulators and latencies
    long total_pre_ms = 0, total_inf_ms = 0, total_post_ms = 0;
    std::vector<long> e2e_latencies;
    for (size_t i = 0; i < images.size(); ++i) {
        std::cout << "" << i << "" << std::endl;
        // 1. Preprocess (returns flattened input tensor data)
        auto t0 = Clock::now();
        auto preprocessed = preprocess_image(images[i], false);
        if (preprocessed.empty()) {
            std::cerr << "Preprocess failed: " << images[i] << std::endl;
            continue;
        }

        auto t1 = Clock::now();
        long pre_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        total_pre_ms += pre_ms;

        // Copy preprocessed data into interpreter input
        float* input = interpreter->typed_input_tensor<float>(0);
        std::copy(preprocessed.begin(), preprocessed.end(), input);
        // 2. Inference
        auto t2 = Clock::now();

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed on image " << images[i] << std::endl;
            continue;
        }

        auto t3 = Clock::now();
        long inf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        total_inf_ms += inf_ms;

        // 3. Postprocess and output top predictions
        auto t4 = Clock::now();
        TfLiteTensor* out_t = interpreter->tensor(interpreter->outputs()[0]);
        int numel = 1;
        for (int d = 0; d < out_t->dims->size; ++d) numel *= out_t->dims->data[d];
        const float* out_data = interpreter->typed_output_tensor<float>(0);
        std::vector<float> output(out_data, out_data + numel);
        int num_classes = out_t->dims->data[1];
        print_top_predictions(output, num_classes, 5, true);
        auto t5 = Clock::now();
        long post_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
        total_post_ms += post_ms;

        long e2e_ms = pre_ms + inf_ms + post_ms;
        e2e_latencies.push_back(e2e_ms);

        std::cout << "Image[" << i << "] Pre: " << pre_ms
                  << " ms, Inf: " << inf_ms
                  << " ms, Post: " << post_ms
                  << " ms, E2E: " << e2e_ms << " ms" << std::endl;
    }

    // Summary
    size_t count = e2e_latencies.size();
    if (count > 0) {
        double avg_pre = static_cast<double>(total_pre_ms) / count;
        double avg_inf = static_cast<double>(total_inf_ms) / count;
        double avg_post = static_cast<double>(total_post_ms) / count;
        auto min_e2e = *std::min_element(e2e_latencies.begin(), e2e_latencies.end());
        auto max_e2e = *std::max_element(e2e_latencies.begin(), e2e_latencies.end());
        double avg_e2e = std::accumulate(e2e_latencies.begin(), e2e_latencies.end(), 0.0) / count;
        double throughput = 1000.0 / avg_e2e;

        std::cout << "\nSummary over " << count << " images:\n";
        std::cout << "Avg Preprocess: " << avg_pre << " ms\n";
        std::cout << "Avg Inference: " << avg_inf << " ms\n";
        std::cout << "Avg Postprocess: " << avg_post << " ms\n";
        std::cout << "Min E2E: " << min_e2e << " ms, Max E2E: " << max_e2e << " ms, Avg E2E: " << avg_e2e << " ms\n";
        std::cout << "Throughput: " << throughput << " images/s" << std::endl;
    }

    return 0;
}
