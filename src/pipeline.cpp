#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/delegates/gpu/delegate.h"             // for gpu delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"
#include "thread_safe_queue.hpp"

#include <opencv2/opencv.hpp> //opencv
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <pthread.h>

// --- Data container used to pass results between pipeline stages ---
struct IntermediateResult {
    int index;                            // Index of the input image (used for tracking)
    std::vector<float> data;             // Flattened data (input/output tensor contents)
    std::vector<int> tensor_boundaries;  // Marks boundaries between multiple output tensors (if any)
};

// --- Thread-safe queues for inter-stage communication ---
// queue0: connects stage0 (preprocessing) to stage1 (first inference)
// queue1: connects stage1 to stage2 (final inference)
ThreadSafeQueue<IntermediateResult> queue0;
ThreadSafeQueue<IntermediateResult> queue1;


void stage0_worker(const std::vector<std::string>& images, int rate_ms) {
    std::cout << "[stage0] Started preprocessing thread\n";
    auto next_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); ++i) {
        std::cout << "[stage0] Loading image: " << images[i] << std::endl;
        // --- Load image from file ---
        cv::Mat origin_image = cv::imread(images[i]);
        if (origin_image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            continue;
        }

        // --- Get expected input shape from first-stage model ---
        // NOTE: input tensor shape should be passed or globally accessible
        int input_height = 224; // ← 이 값은 추후 인터프리터에서 읽도록 수정 가능
        int input_width = 224;

        // --- Preprocessing to float32 HWC layout normalized image ---
        std::cout << "[stage0] Preprocessing image: " << images[i] << "\n" << std::endl;
        cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);
        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            continue;
        }

        // --- Convert cv::Mat to flat float vector ---
        std::vector<float> input_vector(preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(input_vector.data(), preprocessed_image.ptr<float>(), input_vector.size() * sizeof(float));

        // --- Package into IntermediateResult and enqueue ---
        IntermediateResult ir;
        ir.index = i;
        ir.data = std::move(input_vector);

        std::cout << "[stage0] Enqueuing preprocessed image index: " << ir.index << std::endl;
        queue0.push(ir);

        next_time += std::chrono::milliseconds(rate_ms);
        std::this_thread::sleep_until(next_time);
    }

    std::cout << "[stage0] Finished preprocessing. Signaling shutdown.\n";
    queue0.signal_shutdown();
}


void stage1_worker(tflite::Interpreter* interp) {
    std::cout << "[stage1] Started inference thread (model0 / GPU)\n";
    IntermediateResult ir;

    // --- [1] Continuously pop preprocessed data from queue0 (output of stage0) ---
    while (queue0.pop(ir)) {

        // --- [2] Copy input data into the input tensor of model0 ---
        std::cout << "[stage1] Dequeued image index: " << ir.index << std::endl;
        float* input = interp->typed_input_tensor<float>(0);
        std::copy(ir.data.begin(), ir.data.end(), input);

        // --- [3] Run inference using model0 (typically GPU-accelerated) ---
        std::cout << "[stage1] Invoking model0...\n";
        interp->Invoke();

        // --- [4] Flatten all output tensors into a single vector ---
        std::vector<float> flat;
        std::vector<int> bounds{0};  // Used to track tensor boundaries in the flattened vector

        for (int idx : interp->outputs()) {
            TfLiteTensor* t = interp->tensor(idx);

            // Calculate number of elements in the tensor
            int sz = 1;
            for (int d = 0; d < t->dims->size; ++d)
                sz *= t->dims->data[d];

            // Append tensor data to flattened vector
            int prev = flat.size();
            flat.resize(prev + sz);
            std::copy(t->data.f, t->data.f + sz, flat.begin() + prev);

            // Record boundary position for this output tensor
            bounds.push_back(prev + sz);
        }

        // --- [5] Store processed results and enqueue for stage2 ---
        ir.data = std::move(flat);
        ir.tensor_boundaries = std::move(bounds);
        std::cout << "[stage1] Enqueuing result for image index: " << ir.index << std::endl;
        queue1.push(ir);
    }

    // --- [6] Signal stage2 that no more data will arrive ---
    std::cout << "[stage1] Finished inference. Signaling shutdown.\n";
    queue1.signal_shutdown();
}


void stage2_worker(tflite::Interpreter* interp) {
    std::cout << "[stage2] Started inference thread (model1 / CPU)\n";
    IntermediateResult ir;

    // --- [1] Continuously pop data from queue1 (output of stage1) ---
    while (queue1.pop(ir)) {

        // --- [2] Copy intermediate data into the input tensor of model1 ---
        std::cout << "[stage2] Dequeued intermediate result for index: " << ir.index << std::endl;
        float* input = interp->typed_input_tensor<float>(0);
        std::copy(ir.data.begin(), ir.data.end(), input);

        // --- [3] Run inference on stage2 (CPU-based) ---
        std::cout << "[stage2] Invoking model1...\n";
        interp->Invoke();

        // --- [4] Retrieve output tensor from the model ---
        TfLiteTensor* out = interp->tensor(interp->outputs()[0]);
        int numel = 1;
        for (int d = 0; d < out->dims->size; ++d)
            numel *= out->dims->data[d];  // Compute total number of elements

        // --- [5] Copy output data into a vector for post-processing or printing ---
        std::vector<float> out_data(numel);
        std::copy(interp->typed_output_tensor<float>(0),
                  interp->typed_output_tensor<float>(0) + numel,
                  out_data.begin());

        // --- [6] Print top-5 predictions from the output vector ---
        std::cout << "[stage2] Top-5 prediction for image index " << ir.index << ":\n";
        auto label_map = util::load_class_labels("labels.json");
        util::print_top_predictions(out_data, out->dims->data[1], 5, false, label_map);
    }
    
    std::cout << "[stage2] Finished all inference.\n";
}

int main(int argc, char* argv[]) {
    // --- [1] Parse model paths and input arguments ---
    const char* model0_path = argv[1];  // Path to first model (used in stage1)
    const char* model1_path = argv[2];  // Path to second model (used in stage2)
    std::vector<std::string> images;    // List of input image paths
    int rate_ms = 0;                    // Input rate in milliseconds

    // Parse additional arguments: image files and input rate
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-rate=", 0) == 0)
            rate_ms = std::stoi(arg.substr(13));  // Extract rate from --input-rate=XX
        else
            images.push_back(arg);  // Treat as image file path
    }

    // --- [2] Load TFLite models ---
    auto model0 = tflite::FlatBufferModel::BuildFromFile(model0_path);
    auto model1 = tflite::FlatBufferModel::BuildFromFile(model1_path);

    // --- [3] Build interpreters with operation resolver ---
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp0, interp1;
    tflite::InterpreterBuilder(*model0, resolver)(&interp0);
    tflite::InterpreterBuilder(*model1, resolver)(&interp1);

    // --- [4] Configure number of threads for each model ---
    interp0->SetNumThreads(1);  // GPU delegate runs single-threaded
    interp1->SetNumThreads(4);  // CPU model can utilize 4 threads

    // --- [5] Configure and apply GPU delegate to model0 ---
    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);
    interp0->ModifyGraphWithDelegate(gpu);  // Enable GPU acceleration

    // --- [6] Allocate tensors for both interpreters ---
    interp0->AllocateTensors();
    interp1->AllocateTensors();

    // --- [7] Launch pipeline threads: stage0 (preprocess), stage1 (GPU), stage2 (CPU) ---
    std::thread t0(stage0_worker, std::ref(images), rate_ms);  // Image preprocessing
    std::thread t1(stage1_worker, interp0.get());              // First-stage inference
    std::thread t2(stage2_worker, interp1.get());              // Second-stage inference

    // --- [8] Wait for all threads to complete ---
    t0.join();
    t1.join();
    t2.join();

    // --- [9] Deallocate GPU delegate ---
    if (gpu) TfLiteGpuDelegateV2Delete(gpu);

    return 0;
}
