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
        util::timer_start("stage0:total");

        std::cout << "[stage0] Loading image: " << images[i] << std::endl;
        util::timer_start("stage0:load_image");
        cv::Mat origin_image = cv::imread(images[i]);
        util::timer_stop("stage0:load_image");

        if (origin_image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            util::timer_stop("stage0:total");
            continue;
        }

        int input_height = 224;
        int input_width = 224;

        std::cout << "[stage0] Preprocessing image: " << images[i] << "\n" << std::endl;
        util::timer_start("stage0:preprocess");
        cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);
        util::timer_stop("stage0:preprocess");

        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            util::timer_stop("stage0:total");
            continue;
        }

        util::timer_start("stage0:flatten");
        std::vector<float> input_vector(preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(input_vector.data(), preprocessed_image.ptr<float>(), input_vector.size() * sizeof(float));
        util::timer_stop("stage0:flatten");

        IntermediateResult ir;
        ir.index = i;
        ir.data = std::move(input_vector);

        std::cout << "[stage0] Enqueuing preprocessed image index: " << ir.index << std::endl;
        util::timer_start("stage0:enqueue");
        queue0.push(ir);
        util::timer_stop("stage0:enqueue");

        util::timer_stop("stage0:total");

        next_time += std::chrono::milliseconds(rate_ms);
        std::this_thread::sleep_until(next_time);
    }

    std::cout << "[stage0] Finished preprocessing. Signaling shutdown.\n";
    queue0.signal_shutdown();
}


void stage1_worker(tflite::Interpreter* interp) {
    std::cout << "[stage1] Started inference thread (model0 / GPU)\n";
    IntermediateResult ir;

    while (queue0.pop(ir)) {
        util::timer_start("stage1:total");

        std::cout << "[stage1] Dequeued image index: " << ir.index << std::endl;

        util::timer_start("stage1:copy_input");
        float* input = interp->typed_input_tensor<float>(0);
        std::copy(ir.data.begin(), ir.data.end(), input);
        util::timer_stop("stage1:copy_input");

        std::cout << "[stage1] Invoking model0...\n";
        util::timer_start("stage1:invoke");
        interp->Invoke();
        util::timer_stop("stage1:invoke");

        util::timer_start("stage1:postprocess");
        std::vector<float> flat;
        std::vector<int> bounds{0};

        for (int idx : interp->outputs()) {
            TfLiteTensor* t = interp->tensor(idx);

            int sz = 1;
            for (int d = 0; d < t->dims->size; ++d)
                sz *= t->dims->data[d];

            int prev = flat.size();
            flat.resize(prev + sz);
            std::copy(t->data.f, t->data.f + sz, flat.begin() + prev);
            bounds.push_back(prev + sz);
        }

        ir.data = std::move(flat);
        ir.tensor_boundaries = std::move(bounds);
        util::timer_stop("stage1:postprocess");

        util::timer_start("stage1:enqueue");
        std::cout << "[stage1] Enqueuing result for image index: " << ir.index << std::endl;
        queue1.push(ir);
        util::timer_stop("stage1:enqueue");

        util::timer_stop("stage1:total");
    }

    std::cout << "[stage1] Finished inference. Signaling shutdown.\n";
    queue1.signal_shutdown();
}



void stage2_worker(tflite::Interpreter* interp) {
    std::cout << "[stage2] Started inference thread (model1 / CPU)\n";
    IntermediateResult ir;

    while (queue1.pop(ir)) {
        util::timer_start("stage2:total");

        std::cout << "[stage2] Dequeued intermediate result for index: " << ir.index << std::endl;

        util::timer_start("stage2:copy_input");
        float* input = interp->typed_input_tensor<float>(0);
        std::copy(ir.data.begin(), ir.data.end(), input);
        util::timer_stop("stage2:copy_input");

        std::cout << "[stage2] Invoking model1...\n";
        util::timer_start("stage2:invoke");
        interp->Invoke();
        util::timer_stop("stage2:invoke");

        util::timer_start("stage2:postprocess");
        TfLiteTensor* out = interp->tensor(interp->outputs()[0]);
        int numel = 1;
        for (int d = 0; d < out->dims->size; ++d)
            numel *= out->dims->data[d];

        std::vector<float> out_data(numel);
        std::copy(interp->typed_output_tensor<float>(0),
                  interp->typed_output_tensor<float>(0) + numel,
                  out_data.begin());

        std::cout << "[stage2] Top-5 prediction for image index " << ir.index << ":\n";
        auto label_map = util::load_class_labels("labels.json");
        util::print_top_predictions(out_data, out->dims->data[1], 5, false, label_map);
        util::timer_stop("stage2:postprocess");

        util::timer_stop("stage2:total");
    }

    std::cout << "[stage2] Finished all inference.\n";
}


int main(int argc, char* argv[]) {
    util::timer_start("main:total");

    // --- [1] Parse model paths and input arguments ---
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model0.tflite> <model1.tflite> <image_path> [--input-rate=XX]" << std::endl;
        return 1;
    }

    const char* model0_path = argv[1];
    const char* model1_path = argv[2];
    std::string image_path = argv[3];
    int rate_ms = 0;

    // Optional --input-rate
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-rate=", 0) == 0) {
            rate_ms = std::stoi(arg.substr(13));
        }
    }

    // Fill image list with the same image 100 times
    std::vector<std::string> images(100, image_path);

    // --- [2] Load TFLite models ---
    util::timer_start("main:load_models");
    auto model0 = tflite::FlatBufferModel::BuildFromFile(model0_path);
    auto model1 = tflite::FlatBufferModel::BuildFromFile(model1_path);
    util::timer_stop("main:load_models");

    // --- [3] Build interpreters ---
    util::timer_start("main:build_interpreters");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp0, interp1;
    tflite::InterpreterBuilder(*model0, resolver)(&interp0);
    tflite::InterpreterBuilder(*model1, resolver)(&interp1);
    util::timer_stop("main:build_interpreters");

    // --- [4] Set threading options ---
    interp0->SetNumThreads(1);
    interp1->SetNumThreads(4);

    // --- [5] Apply GPU delegate to model0 ---
    util::timer_start("main:apply_delegate");
    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);
    interp0->ModifyGraphWithDelegate(gpu);
    util::timer_stop("main:apply_delegate");

    // --- [6] Allocate tensors ---
    util::timer_start("main:allocate_tensors");
    interp0->AllocateTensors();
    interp1->AllocateTensors();
    util::timer_stop("main:allocate_tensors");

    // --- [7] Launch pipeline threads ---
    util::timer_start("main:thread_join");
    std::thread t0(stage0_worker, std::ref(images), rate_ms);
    std::thread t1(stage1_worker, interp0.get());
    std::thread t2(stage2_worker, interp1.get());

    t0.join();
    t1.join();
    t2.join();
    util::timer_stop("main:thread_join");

    // --- [8] Cleanup ---
    if (gpu) TfLiteGpuDelegateV2Delete(gpu);

    util::timer_stop("main:total");

    // --- [9] Print timers ---
    util::print_all_timers();

    return 0;
}
