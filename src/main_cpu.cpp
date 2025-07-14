// xnn-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>

#include "opencv2/opencv.hpp" //opencv

#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"

void PrintExecutionPlanOps(std::unique_ptr<tflite::Interpreter>& interpreter);

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
    util::timer_start("Load Model");
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    util::timer_stop("Load Model");

    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Sleep for 1 second to see the output before exit

    /* Build interpreter */
    util::timer_start("Build Interpreter");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    util::timer_stop("Build Interpreter");

    /* Check the created execution plan */
    std::vector<int> execution_plan = interpreter->execution_plan();
    std::cout << "The model contains " << execution_plan.size() << " nodes." << std::endl;
    PrintExecutionPlanOps(interpreter);

    /* Apply XNNPACK delegate */
    util::timer_start("Apply Delegate");
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
    /* Check the modified execution plan */
    execution_plan = interpreter->execution_plan();
    std::cout << "The model contains " << execution_plan.size() << " nodes." << std::endl;
    PrintExecutionPlanOps(interpreter);

    util::timer_stop("Apply Delegate");


    /* Allocate Tensor */
    util::timer_start("Allocate Tensor");
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }
    util::timer_stop("Allocate Tensor");


    util::print_model_summary(interpreter.get(), delegate_applied);

    /* Load input image */
    util::timer_start("Load Input Image");
    cv::Mat origin_image = cv::imread(image_path);
    if (origin_image.empty())
        throw std::runtime_error("Failed to load image: " + image_path);
    util::timer_stop("Load Input Image");

    /* Preprocessing */
    util::timer_start("E2E Total(Pre+Inf+Post)");
    util::timer_start("Preprocessing");

    // Get input tensor info
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    int input_channels = input_tensor->dims->data[3];

    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;

    // Preprocess input data
    cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);

    // Copy HWC float32 cv::Mat to TFLite input tensor
    float *input_tensor_buffer = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor_buffer, preprocessed_image.ptr<float>(),
                preprocessed_image.total() * preprocessed_image.elemSize());

    util::timer_stop("Preprocessing");

    /* Inference */
    util::timer_start("Inference");

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        /* Deallocate delegate */
        if (xnn_delegate)
        {
            TfLiteXNNPackDelegateDelete(xnn_delegate);
        }
        return 1;
    }

    util::timer_stop("Inference");

    /* PostProcessing */
    util::timer_start("Postprocessing");

    // Get output tensor
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    std::cout << "[INFO] Output shape : ";
    util::print_tensor_shape(output_tensor);
    std::cout << std::endl;

    float *logits = interpreter->typed_output_tensor<float>(0);
    int num_classes = output_tensor->dims->data[1];

    std::vector<float> probs(num_classes);
    util::softmax(logits, probs, num_classes);

    util::timer_stop("Postprocessing");
    util::timer_stop("E2E Total(Pre+Inf+Post)");

    /* Print Results */
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

    /* Print Timers */
    util::print_all_timers();
    std::cout << "========================" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Sleep for 1 second to see the output before exit

    /* Deallocate delegate */
    if (xnn_delegate)
    {
        TfLiteXNNPackDelegateDelete(xnn_delegate);
    }


    return 0;
}

void PrintExecutionPlanOps(std::unique_ptr<tflite::Interpreter>& interpreter) {
    std::cout << "The model contains " << interpreter->execution_plan().size() 
            << " nodes in execution plan." << std::endl;

    for (int node_index : interpreter->execution_plan()) {
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        if (!node_and_reg) {
            std::cerr << "Failed to get node " << node_index << std::endl;
            continue;
        }

        const TfLiteNode& node = node_and_reg->first;
        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << node_index << ": ";

        if (registration.builtin_code != tflite::BuiltinOperator_CUSTOM) {
            std::cout << tflite::EnumNameBuiltinOperator(
                static_cast<tflite::BuiltinOperator>(registration.builtin_code));
        } else {
            std::cout << "CUSTOM: " 
                << (registration.custom_name ? registration.custom_name : "unknown");
        }
        std::cout << std::endl;
    }

    for (int node_index = 0; node_index < interpreter->nodes_size(); ++node_index) {
        auto* node_and_reg = interpreter->node_and_registration(node_index);
        if (!node_and_reg) continue;

        const TfLiteNode& node = node_and_reg->first;
        const TfLiteRegistration& reg = node_and_reg->second;

        std::cout << "Node " << node_index << ": ";
        if (reg.builtin_code != tflite::BuiltinOperator_CUSTOM) {
            std::cout << tflite::EnumNameBuiltinOperator(
                static_cast<tflite::BuiltinOperator>(reg.builtin_code));
        } else {
            std::cout << "CUSTOM: " << (reg.custom_name ? reg.custom_name : "unknown");
        }

        if (node.delegate != nullptr) {
            std::cout << " [DELEGATED]";
        }

        std::cout << std::endl;
    }
}