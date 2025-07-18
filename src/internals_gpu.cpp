// gpu-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp> //opencv

#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/delegates/gpu/delegate.h"             // for gpu delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"


int main(int argc, char *argv[])
{
    std::cout << "====== main_gpu ====" << std::endl;

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

    /* Build interpreter */
    util::timer_start("Build Interpreter");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    util::timer_stop("Build Interpreter");

    /* Apply GPU Delegate */
    util::timer_start("Apply Delegate");
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

    std::cout << "GPU delegate create 1" << std::endl;
    TfLiteDelegate *gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
    std::cout << "GPU delegate create 2" << std::endl;
    bool delegate_applied = false;
    std::cout << "GPU delegate apply 1" << std::endl;
    if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk)
    {
        delegate_applied = true;
    }
    else
    {
        std::cerr << "Failed to apply GPU delegate" << std::endl;
    }
    std::cout << "GPU delegate apply 2" << std::endl;

    /* ======================================================================================================== */
    /* Code snippet for checking how the subgraph changes after applying a delegate */
    std::cout << "\nNumber of nodes of subgraph 0: " << interpreter->nodes_size() << std::endl;
    for(int node_index = 0; node_index < interpreter->nodes_size(); node_index++) {
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        
        const TfLiteNode& node = node_and_reg->first;
        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << node_index << ": "
        << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code))
        << std::endl;
    }

    std::cout << "\nExecution plan size of subgraph 0: " << interpreter->execution_plan().size() << std::endl;
    util::print_execution_plan(interpreter);
    // input and output tensors of the delegate node
    {
        // Number of tensors
        std::cout << "\nNumber of tensors in subgraph 0: " << interpreter->tensors_size() << std::endl;

        for (int i = 0; i < interpreter->tensors_size(); ++i) {
            const TfLiteTensor* t = interpreter->tensor(i);
            if (true/*t->allocation_type != kTfLiteNone*/) {
                std::cout << "Tensor " << i << " is used by interpreter, alloc: ";
                switch(t->allocation_type) {
                    case kTfLiteMemNone: 
                        std::cout << "kTfLiteMemNone" << std::endl;
                        break;
                    case kTfLiteMmapRo: 
                        std::cout << "kTfLiteMmapRo" << std::endl;
                        break;
                    case kTfLiteArenaRw:
                        std::cout << "kTfLiteArenaRw" << std::endl;
                        break;
                    case kTfLiteArenaRwPersistent: 
                        std::cout << "kTfLiteArenaRwPersistent" << std::endl;
                        break;
                    case kTfLiteDynamic: 
                        std::cout << "kTfLiteDynamic" << std::endl;
                        break;
                    case kTfLitePersistentRo: 
                        std::cout << "kTfLitePersistentRo" << std::endl;
                        break;
                    case kTfLiteCustom: 
                        std::cout << "kTfLiteCustom" << std::endl;
                        break;
                    default: 
                        std::cout << "Unknown" << std::endl;
                        break;
                }
            }
        }

        const TfLiteNode& node = (interpreter->node_and_registration(50))->first; // We already know the node index
        const TfLiteRegistration& reg = (interpreter->node_and_registration(50))->second;
        
        // Access input tensors
        std::cout << "\nInputs:\n";
        for (int i = 0; i < node.inputs->size; ++i) {
            int tensor_index = node.inputs->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "])\n";
        }
        std::cout << std::endl;

        // Access intermediate tensors
        std::cout << "Intermediates:\n";
        for (int i = 0; i < node.intermediates->size; ++i) {
            int tensor_index = node.intermediates->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]) ";
        }
        std::cout << std::endl;

        // Access intermediate tensors
        std::cout << "Temporaries:\n";
        for (int i = 0; i < node.temporaries->size; ++i) {
            int tensor_index = node.temporaries->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]) ";
        }
        std::cout << std::endl;

        // Access output tensors
        std::cout << "Outputs:\n";
        for (int i = 0; i < node.outputs->size; ++i) {
            int tensor_index = node.outputs->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]) ";
        }
        std::cout << std::endl;
    }
    /* ======================================================================================================== */

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

    /* Deallocate delegate */
    if (gpu_delegate)
    {
        TfLiteGpuDelegateV2Delete(gpu_delegate);
    }
    return 0;
}
