#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
// #include <tensorflow/lite/delegates/gpu/delegate.h>
// #include <tensorflow/lite/tools/delegates/delegate_provider.h>
#include <iostream>
#include <memory>

class TFLiteInferenceApp {
private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteDelegate* gpu_delegate_ = nullptr;

    // Custom error reporter
    class ErrorReporter : public tflite::ErrorReporter {
    public:
        int Report(const char* format, va_list args) override {
            char buffer[1024];
            int size = vsnprintf(buffer, sizeof(buffer), format, args);
            std::cerr << "TFLite Error: " << buffer << std::endl;
            return size;
        }
    } error_reporter_;

public:
    bool LoadModel(const char* model_path) {
        // Load model
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path, &error_reporter_);
        if (!model_) {
            std::cerr << "Failed to load model" << std::endl;
            return false;
        }
        
        // Log model details
        std::cout << "Model loaded successfully" << std::endl;
        
        return true;
    }

    bool BuildInterpreter() {
        // Create interpreter builder
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model_, resolver);

        // Set number of threads
        builder.SetNumThreads(4);

        // Create interpreter
        if (builder.operator()(&interpreter_) != kTfLiteOk) {
            std::cerr << "Failed to build interpreter" << std::endl;
            return false;
        }

        // Print interpreter memory plan
        PrintMemoryPlan();
        
        // Print original execution plan
        std::cout << "\n=== Original Execution Plan (Before GPU Delegation) ===" << std::endl;
        PrintExecutionPlan();

        return true;
    }

    void PrintMemoryPlan() {
        std::cout << "\n=== Memory Plan ===" << std::endl;
        std::cout << "Tensors count: " << interpreter_->tensors_size() << std::endl;
        
        size_t total_bytes = 0;
        
        // Print details for each tensor
        for (size_t i = 0; i < interpreter_->tensors_size(); ++i) {
            TfLiteTensor* tensor = interpreter_->tensor(i);
            total_bytes += tensor->bytes;
            
            // Get allocation type string
            std::string allocation_type;
            switch(tensor->allocation_type) {
                case kTfLiteMmapRo:
                    allocation_type = "CONSTANT (kTfLiteMmapRo)";
                    break;
                case kTfLiteArenaRw:
                    allocation_type = "DYNAMIC (kTfLiteArenaRw)";
                    break;
                case kTfLiteArenaRwPersistent:
                    allocation_type = "PERSISTENT (kTfLiteArenaRwPersistent)";
                    break;
                case kTfLiteDynamic:
                    allocation_type = "DYNAMIC (kTfLiteDynamic)";
                    break;
                case kTfLiteCustom:
                    allocation_type = "CUSTOM (kTfLiteCustom)";
                    break;
                default:
                    allocation_type = "UNKNOWN (" + std::to_string(tensor->allocation_type) + ")";
            }
            
            std::cout << "\nTensor " << i << ":" << std::endl;
            std::cout << "  Name: " << (tensor->name ? tensor->name : "unnamed") << std::endl;
            std::cout << "  Type: " << TfLiteTypeGetName(tensor->type) << std::endl;
            std::cout << "  Allocation Type: " << allocation_type << std::endl;
            std::cout << "  Bytes: " << tensor->bytes << std::endl;
            std::cout << "  Dims: ";
            
            if (tensor->dims) {
                for (int d = 0; d < tensor->dims->size; ++d) {
                    std::cout << tensor->dims->data[d];
                    if (d < tensor->dims->size - 1) std::cout << "x";
                }
            } else {
                std::cout << "no dimensions";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nTotal tensor memory: " << total_bytes << " bytes" << std::endl;
    }

    void PrintExecutionPlan() {
        std::cout << "Nodes in the graph: " << interpreter_->nodes_size() << std::endl;

        // Print details for each node in the execution plan
        const auto& execution_plan = interpreter_->execution_plan();
        
        std::cout << "\nDetailed Node Information:" << std::endl;
        for (size_t i = 0; i < execution_plan.size(); ++i) {
            int node_index = execution_plan[i];
            const auto* node_and_reg = interpreter_->node_and_registration(node_index);
            
            // Get operation name
            std::string op_name;
            if (node_and_reg->second.custom_name) {
                op_name = node_and_reg->second.custom_name;
            } else {
                op_name = "BuiltinOp(" + std::to_string(node_and_reg->second.builtin_code) + ")";
            }
            
            std::cout << "\nNode " << i << " (Op: " << op_name << "):" << std::endl;
            
            // Print inputs
            std::cout << "  Inputs [" << node_and_reg->first.inputs->size << "]: ";
            for (int j = 0; j < node_and_reg->first.inputs->size; ++j) {
                int input_idx = node_and_reg->first.inputs->data[j];
                if (input_idx >= 0) {  // -1 indicates optional input that's not present
                    std::cout << input_idx;
                    // Print tensor name if available
                    const TfLiteTensor* tensor = interpreter_->tensor(input_idx);
                    if (tensor && tensor->name) {
                        std::cout << "(" << tensor->name << ")";
                    }
                    std::cout << " ";
                } else {
                    std::cout << "None ";
                }
            }
            
            // Print outputs
            std::cout << "\n  Outputs [" << node_and_reg->first.outputs->size << "]: ";
            for (int j = 0; j < node_and_reg->first.outputs->size; ++j) {
                int output_idx = node_and_reg->first.outputs->data[j];
                std::cout << output_idx;
                // Print tensor name if available
                const TfLiteTensor* tensor = interpreter_->tensor(output_idx);
                if (tensor && tensor->name) {
                    std::cout << "(" << tensor->name << ")";
                }
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    }

    // bool SetupGPUDelegate() {
    //     // GPU delegate options
    //     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    //     options.is_precision_loss_allowed = 0;
    //     options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

    //     // Create GPU delegate
    //     gpu_delegate_ = TfLiteGpuDelegateV2Create(&options);
    //     if (!gpu_delegate_) {
    //         std::cerr << "Failed to create GPU delegate" << std::endl;
    //         return false;
    //     }

    //     // Modify interpreter to use GPU delegate
    //     if (interpreter_->ModifyGraphWithDelegate(gpu_delegate_) != kTfLiteOk) {
    //         std::cerr << "Failed to modify graph with GPU delegate" << std::endl;
    //         return false;
    //     }

    //     std::cout << "\n=== Memory Plan After GPU Delegation ===" << std::endl;
    //     PrintMemoryPlan();

    //     std::cout << "\n=== Execution Plan After GPU Delegation ===" << std::endl;
    //     PrintExecutionPlan();

    //     std::cout << "\nGPU delegate setup successfully" << std::endl;
    //     return true;
    // }

    bool AllocateTensors() {
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors" << std::endl;
            return false;
        }

        return true;
    }

    bool Invoke() {
        if (interpreter_->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke interpreter" << std::endl;
            return false;
        }
        return true;
    }

    void PrintOutputTensor() {
        std::cout << "\n=== Output Tensor ===" << std::endl;
        auto output = interpreter_->output_tensor(0);
        
        std::cout << "Output tensor name: " << output->name << std::endl;
        std::cout << "Output tensor type: " << TfLiteTypeGetName(output->type) << std::endl;
        std::cout << "Output tensor dims: ";
        for (int i = 0; i < output->dims->size; ++i) {
            std::cout << output->dims->data[i];
            if (i < output->dims->size - 1) std::cout << "x";
        }
        std::cout << std::endl;

        // Print first few values of output tensor
        float* output_data = interpreter_->typed_output_tensor<float>(0);
        std::cout << "First 5 values: ";
        for (int i = 0; i < 5 && i < output->dims->data[0]; ++i) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;
    }

    void PrintSubgraphStructure() {
        std::cout << "\n=== Subgraph Structure ===" << std::endl;
        
        // Get total number of nodes
        int total_nodes = interpreter_->nodes_size();
        std::cout << "Total nodes: " << total_nodes << std::endl;

        // Create a map of tensor dependencies
        std::map<int, std::vector<int>> tensor_to_nodes;  // tensor_id -> vector of node indices
        std::map<int, std::vector<int>> node_inputs;      // node_id -> vector of input tensor indices
        std::map<int, std::vector<int>> node_outputs;     // node_id -> vector of output tensor indices

        // Analyze node connections
        for (int node_idx = 0; node_idx < total_nodes; ++node_idx) {
            const auto* node_and_reg = interpreter_->node_and_registration(node_idx);
            
            // Get operation name
            std::string op_name;
            if (node_and_reg->second.custom_name) {
                op_name = node_and_reg->second.custom_name;
            } else {
                op_name = "BuiltinOp(" + std::to_string(node_and_reg->second.builtin_code) + ")";
            }

            std::cout << "\nNode " << node_idx << " (" << op_name << "):" << std::endl;
            
            // Process inputs
            std::cout << "  Inputs: ";
            for (int j = 0; j < node_and_reg->first.inputs->size; ++j) {
                int input_idx = node_and_reg->first.inputs->data[j];
                if (input_idx >= 0) {
                    std::cout << input_idx;
                    const TfLiteTensor* tensor = interpreter_->tensor(input_idx);
                    if (tensor && tensor->name) {
                        std::cout << "(" << tensor->name << ")";
                    }
                    std::cout << " ";
                    
                    // Record dependency
                    tensor_to_nodes[input_idx].push_back(node_idx);
                    node_inputs[node_idx].push_back(input_idx);
                }
            }
            std::cout << std::endl;

            // Process outputs
            std::cout << "  Outputs: ";
            for (int j = 0; j < node_and_reg->first.outputs->size; ++j) {
                int output_idx = node_and_reg->first.outputs->data[j];
                std::cout << output_idx;
                const TfLiteTensor* tensor = interpreter_->tensor(output_idx);
                if (tensor && tensor->name) {
                    std::cout << "(" << tensor->name << ")";
                }
                std::cout << " ";
                
                // Record dependency
                node_outputs[node_idx].push_back(output_idx);
            }
            std::cout << std::endl;
        }

        // Print data flow analysis
        std::cout << "\n=== Data Flow Analysis ===" << std::endl;
        for (const auto& pair : tensor_to_nodes) {
            int tensor_id = pair.first;
            const auto& using_nodes = pair.second;
            
            const TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
            std::cout << "\nTensor " << tensor_id;
            if (tensor && tensor->name) {
                std::cout << " (" << tensor->name << ")";
            }
            std::cout << " is used by nodes: ";
            for (int node_idx : using_nodes) {
                std::cout << node_idx << " ";
            }
            std::cout << std::endl;
        }
    }

    ~TFLiteInferenceApp() {
        // if (gpu_delegate_) {
        //     TfLiteGpuDelegateV2Delete(gpu_delegate_);
        // }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <tflite_model_path>" << std::endl;
        return 1;
    }

    TFLiteInferenceApp app;
    std::cout << "Start!!" << std::endl;

    // Step 1: Load model
    if (!app.LoadModel(argv[1])) {
        return 1;
    }

    // Step 2: Build interpreter
    if (!app.BuildInterpreter()) {
        return 1;
    }

    app.PrintSubgraphStructure();

    // Step 3: Setup GPU delegate
    // if (!app.SetupGPUDelegate()) {
    //     return 1;
    // }

    // Step 4: Allocate tensors
    if (!app.AllocateTensors()) {
        return 1;
    }

    // Step 5: Run inference
    if (!app.Invoke()) {
        return 1;
    }

    // Step 6: Print output tensor
    app.PrintOutputTensor();

    return 0;
}