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

#include "tensorflow/compiler/mlir/lite/version.h" // TFLITE_SCHEMA_VERSION is defined inside

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
    /* The model file is mapped to the user-space memory of the process */
    // Use 


    /* Build interpreter */
    util::timer_start("Build Interpreter");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    /* Main data structures of the interpreter builder */
    // Simply, the _model variable and _resolver variable of the builder object is set as model and resolver, respectively.
    tflite::InterpreterBuilder builder(*model, resolver);
    
    /* Main data structures before the operator of the interpreter builder */
    std::unique_ptr<tflite::Interpreter> interpreter; // interpreter is a nullptr

    // Mapping between the operations in the model and the resolver is done and it is saved as an node_and_registration variable in the interpreter.
    // Subgraphs, nodes, tensors, and execution plan are created.
    builder(&interpreter);
    /* ======================================================================================================== */
    /* A simple code snippet for simulating what happens when the operator of the interpreter builder is called */
    // 1. Model Validation

    // Get the root object of the FlatBuffer model.
    // This provides access to the model structure (e.g., subgraphs, tensors, operators)
    const tflite::Model* fb_model = model->GetModel(); // fb means flatbuffer
    std::cout << "Schema version of the model: " << fb_model->version() 
    << "\nSupported schema version: " << TFLITE_SCHEMA_VERSION << std::endl;

    // 2. Operator mapping
    const auto* op_codes = fb_model->operator_codes(); // It is a vector of tflite::OperatorCode
    std::cout << "\nTotal " << op_codes->size() << " operators in the model" << std::endl;

    for (int i = 0; i < op_codes->size(); i++) {
        const auto* opcode = op_codes->Get(i); // The i th operator code in the op_codes
        auto builtin_code = opcode->builtin_code(); // An enum indicating the type of the operator like CONV_2D, RELU, etc.
        std::string op_name = tflite::EnumNameBuiltinOperator(builtin_code);
        int op_version = opcode->version(); // Version of the operator
        const TfLiteRegistration* reg = resolver.FindOp(builtin_code, op_version); // Checks whether the OpResolver supports the operator
        
        std::cout << "[" << i << "] " << op_name << ", version: " << op_version 
        << ", supported: " << (reg ? "Y" : "N") << std::endl;
    }

    // 3. Internal data instantiation
    // 3-1. Extracts subgraph information from the model
    const auto* subGraphs = fb_model->subgraphs();
    std::cout << "\nNumber of subgraphs: " << subGraphs->size() << std::endl;
    for( int i = 0; i < subGraphs->size(); i++){
        // Note: tflite::SubGraph is for FlatBuffer serialized subgraph info
        // and tflite::Subgraph is for subgraph class that the interpreter uses
        const tflite::SubGraph* subGraph = subGraphs->Get(i); // Gets the i th SubGraph of the model
        std::cout << "SubGraph [" << i << "] " 
            << (subGraph->name() ? subGraph->name()->str() : "(unnamed)") << std::endl;
        // The space for subgraphs are reserved in the interpreter

        // 3-2. Parse tensor information from the buffer information in the SubGraph
        // and set the tensor variables in the subgraph
        const auto* buffers = fb_model->buffers(); // Global raw data about weights, bias, and others, shared across subgraphs
        const auto* tensors = subGraph->tensors(); // Tensor data structure that contains shape, type, pointer to a buffer. Not shared across subgraphs
        
    }


    // 3-3. Parses node information in the subgraph, which is a vector of operators in execution order.
    // Then sets 

    // Now let's check the interpreter
    std::cout << "Number of subgraphs: " << interpreter->subgraphs_size() << std::endl;
    std::cout << "Number of nodes: " << interpreter->nodes_size() << std::endl;
    std::cout << "Execution plan size: " << interpreter->execution_plan().size() << std::endl;
    util::print_execution_plan(interpreter);
    /* ======================================================================================================== */

    util::timer_stop("Build Interpreter");

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
    // Is there a way that we can check whether the delegate is applied or not?
    // Does the interpreter go through the execution plan or the nodes?
    util::print_execution_plan(interpreter);

    util::timer_stop("Apply Delegate");


    /* Allocate Tensor */
    // Types of tensors, what happens inside it
    // How the memory space for ArenaRW tensors changes after AllocateTensors() is called
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