#include <cstdio>
#include <memory>
#include <vector>
#include <cstdlib>

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

#include "TFLiteDelegate/QnnTFLiteDelegate.h" // for QNN delegate

#define TFLITE_MINIMAL_CHECK(x)                                     \
    if (!(x))                                                       \
    {                                                               \
        fprintf(stderr, "❌ Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                    \
    }

int main(int argc, char *argv[])
{
    setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <tflite model>\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];
    printf("🔍 Loading model from: %s\n", filename);

    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Create QNN Delegate options structure.
    TfLiteQnnDelegateOptions options = TfLiteQnnDelegateOptionsDefault();

    // Set the mandatory backend_type option. All other options have default values.
    // options.backend_type = kHtpBackend; //	Qualcomm Hexagon Tensor Processor (HTP), 고성능 NPU backend
    options.backend_type = kGpuBackend; // GPU backend 
    // options.backend_type = kDspBackend; //Hexagon DSP backend (HTP보다 일반적 DSP 오프로드용)
    TfLiteDelegate *qnn_delegate = TfLiteQnnDelegateCreate(&options);
    bool delegate_applied = false;

    if (qnn_delegate)
    {
        if (interpreter->ModifyGraphWithDelegate(qnn_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
        }
    }

    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    printf("\n=== Model Summary ===\n");
    printf("📥 Input tensor count  : %zu\n", interpreter->inputs().size());
    printf("📤 Output tensor count : %zu\n", interpreter->outputs().size());
    printf("📦 Total tensor count  : %ld\n", interpreter->tensors_size());
    printf("🔧 Node (op) count     : %zu\n", interpreter->nodes_size());
    printf("🧩 QNN Delegate applied: %s\n", delegate_applied ? "Yes ✅" : "No ❌");

    if (interpreter->Invoke() == kTfLiteOk)
    {
        printf("🚀 Inference completed successfully.\n");
    }
    else
    {
        printf("❌ Inference failed.\n");
    }

    TfLiteQnnDelegateDelete(qnn_delegate);
    return 0;
}
