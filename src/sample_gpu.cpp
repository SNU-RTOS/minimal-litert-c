#include <cstdio>
#include <memory>

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: sample <tflite model>\n");
    return 1;
  }

  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Create interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // === GPU Delegate 추가 ===
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  // 원하는 경우 옵션을 조정할 수 있음
  gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;

  TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
  TFLITE_MINIMAL_CHECK(gpu_delegate != nullptr);

  if (interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk) {
    fprintf(stderr, "Failed to apply GPU delegate.\n");
    TfLiteGpuDelegateV2Delete(gpu_delegate);
    return 1;
  }

  // Allocate tensors
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // TODO: Fill input tensors
  // T* input = interpreter->typed_input_tensor<T>(i);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // TODO: Read output tensors
  // T* output = interpreter->typed_output_tensor<T>(i);

  // Cleanup delegate
  TfLiteGpuDelegateV2Delete(gpu_delegate);

  return 0;
}
