#include <cstdio>
#include <memory>
#include <vector>
#include <cstdlib>

#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/delegates/gpu/delegate.h"

#define TFLITE_MINIMAL_CHECK(x)                                     \
    if (!(x))                                                       \
    {                                                               \
        fprintf(stderr, "❌ Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                    \
    }

void PrintTopKPredictions(const TfLiteTensor* output_tensor, int top_k = 5)
{
    const float* scores = output_tensor->data.f;
    int num_classes = output_tensor->dims->data[output_tensor->dims->size - 1];

    // Pair of <score, index>
    std::vector<std::pair<float, int>> score_index_pairs;
    for (int i = 0; i < num_classes; ++i)
    {
        score_index_pairs.emplace_back(scores[i], i);
    }

    // Sort descending by score
    std::partial_sort(
        score_index_pairs.begin(), score_index_pairs.begin() + top_k, score_index_pairs.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

    printf("\n🔝 Top %d Predictions:\n", top_k);
    for (int i = 0; i < top_k && i < num_classes; ++i)
    {
        printf("  #%d: Class %d => Score: %.6f\n", i + 1, score_index_pairs[i].second, score_index_pairs[i].first);
    }
}

int main(int argc, char *argv[])
{
    printf("====== verify_gpu ====\n");
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

    // Apply GPU delegate (OpenCL)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    TfLiteDelegate *gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
    bool delegate_applied = false;

    if (gpu_delegate)
    {
        if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk)
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
    printf("🧩 GPU Delegate applied: %s\n", delegate_applied ? "Yes ✅" : "No ❌");

    if (interpreter->Invoke() == kTfLiteOk)
    {
        printf("🚀 Inference completed successfully.\n");
    }
    else
    {
        printf("❌ Inference failed.\n");
    }

    TfLiteGpuDelegateV2Delete(gpu_delegate);
    return 0;
}
