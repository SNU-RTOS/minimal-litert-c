// xnn-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "opencv2/opencv.hpp" //opencv

#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"

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
    util::timer_start("Load Model"); //! Metrics (timer_start)



    util::timer_stop("Load Model"); //! Metrics (timer_stop)

    /* Build interpreter */
    util::timer_start("Build Interpreter"); //! Metrics (timer_start)



    util::timer_stop("Build Interpreter"); //! Metrics (timer_stop)

    /* Apply XNNPACK delegate */
    util::timer_start("Apply Delegate"); //! Metrics (timer_start)



    util::timer_stop("Apply Delegate"); //! Metrics (timer_stop)

    /* Allocate Tensor */
    util::timer_start("Allocate Tensor"); //! Metrics (timer_start)



    util::timer_stop("Allocate Tensor"); //! Metrics (timer_stop)
    util::print_model_summary(interpreter.get(), delegate_applied);

    /* Load input image */
    util::timer_start("Load Input Image"); //! Metrics (timer_start)



    util::timer_stop("Load Input Image"); //! Metrics (timer_stop)

    /* Preprocessing */
    util::timer_start("E2E Total(Pre+Inf+Post)"); //! Metrics (timer_start)
    util::timer_start("Preprocessing");           //! Metrics (timer_start)

    // Get input tensor info
    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;

    // Preprocess input data
    cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);

    // Copy HWC float32 cv::Mat to TFLite input tensor
    util::timer_stop("Preprocessing"); //! Metrics (timer_stop)


    /* Inference */
    util::timer_start("Inference"); //! Metrics (timer_start)



    util::timer_stop("Inference"); //! Metrics (timer_stop)

    /* PostProcessing */
    util::timer_start("Postprocessing"); //! Metrics (timer_start)

    
    // Get output tensor

    util::timer_stop("Postprocessing");          //! Metrics  (timer_stop)
    util::timer_stop("E2E Total(Pre+Inf+Post)"); //! Metrics (timer_stop)

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
    util::print_all_timers(); //! Metrics (print timers)
    std::cout << "========================" << std::endl;
    
    /* Deallocate delegate */

    return 0;
}
