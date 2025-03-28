#ifndef _UTIL_H_
#define _UTIL_H_

#include <jsoncpp/json/json.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace util
{
    //**** For Section  2.4 ****/
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    struct TimerResult
    {
        TimePoint start;
        TimePoint end;
        int start_index;
        int stop_index;
    };

    static std::unordered_map<std::string, TimerResult> timer_map;
    static int global_index = 0;

    void timer_start(const std::string &label);
    void timer_stop(const std::string &label);
    void print_all_timers();
    //*==========================================*/

    // Loads class labels from a JSON file, expects JSON format like: { "0": ["n01440764", "tench"], ... }
    std::unordered_map<int, std::string> load_class_labels(const std::string &json_path);

    // Print shape of tensor
    void print_tensor_shape(const TfLiteTensor *tensor);

    // Print model summary
    void print_model_summary(tflite::Interpreter *interpreter, bool delegate_applied);

    // Get TopK indices of probs
    std::vector<int> get_topK_indices(const std::vector<float> &data, int k);

} // namespace util

#endif // _UTIL_H_
