#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H
#define STB_IMAGE_IMPLEMENTATION

#include <queue>  // Added to fix std::queue usage
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>
#include <thread>
#include <random>
#include <condition_variable>
#include "stb_image.h"

// Generating random tensor as input
std::vector<float> preprocess_image(const std::string& image_path, bool debug_output = false) {
    // Load image using stb_image
    int img_width, img_height, img_channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &img_width, &img_height, &img_channels, 3); // force RGB

    int height = 224;
    int width = 224;
    int channels = 3;

    // Resize image using simple bilinear interpolation
    std::vector<unsigned char> resized_img(width * height * 3);
    for (int y = 0; y < height; ++y) {
        float in_y = (float)y / height * img_height;
        int y0 = std::min((int)in_y, img_height - 1);
        for (int x = 0; x < width; ++x) {
            float in_x = (float)x / width * img_width;
            int x0 = std::min((int)in_x, img_width - 1);
            for (int c = 0; c < 3; ++c) {
                resized_img[(y * width + x) * 3 + c] = img_data[(y0 * img_width + x0) * 3 + c];
            }
        }
    }

    stbi_image_free(img_data); // free original image

    // Convert to float and subtract mean
    const float mean[3] = {103.939f, 116.779f, 123.68f}; // BGR
    std::vector<float> buffer(width * height * 3);
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                int idx = (y * width + x) * 3 + c;
                float val = static_cast<float>(resized_img[idx]) - mean[c];
                buffer[idx] = val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
    }

    if (debug_output) {
        std::cout << "Image preprocessed (no OpenCV): " << image_path << std::endl;
        std::cout << "Input shape: " << height << "x" << width << "x" << channels << std::endl;
        std::cout << "Pixel value range after normalization: [" << min_val << ", " << max_val << "]" << std::endl;
    }

    return buffer;
}

bool preprocess_image_rd(const std::string& image_path, tflite::Interpreter* interpreter, bool debug_output = false) {
    // 입력 텐서 가져오기
    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    if (!input_tensor) {
        std::cerr << "Failed to get input tensor\n";
        return false;
    }

    // 텐서 차원 (NHWC 형태라고 가정)
    int height   = input_tensor->dims->data[1];
    int width    = input_tensor->dims->data[2];
    int channels = input_tensor->dims->data[3];
    int num_elements = height * width * channels;

    // 입력 버퍼에 직접 접근
    float* input_data = interpreter->typed_input_tensor<float>(0);
    if (!input_data) {
        std::cerr << "Failed to access input tensor buffer\n";
        return false;
    }

    // 랜덤 엔진 및 분포 설정 (0.0 ~ 255.0 사이)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    // 최소/최대 값 추적용 변수
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    // 버퍼를 랜덤 값으로 채우기
    for (int i = 0; i < num_elements; ++i) {
        float v = dist(gen);
        input_data[i] = v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    if (debug_output) {
        std::cout << "Generated random input tensor\n";
        std::cout << "Input shape: " << height << " x " << width << " x " << channels << "\n";
        std::cout << "Value range: [" << min_val << ", " << max_val << "]\n";
    }

    return true;
}
// Preprocessing using OpenCV
// bool preprocess_image_CV(const std::string& image_path, tflite::Interpreter* interpreter, bool debug_output = false) {
//     cv::Mat image = cv::imread(image_path);
//     if (image.empty()) {
//         std::cerr << "Failed to load image: " << image_path << std::endl;
//         return false;
//     }
//     TfLiteTensor* input_tensor = interpreter->input_tensor(0);
//     int height = input_tensor->dims->data[1];
//     int width = input_tensor->dims->data[2];
//     int channels = input_tensor->dims->data[3];
//     cv::Mat resized;
//     cv::resize(image, resized, cv::Size(width, height));
//     resized.convertTo(resized, CV_32FC3);
//     const float mean[3] = {103.939f, 116.779f, 123.68f}; // BGR

//     std::vector<float> buffer(height * width * channels);
//     float min_val = std::numeric_limits<float>::max();
//     float max_val = std::numeric_limits<float>::lowest();
    
    
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; ++x) {
//             const cv::Vec3f& pixel = resized.at<cv::Vec3f>(y, x);
//             for (int c = 0; c < channels; ++c) {
//                 int idx = y * width * channels + x * channels + c;
//                 float val = pixel[c] - mean[c];  // Zero-center
//                 buffer[idx] = val;
//                 min_val = std::min(min_val, val);
//                 max_val = std::max(max_val, val);
//             }
//         }
//     }
//     // std::cout << "Normalized RGB image range: [" << min_val << ", " << max_val << "]" << std::endl;
//     float* input_data = interpreter->typed_input_tensor<float>(0);
//     if (!input_data) {
//         std::cerr << "Failed to get input tensor data" << std::endl;
//         return false;
//     }
//     std::memcpy(input_data, buffer.data(), buffer.size() * sizeof(float));

//     return true;
// }

// Post-processing function
void print_top_predictions(const std::vector<float>& output_data, int num_classes, int top_n = 5, bool debug_output = false) {
    if (output_data.empty() || num_classes <= 0) {
        std::cerr << "Invalid output data or number of classes" << std::endl;
        return;
    }

    // Print raw output stats for debugging
    float min_raw = *std::min_element(output_data.begin(), output_data.end());
    float max_raw = *std::max_element(output_data.begin(), output_data.end());
    if(debug_output)
        std::cout << "Raw output range: " << min_raw << " to " << max_raw << std::endl;
    
    // Check if the model outputs are already probabilities
    bool already_probabilities = true;
    float sum_outputs = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        sum_outputs += output_data[i];
    }
    
    // If sum is very close to 1.0, the model might already output probabilities
    if(debug_output){
        if (std::abs(sum_outputs - 1.0f) < 0.01f) {
            std::cout << "Sum of raw outputs is " << sum_outputs << " - model may already output probabilities" << std::endl;
            already_probabilities = true;
        }
    }

    // Print first few raw values
    if(debug_output){
        std::cout << "First 5 raw values: ";
        for (int i = 0; i < std::min(5, num_classes); i++) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> probabilities(num_classes);
    
    // Apply softmax if needed
    if (already_probabilities) {
        // Just copy the probabilities directly
        std::copy(output_data.begin(), output_data.begin() + num_classes, probabilities.begin());
    } else {
        // Find the maximum value to subtract (prevents overflow in exponential)
        float max_val = *std::max_element(output_data.begin(), output_data.begin() + num_classes);
        
        // Apply softmax with numerical stability improvement
        float sum_exp = 0.0f;
        
        for (int i = 0; i < num_classes; i++) {
            probabilities[i] = std::exp(output_data[i] - max_val);
            sum_exp += probabilities[i];
        }
        
        // Check if sum_exp is too small or too large
        if (sum_exp < 1e-10f || sum_exp > 1e10f) {
            std::cout << "Warning: sum_exp is " << sum_exp << " - numerical issue detected" << std::endl;
        }
        
        // Normalize by sum
        for (int i = 0; i < num_classes; i++) {
            probabilities[i] /= sum_exp;
        }
    }

    // Verify probabilities sum to 1
    {
        float sum_prob = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_prob += probabilities[i];
        }

        // Find top predictions
        std::vector<std::pair<float, int>> scores;
        scores.reserve(num_classes);
        for (int i = 0; i < num_classes; i++) {
            scores.push_back({probabilities[i], i});
        }

        // Sort in descending order of probability
        std::sort(scores.begin(), scores.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
    
        // Output top predictions
        if(debug_output){
            std::cout << "Top " << top_n << " predictions:" << std::endl;
            for (int i = 0; i < std::min(top_n, num_classes); i++) {
                std::cout << "  Class " << scores[i].second << ": "
                        << (scores[i].first * 100.0f) << "%" << std::endl;
            }
        }
    }
}

// Function to display tensor information for a given interpreter
void displayModelInfo(tflite::Interpreter* interpreter, const std::string& modelName) {
    if (!interpreter) {
        std::cout << "Error: Null interpreter for " << modelName << std::endl;
        return;
    }

    std::cout << modelName << " Information:" << std::endl;
    
    // Display input tensors info
    size_t num_inputs = interpreter->inputs().size();
    std::cout << "Number of Inputs: " << num_inputs << std::endl;
    
    for (int i = 0; i < num_inputs; i++) {
        TfLiteTensor* input_tensor = interpreter->input_tensor(i);
        
        std::cout << "  Input " << i << " shape: ";
        for (int j = 0; j < input_tensor->dims->size; j++) {
            std::cout << input_tensor->dims->data[j];
            if (j < input_tensor->dims->size - 1)
                std::cout << "x";
        }
        std::cout << " (size: " << input_tensor->bytes << " bytes)";
        std::cout << std::endl;
    }
    
    // Display output tensors info
    size_t num_outputs = interpreter->outputs().size();
    std::cout << "Number of Outputs: " << num_outputs << std::endl;
    
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* output_tensor = interpreter->output_tensor(i);
        
        std::cout << "  Output " << i << " shape: ";
        for (int j = 0; j < output_tensor->dims->size; j++) {
            std::cout << output_tensor->dims->data[j];
            if (j < output_tensor->dims->size - 1)
                std::cout << "x";
        }
        std::cout << " (size: " << output_tensor->bytes << " bytes)";
        std::cout << std::endl;
    }
    
    std::cout << std::endl; // Add blank line between models
}

// Input rate control class
class InputRateController {
private:
    std::chrono::milliseconds rate_ms;
    std::chrono::steady_clock::time_point last_process_time;

public:
    InputRateController(int rate_ms_val) : rate_ms(rate_ms_val) {
        last_process_time = std::chrono::steady_clock::now();
    }

    void wait_for_next_input() {
        if (rate_ms.count() <= 0) return; // Skip if no rate limiting

        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_process_time);
        
        // If not enough time has passed, sleep
        if (elapsed < rate_ms) {
            auto sleep_time = rate_ms - elapsed;
            std::this_thread::sleep_for(sleep_time);
        }
        
        last_process_time = std::chrono::steady_clock::now();
    }
};

// Thread-safe queue for passing data between threads
template <typename T>
class ThreadSafeQueue
{
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond_var;
    std::atomic<bool> shutdown{false};

public:
    void push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(std::move(item));
        lock.unlock();
        cond_var.notify_one();
    }

    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_var.wait(lock, [this]
                      { return !queue.empty() || shutdown; });

        if (shutdown && queue.empty())
        {
            return false;
        }

        item = std::move(queue.front());
        queue.pop();
        return true;
    }

    void signal_shutdown()
    {
        std::unique_lock<std::mutex> lock(mutex);
        shutdown = true;
        lock.unlock();
        cond_var.notify_all();
    }

    size_t size()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }
};

// Structure to hold an image job with timing information
struct ImageJob {
    std::string image_path;                                 // Path to the image
    int index;                                              // Index in the original list
    std::chrono::high_resolution_clock::time_point queued_time; // Time when the job was queued
};

// Input preparation worker class that queues inputs according to input rate
class InputPreparationWorker {
private:
    ThreadSafeQueue<ImageJob>& job_queue;
    std::vector<std::string> image_paths;
    int input_rate_ms;
    std::thread worker_thread;
    std::atomic<bool> running{false};
    std::vector<std::chrono::high_resolution_clock::time_point> queue_times;

public:
    InputPreparationWorker(ThreadSafeQueue<ImageJob>& queue, 
                          const std::vector<std::string>& paths,
                          int rate_ms) 
        : job_queue(queue), image_paths(paths), input_rate_ms(rate_ms) {
        queue_times.resize(paths.size());
    }

    void start() {
        running = true;
        worker_thread = std::thread(&InputPreparationWorker::run, this);
    }

    void stop() {
        running = false;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    const std::vector<std::chrono::high_resolution_clock::time_point>& get_queue_times() const {
        return queue_times;
    }

private:
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < image_paths.size() && running; i++) {
            if (input_rate_ms > 0 && i > 0) {
                // Calculate when this image should be added based on input rate
                auto scheduled_time = start_time + std::chrono::milliseconds(i * input_rate_ms);
                auto now = std::chrono::high_resolution_clock::now();
                
                // Wait until the scheduled time if needed
                if (now < scheduled_time) {
                    std::this_thread::sleep_for(scheduled_time - now);
                }
            }
            
            // Record the time when this image is queued
            auto current_time = std::chrono::high_resolution_clock::now();
            queue_times[i] = current_time;
            
            // Create and queue the job
            ImageJob job;
            job.image_path = image_paths[i];
            job.index = i;
            job.queued_time = current_time;
            job_queue.push(job);
        }
        
        // Signal end of queue
        job_queue.signal_shutdown();
    }
};

#endif // MODEL_UTILS_H