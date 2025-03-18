#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

class TFLiteInferenceApp {
private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    
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

    // Input/Output configuration
    int input_tensor_index_;
    int output_tensor_index_;
    std::vector<int> input_dims_;
    int max_sequence_length_;
    int vocab_size_;

public:
    TFLiteInferenceApp() : input_tensor_index_(0), output_tensor_index_(0), 
                          max_sequence_length_(512), vocab_size_(32000) {} // Adjust these values based on your model

    bool LoadModel(const char* model_path) {
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path, &error_reporter_);
        if (!model_) {
            std::cerr << "Failed to load model" << std::endl;
            return false;
        }
        
        std::cout << "Model loaded successfully" << std::endl;
        return true;
    }

    bool BuildInterpreter() {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model_, resolver);
        builder.SetNumThreads(4);

        if (builder.operator()(&interpreter_) != kTfLiteOk) {
            std::cerr << "Failed to build interpreter" << std::endl;
            return false;
        }

        // Get input and output tensor indices
        input_tensor_index_ = interpreter_->inputs()[0];
        output_tensor_index_ = interpreter_->outputs()[0];

        // Store input dimensions
        TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_index_);
        input_dims_.clear();
        for (int i = 0; i < input_tensor->dims->size; ++i) {
            input_dims_.push_back(input_tensor->dims->data[i]);
        }

        return true;
    }

    bool AllocateTensors() {
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors" << std::endl;
            return false;
        }
        return true;
    }

    // Convert input text to model input format
    std::vector<int> TokenizeInput(const std::string& input_text) {
        // Note: This is a placeholder tokenization.
        // You'll need to implement proper tokenization based on your model's vocabulary
        std::vector<int> tokens;
        std::istringstream iss(input_text);
        std::string word;
        while (iss >> word && tokens.size() < max_sequence_length_) {
            // Simple character-based tokenization for demonstration
            for (char c : word) {
                tokens.push_back(static_cast<int>(c) % vocab_size_);
            }
            tokens.push_back(32); // Space token
        }
        return tokens;
    }

    // Convert model output to text
    std::string DetokenizeOutput(float* output_data, int output_size) {
        // Note: This is a placeholder detokenization.
        // You'll need to implement proper detokenization based on your model's vocabulary
        std::stringstream result;
        for (int i = 0; i < output_size; ++i) {
            // Simple character-based detokenization for demonstration
            if (output_data[i] > 0.5) {  // threshold for demonstration
                result << static_cast<char>(i % 128);
            }
        }
        return result.str();
    }

    std::string ProcessUserInput(const std::string& input_text) {
        // 1. Tokenize input
        std::vector<int> input_tokens = TokenizeInput(input_text);
        
        // 2. Prepare input tensor
        TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_index_);
        int* input_data = interpreter_->typed_input_tensor<int>(0);
        
        // Copy tokens to input tensor
        for (size_t i = 0; i < input_tokens.size() && i < max_sequence_length_; ++i) {
            input_data[i] = input_tokens[i];
        }

        // 3. Run inference
        if (interpreter_->Invoke() != kTfLiteOk) {
            return "Error during inference";
        }

        // 4. Process output
        float* output_data = interpreter_->typed_output_tensor<float>(0);
        TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_index_);
        int output_size = output_tensor->dims->data[output_tensor->dims->size - 1];
        
        // 5. Convert output to text
        return DetokenizeOutput(output_data, output_size);
    }

    void RunInteractiveSession() {
        std::cout << "Interactive session started. Type 'exit' to quit.\n" << std::endl;
        
        std::string input_text;
        while (true) {
            std::cout << "\nUser: ";
            std::getline(std::cin, input_text);
            
            if (input_text == "exit") {
                break;
            }
            
            std::string response = ProcessUserInput(input_text);
            std::cout << "Model: " << response << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <tflite_model_path>" << std::endl;
        return 1;
    }

    TFLiteInferenceApp app;
    
    // Initialize the application
    if (!app.LoadModel(argv[1])) {
        return 1;
    }

    if (!app.BuildInterpreter()) {
        return 1;
    }

    if (!app.AllocateTensors()) {
        return 1;
    }

    // Start interactive session
    app.RunInteractiveSession();

    return 0;
}