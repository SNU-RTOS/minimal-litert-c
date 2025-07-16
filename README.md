# minimal-litert-c

A minimal C++ example project for LiteRT (formerly TensorFlow Lite) inference with CPU, GPU, and QNN delegate support.

## Overview

This project demonstrates how to build and run TensorFlow Lite models using the LiteRT C++ API with multiple hardware acceleration options:

- **CPU inference** with XNNPACK delegate
- **GPU inference** with GPU delegate
- **QNN inference** with Qualcomm Neural Network delegate

## Features

- ✅ CPU inference with XNNPACK optimization
- ✅ GPU acceleration support
- ✅ QNN (Qualcomm Neural Network) support
- ✅ OpenCV integration for image processing
- ✅ JSON-based label management
- ✅ Automated build and test scripts
- ✅ Model verification utilities

## Project Structure

```
├── src/                    # Source code
│   ├── main_cpu.cpp       # CPU inference example
│   ├── main_gpu.cpp       # GPU inference example
│   ├── main_qnn.cpp       # QNN inference example
│   ├── verify_cpu.cpp     # CPU verification utility
│   ├── verify_gpu.cpp     # GPU verification utility
│   ├── verify_qnn.cpp     # QNN verification utility
│   ├── util.cpp           # Utility functions
│   └── util.hpp           # Utility headers
├── models/                 # TensorFlow Lite models
│   ├── mobilenetv3_small.tflite
│   └── mobileone_s0.tflite
├── images/                 # Test images
├── scripts/               # Build and setup scripts
│   ├── build-litert.sh
│   ├── build-litert_gpu_delegate.sh
│   └── install_prerequistes.sh
├── Makefile_*             # Makefiles for different targets
├── setup.sh               # Environment setup script
└── build_and_run.sh       # Build and run automation script
```

## Prerequisites

### System Requirements
- Ubuntu 20.04/22.04/24.04 
- Clang  
- Bazel
- Git

### Dependencies
Run the prerequisite installation script:
```bash
./scripts/install_prerequistes.sh
```

This installs:
- Build tools
- OpenCV development libraries
- JSON libraries (jsoncpp)
- Python development tools
- Bazel build system

## Setup and Installation

### 1. Environment Setup
```bash
# Create and configure .env file with your paths
# Example .env content:
# ROOT_PATH=/path/to/minimal-litert-c
# EXTERNAL_PATH=${ROOT_PATH}/external
# LITERT_PATH=${EXTERNAL_PATH}/LiteRT

# Run setup script
./setup.sh
```

### 2. Build LiteRT
The setup script automatically:
- Clones the LiteRT repository
- Configures the build environment
- Builds the core LiteRT library
- Builds GPU delegate library

## Usage

### Quick Start
```bash
# Build and run CPU inference example
./build_and_run.sh
```

### Manual Build and Run

#### CPU Inference
```bash
# Build CPU version
make -f Makefile_main_cpu -j4

# Run CPU inference
./output/main_cpu ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json
```

#### GPU Inference
```bash
# Build GPU version
make -f Makefile_main_gpu -j4

# Run GPU inference
./output/main_gpu ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json
```

#### QNN Inference
```bash
# Build QNN version
make -f Makefile_main_qnn -j4

# Run QNN inference
./output/main_qnn ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json
```

### Model Verification
Verify that models can be loaded and basic inference works:

```bash
# Verify CPU
make -f Makefile_verify_cpu -j4
./output/verify_cpu ./models/mobileone_s0.tflite

# Verify GPU
make -f Makefile_verify_gpu -j4
./output/verify_gpu ./models/mobileone_s0.tflite

# Verify QNN
make -f Makefile_verify_qnn -j4
./output/verify_qnn ./models/mobileone_s0.tflite
```

# Run Pipelinging
```bash
python model_partitioner.py --model-path ./ --config ./config --model-name efficientnetv2b0 --output-dir ./submodels
./output/pipeline models/sub_model_1.tflite models/sub_model_2.tflite images/_images_68.png 
```

## Supported Models

Currently tested with:
- MobileNetV3 Small
- MobileOne S0

The project supports standard TensorFlow Lite models (.tflite format).

## Output

The application performs image classification and outputs:
- Model loading status
- Input/output tensor information
- Inference timing
- Top-5 predictions with confidence scores

Example output:
```
====== main_cpu ====
🔍 Loading model from: ./models/mobileone_s0.tflite
📊 Model loaded successfully
⚡ Inference time: 15.2ms
🏆 Top predictions:
1. Golden retriever (85.4%)
2. Labrador retriever (12.1%)
3. Nova Scotia duck tolling retriever (1.8%)
```



## Development

### Adding New Models
1. Place your `.tflite` model in the `models/` directory
2. Update the `labels.json` file if needed
3. Modify the input preprocessing in `util.cpp` if required

### Debugging
- Set `TF_CPP_MIN_LOG_LEVEL=0` for verbose logging
- Use the verify utilities to test model compatibility
- Check build logs for compilation issues

## Troubleshooting

### Common Issues

1. **Build Errors**: Ensure all prerequisites are installed
2. **Model Loading Fails**: Check model path and format
3. **GPU Delegate Issues**: Verify GPU drivers and OpenCL support
4. **Missing Libraries**: Run `ldconfig` after installation

### Environment Variables
Create a `.env` file in the project root:
```bash
ROOT_PATH=/path/to/your/project
EXTERNAL_PATH=${ROOT_PATH}/external
LITERT_PATH=${EXTERNAL_PATH}/LiteRT
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with all delegate types
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## References

- [LiteRT Documentation](https://ai.google.dev/edge/litert)
- [TensorFlow Lite C++ API](https://www.tensorflow.org/lite/api_docs/cc)
- [OpenCV Documentation](https://docs.opencv.org/)
