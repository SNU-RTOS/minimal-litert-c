#!/bin/bash
# echo "[INFO] Build sample_cpu"
# make -f Makefile-cpu -j4
# echo "[INFO] Run sample_cpu"
# ./output/sample_cpu ./models/mobilenetv3_small.tflite > sample_cpu_mobilenetv3.log

# echo "[INFO] Build sample_gpu"
# make -f Makefile-gpu -j4
# echo "[INFO] Run sample_gpu"
# ./output/sample_gpu ./models/mobilenetv3_small.tflite > sample_gpu_mobilenetv3.log

# make -f Makefile_main_cpu -j4
# echo "[INFO] Run main_cpu"
# ./output/main_cpu \
#     ./models/resnet34.tflite \
#     ./images/dog.jpg \
#     ./labels.json | tee output_main_cpu.log

# make -f Makefile_main_gpu -j4
# echo "[INFO] Run main_gpu"
# ./output/main_gpu \
#     ./models/resnet34.tflite \
#     ./images/dog.jpg \
#     ./labels.json | tee output_main_gpu.log

# make -f Makefile_main_cpu_metric -j4
# echo "[INFO] Run main_cpu_metric"
# ./output/main_cpu_metric \
#     ./models/resnet34.tflite \
#     ./images/dog.jpg \
#     ./labels.json  | tee output_main_cpu_metric.log

make -f Makefile_main_gpu_metric -j4
echo "[INFO] Run main_gpu_metric"
./output/main_gpu_metric \
    ./models/resnet18_keras_hub.tflite \
    ./images/dog.jpg \
    ./labels.json | tee output_main_gpu_metric.log


make -f Makefile_main_cpu_keras -j4
echo "[INFO] Run main_cpu_keras"
./output/main_cpu_keras \
    ./models/resnet50-keras-application.tflite \
    ./images/dog.jpg \
    ./labels.json | tee output_main_cpu_keras.log

