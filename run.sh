#!/bin/bash
echo "[INFO] Build sample_cpu"
make -f Makefile-cpu -j4
echo "[INFO] Run sample_cpu"
./output/sample_cpu ./models/mobilenetv3_small.tflite > sample_cpu_mobilenetv3.log

echo "[INFO] Build sample_gpu"
make -f Makefile-gpu -j4
echo "[INFO] Run sample_gpu"
./output/sample_gpu ./models/mobilenetv3_small.tflite > sample_gpu_mobilenetv3.log

# make -f Makefile_main_cpu -j4
# echo "[INFO] Run main"
# ./output/main_cpu \
#     ./models/resnet34.tflite \
#     ./images/dog.jpg \
#     ./labels.json | tee output_main_cpu.log

