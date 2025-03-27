#!/bin/bash
echo "[INFO] Build sample_cpu"
make -f Makefile-cpu -j4
echo "[INFO] Run sample_cpu"
./output/sample_cpu ./models/mobileone_s0.apple_in1k.tflite > sample_cpu_mobilenetv3.log

echo "[INFO] Build sample_gpu"
make -f Makefile-gpu -j4
echo "[INFO] Run sample_gpu"
./output/sample_gpu ./models/mobileone_s0.apple_in1k.tflite > sample_gpu_mobilenetv3.log

make -j4
echo "[INFO] Run cv_cpu"
./output/cv_cpu ./models/mobileone_s0.apple_in1k.tflite ./images/dog.jpg
