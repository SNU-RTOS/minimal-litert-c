#!/bin/bash
LOGFILE=verify_cpu_mobileone_s0.log
exec > >(tee "$LOGFILE") 2>&1
echo "[INFO] Build verify_cpu"
make -f Makefile_verify_cpu -j4
echo "[INFO] Run verify_cpu"
./output/verify_cpu ./models/mobileone_s0.tflite

LOGFILE=verify_gpu_mobileone_s0.log
exec > >(tee "$LOGFILE") 2>&1
echo "[INFO] Build verify_gpu"
make -f Makefile_verify_gpu -j4
echo "[INFO] Run verify_gpu"
./output/verify_gpu ./models/mobileone_s0.tflite 

# make -f Makefile_main_cpu -j4
# echo "[INFO] Run main_cpu"
# ./output/main_cpu \
#     ./models/resnet34.tflite \
#     ./images/dog.jpg \
#     ./labels.json | tee output_main_cpu.log

