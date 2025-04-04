#!/bin/bash
echo "[INFO] Build verify_cpu"
make -f Makefile_verify_cpu -j4
echo "[INFO] Run verify_cpu"
./output/verify_cpu ./models/mobileone_s0.tflite 2>&1 |
    tee verify_cpu_mobileone_s0.log

echo "[INFO] Build verify_gpu"
make -f Makefile_verify_gpu -j4
echo "[INFO] Run verify_gpu"
./output/verify_gpu ./models/mobileone_s0.tflite 2>&1 |
    tee verify_gpu_mobileone_s0.log

echo "[INFO] Build verify_qnn"
make -f Makefile_verify_qnn -j4
echo "[INFO] Run verify_qnn"
sudo ./output/verify_qnn ./models/mobileone_s0.tflite 2>&1 |
    tee verify_qnn_mobileone_s0.log
# strace -e open,openat,access ./output/verify_qnn ./models/mobileone_s0.tflite |
#     tee verify_qnn_mobileone_s0.log

make -f Makefile_main_cpu -j4
echo "[INFO] Run main_cpu"
./output/main_cpu \
    ./models/mobileone_s0.tflite \
    ./images/dog.jpg \
    ./labels.json 2>&1 | tee output_main_cpu.log

make -f Makefile_main_gpu -j4
echo "[INFO] Run main_gpu"
./output/main_gpu \
    ./models/mobileone_s0.tflite \
    ./images/dog.jpg \
    ./labels.json 2>&1 | tee output_main_gpu.log

make -f Makefile_main_cpu_metric -j4
echo "[INFO] Run main_cpu_metric"
./output/main_cpu_metric \
    ./models/mobileone_s0.tflite \
    ./images/dog.jpg \
    ./labels.json 2>&1 | tee output_main_cpu_metric.log

make -f Makefile_main_gpu_metric -j4
echo "[INFO] Run main_gpu_metric"
./output/main_gpu_metric \
    ./models/mobileone_s0.tflite \
    ./images/dog.jpg \
    ./labels.json 2>&1 | tee output_main_gpu_metric.log

make -f Makefile_main_qnn_metric -j4
echo "[INFO] Run main_qnn_metric"
sudo ./output/main_qnn_metric \
    ./models/mobileone_s0.tflite \
    ./images/dog.jpg \
    ./labels.json 2>&1 | tee output_main_qnn_metric.log

# make -f Makefile_main_cpu_keras -j4
# echo "[INFO] Run main_cpu_keras"
# ./output/main_cpu_keras \
#    ./models/resnet50-keras-application.tflite \
#    ./images/dog.jpg \
#    ./labels.json 2>&1 | tee output_main_cpu_keras.log
