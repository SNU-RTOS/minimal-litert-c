#!/bin/bash
# This script is used to run the minimal litert-c example
# It builds the verify_cpu and verify_gpu programs and runs them
# It also builds the main_cpu program and runs it
# It is assumed that the litert-c library is already built and available in the lib directory
# The script will log the output of each program to a separate log file

run_verify() {
  local mode="$1"         # Example: cpu
  local model="$2"        # Path of Model: ./models/mobileone_s0.tflite
  local logfile="verify_${mode}_$(basename "${model%.*}").log"

  (
    exec > >(tee "$logfile") 2>&1
    echo "================================"
    echo "[INFO] Build verify_${mode}"
    make -f Makefile_verify_${mode} -j4

    echo "[INFO] Run verify_${mode}"
    ./output/verify_${mode} "$model"
    echo "[INFO] Run verify_${mode} finished"
  )
}

run_main() {
  local mode="$1"          # Example: cpu
  local model="$2"         # Example: ./models/resnet34.tflite
  local image="$3"         # Example: ./images/dog.jpg
  local label="$4"         # Example: ./labels.json

  local model_base
  model_base=$(basename "${model%.*}")
  local logfile="output_main_${mode}_${model_base}.log"

  (
    exec > >(tee "$logfile") 2>&1

    echo "================================"
    echo "[INFO] Build main_${mode}"
    make -f Makefile_main_${mode} -j4

    echo "[INFO] Run main_${mode}"
    ./output/main_${mode} "$model" "$image" "$label"
  )
}


run_main_metric() {
  local mode="$1"          # Example: cpu
  local model="$2"         # Example: ./models/resnet34.tflite
  local image="$3"         # Example: ./images/dog.jpg
  local label="$4"         # Example: ./labels.json

  local model_base
  model_base=$(basename "${model%.*}")
  local logfile="output_main_metric_${mode}_${model_base}.log"

  (
    exec > >(tee "$logfile") 2>&1

    echo "================================"
    echo "[INFO] Build main_${mode}_metric"
    make -f Makefile_main_${mode}_metric -j4

    echo "[INFO] Run main_${mode}_metric"
    ./output/main_${mode}_metric "$model" "$image" "$label"
  )
}

##################### main #####################
# run_verify cpu ./models/mobileone_s0.tflite
# run_verify gpu ./models/resnet.tflite
run_verify qnn ./models/resnet_quantize.tflite
# run_main_metric cpu ./models/resnet_quantize.tflite ./images/dog.jpg ./labels.json
# run_main_metric qnn ./models/resnet_quantize.tflite ./images/dog.jpg ./labels.json
# run_main_metric gpu ./models/resnet_quantize.tflite ./images/dog.jpg ./labels.json


# run_main cpu ./models/mobileone_s0.tflite ./images/dog.jpg ./labels.json

# LOGFILE=verify_cpu_mobileone_s0.log
# exec > >(tee "$LOGFILE") 2>&1
# echo "[INFO] Build verify_cpu"
# make -f Makefile_verify_cpu -j4
# echo "[INFO] Run verify_cpu"
# ./output/verify_cpu ./models/mobileone_s0.tflite

# LOGFILE=verify_gpu_mobileone_s0.log
# exec > >(tee "$LOGFILE") 2>&1
# echo "[INFO] Build verify_gpu"
# make -f Makefile_verify_gpu -j4
# echo "[INFO] Run verify_gpu"
# ./output/verify_gpu ./models/mobileone_s0.tflite 

# make -f Makefile_main_cpu -j4
# echo "[INFO] Run main_cpu"
# ./output/main_cpu \
#     ./models/resnet34.tflite \
#     ./images/dog.jpg \
#     ./labels.json | tee output_main_cpu.log

