#!/bin/bash
source .env

GPU_DELEGATE_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so
echo "[INFO] GPU_DELEGATE_PATH: ${GPU_DELEGATE_PATH}"
########## Build ##########
if [ !"${GPU_DELEGATE_PATH}" ]; then
    cd ${TENSORFLOW_PATH}
    pwd
    bazel build -c opt //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
        --copt=-Os \
        --copt=-fPIC \
        --linkopt=-s
else
    echo "[INFO] libtensorflowlite_gpu_delegate.so is already built, skipping ..."
fi

cd ${ROOT_PATH}
pwd