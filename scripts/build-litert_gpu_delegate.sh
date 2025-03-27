#!/bin/bash
cd ..
source .env
GPU_DELEGATE_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so

########## Build ##########
cd ${TENSORFLOW_PATH}

echo "[INFO] Build gpu delegate .so .."
echo "[INFO] Path: ${GPU_DELEGATE_PATH}"

cd ${TENSORFLOW_PATH}
pwd
bazel build -c dbg //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
    --copt=-Os \
    --copt=-fPIC \
    --copt=-DTFLITE_GPU_DEBUG_LOGGING=1

########## Make symlink ##########
ln -sf ${GPU_DELEGATE_PATH} ${ROOT_PATH}/lib/libtensorflowlite_gpu_delegate.so

cd ${ROOT_PATH}/scripts
pwd
