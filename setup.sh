#!/bin/bash

source .env

TENSORFLOW_VERSION=2.18.0
########## Setup env ##########
BINARY_NAME=minimal
BINARY_PATH=${ROOT_PATH}/bazel-bin/minimal-tflite/minimal/${BINARY_NAME}

echo "[INFO] ROOT_PATH: ${ROOT_PATH}"
echo "[INFO] EXTERNAL PATH: ${EXTERNAL_PATH}"
echo "[INFO] TENSORFLOW_PATH: ${TENSORFLOW_PATH}"

if [ ! -d ${EXTERNAL_PATH} ]; then
    mkdir -p ${EXTERNAL_PATH}
fi

########## Setup external sources ##########
cd ${EXTERNAL_PATH}
pwd
## Clone tensorflow
echo "[INFO] Installing tensorflow"
if [ ! -d "./tensorflow" ]; then
    git clone --branch v${TENSORFLOW_VERSION} --depth 1 https://github.com/tensorflow/tensorflow.git
    cd ${TENSORFLOW_PATH}
    ./configure    
else
    echo "[INFO] tensorflow is already installed, skipping ..."
fi


########## Update Path of tensorflow ##########
cd ${ROOT_PATH}/scripts
./build-litert.sh
./build-litert_gpu_delegate.sh
