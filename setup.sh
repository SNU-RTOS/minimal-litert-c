#!/bin/bash

source .env

TENSORFLOW_VERSION=2.18.0
########## Setup env ##########

echo "[INFO] ROOT_PATH: ${ROOT_PATH}"
echo "[INFO] EXTERNAL PATH: ${EXTERNAL_PATH}"
echo "[INFO] TENSORFLOW_PATH: ${TENSORFLOW_PATH}"

if [ ! -d ${EXTERNAL_PATH} ]; then
    mkdir -p ${EXTERNAL_PATH}
fi

########## Setup external sources #########S#
cd ${EXTERNAL_PATH}
pwd
## Clone tensorflow
echo "[INFO] Installing tensorflow"
if [ ! -d "./tensorflow" ]; then
    git clone --branch v${TENSORFLOW_VERSION} \
        --depth 1 https://github.com/tensorflow/tensorflow.git
    cd ${TENSORFLOW_PATH}
    ./configure    
else
    echo "[INFO] tensorflow is already installed, skipping ..."
fi


########## Build LiteRT ##########
cd ${ROOT_PATH}/scripts
./build-litert.sh
./build-litert_gpu_delegate.sh


########## Make folders ##########
cd ${ROOT_PATH}
if [ ! -d "inc" ]; then
    mkdir inc    
fi

if [ ! -d "lib" ]; then
    mkdir lib   
fi

if [ ! -d "obj" ]; then
    mkdir obj  
fi

if [ ! -d "output" ]; then
    mkdir output  
fi


echo "[INFO] Setup Finished"