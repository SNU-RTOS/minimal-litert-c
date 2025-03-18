#!/bin/bash

cmake -S . -B build 
cd build
make -j6

#   -DBUILD_SHARED_LIBS=ON \
#   -DTFLITE_HOST_TOOLS_DIR=/home/rtos-ghpark/workspace/source-packages/flatbuffers \