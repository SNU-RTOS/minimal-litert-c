#!/bin/bash

cd ..
source .env

cd external

sudo apt install libopencv-dev -y
# wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip
# wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.11.0.zip
# unzip opencv.zip
# unzip opencv_contrib.zip
# mkdir opencv_build
# cd opencv_build
# cmake -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/usr/src \
#     -DWITH_OPENCL=OFF \
#     -DWITH_OPENCL_SVM=OFF \
#     -DWITH_OPENCL_D3D11_NV=OFF \
#     -DWITH_OPENGL=OFF \
#     -DBUILD_opencv_world=ON \
#     -DBUILD_TESTS=OFF \
#     -DBUILD_PERF_TESTS=OFF \
#     -DWITH_TBB=OFF \
#     -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.11.0/modules/ \
#     ../opencv-4.11.0/

# make -j4
