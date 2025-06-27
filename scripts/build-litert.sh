#!/bin/bash
cd ..
source .env
# LITERT_LIB_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/libtensorflowlite.so
LITERT_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/libtensorflowlite.so
FLATBUFFER_PATH=${LITERT_PATH}/bazel-LiteRT/external/flatbuffers/include/flatbuffers
TENSORFLOW_PATH=${LITERT_PATH}/bazel-LiteRT/external/org_tensorflow/tensorflow

########## Build ##########
cd ${LITERT_PATH}

echo "[INFO] Build LiteRT .so .."
echo "[INFO] Path: ${LITERT_LIB_PATH}"

cd ${LITERT_PATH}
pwd

# Release mode (Note: -Wno-incompatible-pointer-types could cause undefined behavior)
# bazel build -c opt //tflite:tensorflowlite \
#     --copt=-Os \
#     --copt=-fPIC \
#     --copt=-Wno-incompatible-pointer-types \
#     --linkopt=-s

########## Make symlink ##########
ln -sf ${LITERT_LIB_PATH} ${ROOT_PATH}/lib/libtensorflowlite.so
ln -sf ${FLATBUFFER_PATH} ${ROOT_PATH}/inc/
ln -sf ${LITERT_PATH}/tflite ${ROOT_PATH}/inc 
ln -sf ${TENSORFLOW_PATH} ${ROOT_PATH}/inc/

cd ${ROOT_PATH}/scripts
pwd




