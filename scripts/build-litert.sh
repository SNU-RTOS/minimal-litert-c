#!/bin/bash
cd ..
source .env
LITERT_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/libtensorflowlite.so
FLATBUFFER_PATH=${TENSORFLOW_PATH}/bazel-tensorflow/external/flatbuffers/include/flatbuffers

########## Build ##########
cd ${TENSORFLOW_PATH}

echo "[INFO] Build LiteRT .so .."
echo "[INFO] Path: ${LITERT_PATH}"

cd ${TENSORFLOW_PATH}
pwd

# Release mode
bazel build -c opt //tensorflow/lite:tensorflowlite \
    --copt=-Os \
    --copt=-fPIC 

# Debug mode
# bazel build -c dbg //tensorflow/lite:tensorflowlite \
#     --copt=-Os \
#     --copt=-fPIC 

########## Make symlink ##########
ln -sf ${LITERT_PATH} ${ROOT_PATH}/lib/libtensorflowlite.so
ln -sf ${FLATBUFFER_PATH} ${ROOT_PATH}/inc/
ln -sf ${TENSORFLOW_PATH}/tensorflow ${ROOT_PATH}/inc/ 

cd ${ROOT_PATH}/scripts
pwd
