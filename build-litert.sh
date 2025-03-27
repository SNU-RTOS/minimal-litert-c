#!/bin/bash
source .env

########## Build ##########
cd ${TENSORFLOW_PATH}

LITERT_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/libtensorflowlite.so
echo "[INFO] LITERT_PATH: ${LITERT_PATH}"
########## Build ##########
if [ !"${LITERT_PATH}" ]; then
    cd ${TENSORFLOW_PATH}
    pwd
    bazel build -c opt //tensorflow/lite:tensorflowlite  \
        --copt=-Os \
        --copt=-fPIC \
        --linkopt=-s
    
else
    echo "[INFO] libtensorflowlite.so is already built, skipping ..."
fi
cd ${ROOT_PATH}
pwd

# ########## Make soft symlink ##########
# echo "[INFO] Succefully built ${BINARY_NAME}"
# echo "[INFO] Making soft symbolic link ${BINARY_NAME} from ${BINARY_PATH} to ${ROOT_PATH}"
# if [ "${BINARY_NAME}" ]; then
#     rm ${BINARY_NAME}
# fi
# ln -s ${BINARY_PATH} ${BINARY_NAME}

# echo "[INFO] Setup finished."