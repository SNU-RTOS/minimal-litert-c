#!/bin/bash

# ──────────────────────────────────────────────────────────────────────────────
source common.sh
cd ..
source .env

# ── Build Configuration ───────────────────────────────────────────────────────
BUILD_MODE=${1:-release}
setup_build_config "$BUILD_MODE"

# ── paths ─────────────────────────────────────────────────────────────────────
LITERT_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/libtensorflowlite.so

TENSORFLOW_INC_PATH=${LITERT_PATH}/bazel-litert/external/org_tensorflow/tensorflow
FLATBUFFER_INC_PATH=${LITERT_PATH}/bazel-litert/external/flatbuffers/include/flatbuffers
LITERT_INC_PATH=${LITERT_PATH}/tflite


echo "[INFO] Build LiteRT ($BUILD_MODE mode)…"
echo "[INFO] Core:      ${LITERT_LIB_PATH}"

cd "${LITERT_PATH}" || exit 1
pwd

# 1) Build
bazel build ${BAZEL_CONF} \
    //tflite:tensorflowlite \
    ${NO_GL_FLAG} \
    ${COPT_FLAGS} \
    ${LINKOPTS}

bazel shutdown

## ──────────── Libs ──────────────────────────────────────────────
create_symlink_or_fail "${LITERT_LIB_PATH}" \
                       "${ROOT_PATH}/lib/libtensorflowlite.so" \
                       "libtensorflowlite.so"

## ──────────── Headers ──────────────────────────────────────────────
create_symlink_or_fail "${LITERT_INC_PATH}" \
                       "${ROOT_PATH}/inc/" \
                       "LiteRT header files"

create_symlink_or_fail "${TENSORFLOW_INC_PATH}" \
                        "${ROOT_PATH}/inc/" \
                        "Tensorflow header files"

create_symlink_or_fail "${FLATBUFFER_INC_PATH}" \
                       "${ROOT_PATH}/inc/" \
                       "Flatbuffers header files"

# ── Generate Protobuf Headers ─────────────────────────────────────────────────

echo "[INFO] LiteRT build completed successfully!"

cd "${ROOT_PATH}/scripts" || exit 1
pwd