#!/bin/bash

# 모델 및 이미지 경로 설정
MODEL0="./models/sub_model_1_resnet.tflite"
MODEL1="./models/sub_model_2_resnet.tflite"
IMAGE="./images/_images_161.png"
INPUT_RATE=0

# 이미지 경로를 100번 반복하여 인자 리스트 생성
IMAGE_ARGS=()
for i in {1..100}; do
    IMAGE_ARGS+=("$IMAGE")
done

# 실행
./output/pipeline "$MODEL0" "$MODEL1" "${IMAGE_ARGS[@]}" --input-rate=$INPUT_RATE
