# Makefile

# 디렉토리 설정
ROOT_DIR := /root
SRC_DIR := src
OBJ_DIR := obj
TARGET := output/test

# 컴파일러 및 플래그 설정
CXX := g++
CXXFLAGS := -std=c++17  

# OpenGL 및 EGL, GLES 추가
LDFLAGS := -Wl,--rpath=/usr/lib \
-ltensorflowlite -ltensorflowlite_gpu_delegate -lEGL -lGLESv2 \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lpthread

# 파일 설정
INCS := -I/root/ghpark/LiteRT_LLM_inference_app/external/tensorflow/ \
		-I/root/ghpark/LiteRT_LLM_inference_app/external/tensorflow/bazel-tensorflow/external/flatbuffers/include \
		-I/usr/include \
        -I/usr/include/opencv4 \
        -I/usr/include/opencv2
# INCS := -I$(ROOT_DIR)/inc -I/root/tensorflow/tensorflow/core -I/root/tensorflow/tensorflow/lite
LIBS := -L/usr/lib -L /root/ghpark/LiteRT_LLM_inference_app/external/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu 

SRCS := test.cpp
HDRS := $(shell find $(SRC_DIR) -name '*.hpp')
OBJS := $(SRCS:%.cpp=%.o)
OBJECTS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
DEPS = $(OBJECTS:.o=.d)


.PHONY: all clean

# 기본 타겟
all: $(TARGET) 
	@echo The build completed successfully

# 오브젝트 파일 생성 규칙
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(HDRS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -c $< -o $@ -MD $(LDFLAGS)

# 실행 파일 생성 규칙
$(TARGET): $(OBJECTS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) $(OBJECTS) -o $(TARGET) $(LDFLAGS) 

# 정리 규칙 
clean:
	rm -f $(TARGET)
	rm -f $(OBJECTS)
	rm -f $(DEPS)

-include $(DEPS)
