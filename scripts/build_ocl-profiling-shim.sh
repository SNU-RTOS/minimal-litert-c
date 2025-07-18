#!/bin/bash

# Build libOpenCL_shim_profiling.so
g++ -shared -w -fPIC -O2 -I./common src/libOpenCL_profiling_shim.cpp -ldl -pthread -o output/libOpenCL_profiling_shim.so