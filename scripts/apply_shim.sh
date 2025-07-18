#!/bin/bash

# Changing symbolic links
sudo ln -sf "$(pwd)/output/libOpenCL_profiling_shim.so" /usr/lib/libOpenCL_profiling_shim.so
sudo ln -sf /usr/lib/libOpenCL_profiling_shim.so /usr/lib/libOpenCL.so