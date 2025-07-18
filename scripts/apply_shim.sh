#!/bin/bash

# Changing symbolic links
sudo ln -sf "$(pwd)/output/libOpenCL_shim_profiling.so" /usr/lib/libOpenCL_shim_profiling.so
sudo ln -sf /usr/lib/libOpenCL_shim_profiling.so /usr/lib/libOpenCL.so