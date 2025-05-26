#!/bin/bash

# Pipeline command
MODEL1="./models/densenet.tflite"
CMD_PIPELINE="./output/run_model $MODEL1"

# Append all image paths from 0 to 1
for i in $(seq 1 50); do
  CMD_PIPELINE="$CMD_PIPELINE images/dog2.png"
done

# Add the options directly
CMD_PIPELINE="$CMD_PIPELINE"

# Execute the command
echo "Executing: $CMD_PIPELINE"
$CMD_PIPELINE