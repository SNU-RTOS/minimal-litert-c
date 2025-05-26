#!/bin/bash

# === Configuration ===
CGROUP_NAME="my_limit_group"
CGROUP_PATH="/sys/fs/cgroup/$CGROUP_NAME"
MEM_LIMIT=$((10 * 1024 * 1024))  # 10 MB
MODEL="./models/resnet.tflite"
IMAGE="images/dog2.png"

# === 1. Create and configure cgroup ===
sudo mkdir -p "$CGROUP_PATH"

# Set memory and swap limits
echo "$MEM_LIMIT" | sudo tee "$CGROUP_PATH/memory.max" > /dev/null
echo 0 | sudo tee "$CGROUP_PATH/memory.swap.max" > /dev/null

# Enable unified OOM kill
echo 1 | sudo tee "$CGROUP_PATH/memory.oom.group" > /dev/null

# === 2. Build model command ===
CMD="./output/run_model $MODEL"
for i in $(seq 1 50); do
  CMD="$CMD $IMAGE"
done

# === 3. Run the command in background ===
echo "Launching command with 10MB memory limit:"
echo "$CMD"
bash -c "$CMD" &
CMD_PID=$!

# === 4. Assign process to cgroup ===
echo "$CMD_PID" | sudo tee "$CGROUP_PATH/cgroup.procs" > /dev/null

# === 5. Monitor memory usage and wait ===
# echo "Monitoring memory usage..."
# while kill -0 "$CMD_PID" 2>/dev/null; do
#   MEM_CURRENT=$(cat "$CGROUP_PATH/memory.current")
#   echo "Memory used: $MEM_CURRENT bytes"
#   sleep 1
# done

echo "Process finished. Check dmesg for OOM kill logs if it was terminated."
