#!/bin/bash

# 设置CUDA环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

# 运行GPU监控程序
# 参数说明:
#   -i <seconds>: 设置刷新间隔（秒）
#   --once: 只运行一次，不持续监控

cd "$(dirname "$0")"
./gpu_monitor "$@" 