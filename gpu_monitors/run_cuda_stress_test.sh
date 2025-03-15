#!/bin/bash

# 默认运行时间（秒）
RUN_TIME=60

# 解析命令行参数
if [ $# -ge 1 ]; then
    RUN_TIME=$1
fi

echo "启动CUDA压力测试，将运行 $RUN_TIME 秒"
echo "可以在另一个终端中运行 ./gpu_monitor 来监控GPU使用情况"

# 运行CUDA压力测试程序
./cuda_stress_test $RUN_TIME

echo "CUDA压力测试完成" 