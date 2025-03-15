#!/bin/bash

# 运行所有矩阵乘法测试

echo "===== 运行默认大小（1024x1024）的矩阵乘法 ====="
./matrix_multiply
echo ""

echo "===== 运行小矩阵（256x256）测试 ====="
bash ./test_small.sh
echo ""

echo "===== 是否要运行所有大小的测试？这可能需要较长时间（y/n）====="
read -p "请输入选择: " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo "===== 运行不同大小矩阵的测试 ====="
    bash ./test_different_sizes.sh
    echo ""
fi

echo "所有测试完成！" 