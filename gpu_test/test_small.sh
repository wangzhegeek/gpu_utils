#!/bin/bash

# 测试小矩阵大小的性能

# 编译函数，接受矩阵大小作为参数
compile() {
    local size=$1
    echo "编译矩阵大小为 ${size}x${size} 的版本..."
    sed -i "s/#define MATRIX_SIZE [0-9]*/#define MATRIX_SIZE $size/" matrix_multiply.cu
    make clean && make
}

# 运行测试
run_test() {
    local size=$1
    echo "===== 测试矩阵大小: ${size}x${size} ====="
    ./matrix_multiply
    echo ""
}

# 测试小矩阵大小
compile 256
run_test 256

# 恢复默认矩阵大小
compile 1024 