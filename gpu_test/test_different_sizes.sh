#!/bin/bash

# 测试不同矩阵大小的性能

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

# 测试不同的矩阵大小
# 使用空格分隔的字符串代替数组，以提高兼容性
SIZES="512 1024 2048"

for size in $SIZES; do
    compile $size
    run_test $size
done

# 恢复默认矩阵大小
compile 1024 