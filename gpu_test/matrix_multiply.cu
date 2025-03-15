#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <sys/time.h>

// 定义矩阵大小
#define MATRIX_SIZE 1024  // 可以根据需要调整矩阵大小

// 定义CUDA核函数的线程块大小
#define BLOCK_SIZE 32

// 用于计时的函数
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

// 初始化矩阵
void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)(rand() % 100) / 100.0f;
    }
}

// 打印矩阵（用于调试小矩阵）
void print_matrix(float *matrix, int size) {
    if (size > 16) {
        printf("矩阵太大，只打印左上角 4x4 部分：\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%.2f\t", matrix[i * size + j]);
            }
            printf("\n");
        }
    } else {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                printf("%.2f\t", matrix[i * size + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

// CPU实现的矩阵乘法
void matrix_multiply_cpu(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// CUDA核函数：朴素的矩阵乘法实现
__global__ void matrix_multiply_kernel_naive(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// CUDA核函数：使用共享内存优化的矩阵乘法实现
__global__ void matrix_multiply_kernel_shared(float *A, float *B, float *C, int size) {
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // 循环计算一个线程块内的结果
    for (int i = 0; i < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        // 加载数据到共享内存
        if (row < size && i * BLOCK_SIZE + tx < size) {
            sharedA[ty][tx] = A[row * size + i * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (i * BLOCK_SIZE + ty < size && col < size) {
            sharedB[ty][tx] = B[(i * BLOCK_SIZE + ty) * size + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算部分结果
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写回结果
    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}

// GPU实现的矩阵乘法（使用朴素核函数）
void matrix_multiply_gpu_naive(float *A, float *B, float *C, int size) {
    float *d_A, *d_B, *d_C;
    int matrix_bytes = size * size * sizeof(float);
    
    // 分配设备内存
    cudaMalloc((void**)&d_A, matrix_bytes);
    cudaMalloc((void**)&d_B, matrix_bytes);
    cudaMalloc((void**)&d_C, matrix_bytes);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matrix_bytes, cudaMemcpyHostToDevice);
    
    // 设置核函数的执行配置
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动核函数
    matrix_multiply_kernel_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    
    // 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, matrix_bytes, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// GPU实现的矩阵乘法（使用共享内存优化的核函数）
void matrix_multiply_gpu_shared(float *A, float *B, float *C, int size) {
    float *d_A, *d_B, *d_C;
    int matrix_bytes = size * size * sizeof(float);
    
    // 分配设备内存
    cudaMalloc((void**)&d_A, matrix_bytes);
    cudaMalloc((void**)&d_B, matrix_bytes);
    cudaMalloc((void**)&d_C, matrix_bytes);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matrix_bytes, cudaMemcpyHostToDevice);
    
    // 设置核函数的执行配置
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动核函数
    matrix_multiply_kernel_shared<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    
    // 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, matrix_bytes, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// GPU实现的矩阵乘法（使用cuBLAS库）
void matrix_multiply_cublas(float *A, float *B, float *C, int size) {
    float *d_A, *d_B, *d_C;
    int matrix_bytes = size * size * sizeof(float);
    
    // 分配设备内存
    cudaMalloc((void**)&d_A, matrix_bytes);
    cudaMalloc((void**)&d_B, matrix_bytes);
    cudaMalloc((void**)&d_C, matrix_bytes);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matrix_bytes, cudaMemcpyHostToDevice);
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // 执行矩阵乘法：C = A * B
    // 注意：cuBLAS使用列主序，而我们使用行主序，所以这里计算的是 B^T * A^T = (A * B)^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, d_B, size, d_A, size, &beta, d_C, size);
    
    // 等待计算完成
    cudaDeviceSynchronize();
    
    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, matrix_bytes, cudaMemcpyDeviceToHost);
    
    // 销毁cuBLAS句柄
    cublasDestroy(handle);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 验证结果是否正确
bool verify_result(float *A, float *B, int size) {
    // 允许的误差范围
    const float epsilon = 1e-2;
    
    for (int i = 0; i < size * size; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("结果验证失败：A[%d] = %.6f, B[%d] = %.6f, 差值 = %.6f\n", 
                   i, A[i], i, B[i], fabs(A[i] - B[i]));
            return false;
        }
    }
    
    return true;
}

int main() {
    // 设置随机数种子
    srand(time(NULL));
    
    // 矩阵大小
    int size = MATRIX_SIZE;
    int matrix_bytes = size * size * sizeof(float);
    
    // 分配主机内存
    float *A = (float*)malloc(matrix_bytes);
    float *B = (float*)malloc(matrix_bytes);
    float *C_cpu = (float*)malloc(matrix_bytes);
    float *C_gpu_naive = (float*)malloc(matrix_bytes);
    float *C_gpu_shared = (float*)malloc(matrix_bytes);
    float *C_cublas = (float*)malloc(matrix_bytes);
    
    if (!A || !B || !C_cpu || !C_gpu_naive || !C_gpu_shared || !C_cublas) {
        printf("内存分配失败\n");
        return -1;
    }
    
    // 初始化矩阵
    printf("初始化 %d x %d 的矩阵...\n", size, size);
    init_matrix(A, size);
    init_matrix(B, size);
    
    // 打印矩阵（仅用于调试小矩阵）
    if (size <= 16) {
        printf("矩阵 A:\n");
        print_matrix(A, size);
        printf("矩阵 B:\n");
        print_matrix(B, size);
    }
    
    // 使用CPU计算矩阵乘法
    printf("使用CPU计算矩阵乘法...\n");
    double cpu_start = get_time();
    matrix_multiply_cpu(A, B, C_cpu, size);
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;
    printf("CPU计算时间: %.6f 秒\n", cpu_time);
    
    // 使用GPU计算矩阵乘法（朴素实现）
    printf("使用GPU计算矩阵乘法（朴素实现）...\n");
    double gpu_naive_start = get_time();
    matrix_multiply_gpu_naive(A, B, C_gpu_naive, size);
    double gpu_naive_end = get_time();
    double gpu_naive_time = gpu_naive_end - gpu_naive_start;
    printf("GPU朴素实现计算时间: %.6f 秒\n", gpu_naive_time);
    printf("加速比: %.2f 倍\n", cpu_time / gpu_naive_time);
    
    // 验证结果
    printf("验证GPU朴素实现结果...\n");
    if (verify_result(C_cpu, C_gpu_naive, size)) {
        printf("GPU朴素实现结果正确\n");
    } else {
        printf("GPU朴素实现结果错误\n");
    }
    
    // 使用GPU计算矩阵乘法（共享内存优化）
    printf("使用GPU计算矩阵乘法（共享内存优化）...\n");
    double gpu_shared_start = get_time();
    matrix_multiply_gpu_shared(A, B, C_gpu_shared, size);
    double gpu_shared_end = get_time();
    double gpu_shared_time = gpu_shared_end - gpu_shared_start;
    printf("GPU共享内存优化计算时间: %.6f 秒\n", gpu_shared_time);
    printf("加速比（相对于CPU）: %.2f 倍\n", cpu_time / gpu_shared_time);
    printf("加速比（相对于GPU朴素实现）: %.2f 倍\n", gpu_naive_time / gpu_shared_time);
    
    // 验证结果
    printf("验证GPU共享内存优化结果...\n");
    if (verify_result(C_cpu, C_gpu_shared, size)) {
        printf("GPU共享内存优化结果正确\n");
    } else {
        printf("GPU共享内存优化结果错误\n");
    }
    
    // 使用cuBLAS计算矩阵乘法
    printf("使用cuBLAS计算矩阵乘法...\n");
    double cublas_start = get_time();
    matrix_multiply_cublas(A, B, C_cublas, size);
    double cublas_end = get_time();
    double cublas_time = cublas_end - cublas_start;
    printf("cuBLAS计算时间: %.6f 秒\n", cublas_time);
    printf("加速比（相对于CPU）: %.2f 倍\n", cpu_time / cublas_time);
    printf("加速比（相对于GPU朴素实现）: %.2f 倍\n", gpu_naive_time / cublas_time);
    printf("加速比（相对于GPU共享内存优化）: %.2f 倍\n", gpu_shared_time / cublas_time);
    
    // 验证结果
    printf("验证cuBLAS结果...\n");
    if (verify_result(C_cpu, C_cublas, size)) {
        printf("cuBLAS结果正确\n");
    } else {
        printf("cuBLAS结果错误\n");
    }
    
    // 打印结果矩阵（仅用于调试小矩阵）
    if (size <= 16) {
        printf("结果矩阵 C (CPU):\n");
        print_matrix(C_cpu, size);
        printf("结果矩阵 C (GPU朴素实现):\n");
        print_matrix(C_gpu_naive, size);
        printf("结果矩阵 C (GPU共享内存优化):\n");
        print_matrix(C_gpu_shared, size);
        printf("结果矩阵 C (cuBLAS):\n");
        print_matrix(C_cublas, size);
    }
    
    // 释放内存
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu_naive);
    free(C_gpu_shared);
    free(C_cublas);
    
    return 0;
} 