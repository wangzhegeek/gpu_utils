#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

// 矩阵大小
#define MATRIX_SIZE 2048

// 额外内存块大小（MB）
#define EXTRA_MEMORY_MB 256

// CUDA核函数：执行矩阵乘法 C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

// 初始化矩阵
void initMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// 获取当前进程的可执行文件路径
void printProcessPath() {
    char path[1024];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        printf("当前进程路径: %s\n", path);
    } else {
        printf("无法获取进程路径\n");
    }
}

int main(int argc, char *argv[]) {
    // 默认运行时间（秒）
    int runTime = 60;
    
    // 解析命令行参数
    if (argc > 1) {
        runTime = atoi(argv[1]);
    }
    
    printf("CUDA矩阵乘法压力测试\n");
    printf("将运行 %d 秒\n", runTime);
    
    // 打印进程ID和路径
    printf("进程ID: %d\n", getpid());
    printProcessPath();
    
    // 矩阵大小
    int size = MATRIX_SIZE;
    size_t matrixBytes = size * size * sizeof(float);
    
    printf("矩阵大小: %d x %d (%zu MB)\n", size, size, matrixBytes / (1024 * 1024));
    
    // 分配主机内存
    float *h_A = (float*)malloc(matrixBytes);
    float *h_B = (float*)malloc(matrixBytes);
    float *h_C = (float*)malloc(matrixBytes);
    
    // 初始化输入矩阵
    initMatrix(h_A, size);
    initMatrix(h_B, size);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixBytes);
    cudaMalloc(&d_B, matrixBytes);
    cudaMalloc(&d_C, matrixBytes);
    
    // 分配额外的GPU内存，用于增加显存占用
    size_t extraBytes = (size_t)EXTRA_MEMORY_MB * 1024 * 1024;
    void *d_extra = nullptr;
    cudaMalloc(&d_extra, extraBytes);
    
    printf("额外分配GPU内存: %d MB\n", EXTRA_MEMORY_MB);
    printf("总GPU内存使用: %.2f GB\n", (matrixBytes * 3 + extraBytes) / (1024.0 * 1024 * 1024));
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice);
    
    // 定义线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("线程块大小: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("网格大小: %d x %d\n", blocksPerGrid.x, blocksPerGrid.y);
    
    // 记录开始时间
    time_t startTime = time(NULL);
    int iterations = 0;
    
    // 循环执行矩阵乘法，直到达到指定的运行时间
    while (difftime(time(NULL), startTime) < runTime) {
        // 执行矩阵乘法
        matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
        
        // 等待GPU完成
        cudaDeviceSynchronize();
        
        iterations++;
        
        // 每10次迭代打印一次状态
        if (iterations % 10 == 0) {
            printf("已完成 %d 次迭代，已运行 %.1f 秒\n", 
                   iterations, difftime(time(NULL), startTime));
        }
    }
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost);
    
    // 打印结果的一小部分，以验证计算
    printf("\n计算结果示例 (左上角 2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%f ", h_C[i * size + j]);
        }
        printf("\n");
    }
    
    printf("\n总共完成 %d 次矩阵乘法迭代\n", iterations);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_extra);
    
    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
} 