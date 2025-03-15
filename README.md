# GPU-Utils

GPU-Utils 是一个用于 NVIDIA GPU 监控和 CUDA 矩阵计算的工具集合。该项目包含两个主要模块：GPU 监控工具和 CUDA 矩阵乘法性能测试工具。

## 项目结构

- `gpu_monitors/`: GPU 监控工具，用于实时监控 NVIDIA GPU 的状态
- `gpu_test/`: CUDA 矩阵乘法性能测试工具，用于测试不同实现方式的性能

## 系统要求

- NVIDIA GPU
- CUDA Toolkit 12.4 或更高版本
- C++17 兼容的编译器
- Linux 操作系统（已在 Ubuntu 20.04 和 22.04 上测试）

## 安装方法

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/gpu-utils.git
cd gpu-utils
```

2. 编译 GPU 监控工具：

```bash
cd gpu_monitors
make
```

3. 编译 CUDA 矩阵乘法工具：

```bash
cd ../gpu_test
make
```

## 功能特点

### GPU 监控工具

- 支持监控多台 NVIDIA GPU
- 显示 GPU 温度、利用率、内存使用情况、功率等信息
- 显示 GPU 上运行的进程信息（包括进程 ID、显存使用量和进程路径）
- 支持自定义刷新间隔
- 支持单次运行或持续监控模式
- 包含 CUDA 压力测试程序，用于测试 GPU 负载

### CUDA 矩阵乘法工具

- 支持大规模矩阵乘法计算
- 提供多种实现方式：
  - CPU 实现
  - GPU 朴素实现
  - GPU 共享内存优化实现
  - cuBLAS 库实现
- 自动验证计算结果的正确性
- 性能对比和加速比分析

## 使用方法

### GPU 监控工具

```bash
cd gpu_monitors

# 使用脚本运行（推荐）
./run_gpu_monitor.sh        # 默认每秒刷新一次
./run_gpu_monitor.sh -i 5   # 设置刷新间隔为 5 秒
./run_gpu_monitor.sh --once # 只运行一次

# 运行 CUDA 压力测试
./run_cuda_stress_test.sh      # 默认运行 60 秒
./run_cuda_stress_test.sh 120  # 指定运行时间（秒）
```

### CUDA 矩阵乘法工具

```bash
cd gpu_test

# 运行默认大小（1024x1024）的矩阵乘法
./matrix_multiply

# 测试小矩阵（256x256）的性能
./test_small.sh

# 测试不同大小矩阵的性能
./test_different_sizes.sh

# 运行所有测试
./run_all_tests.sh
```

## 详细文档

- [GPU 监控工具文档](gpu_monitors/README.md)
- [CUDA 矩阵乘法工具文档](gpu_test/README.md)

## 贡献指南

欢迎提交 Issues 和 Pull Requests 来改进这个项目。在提交 PR 之前，请确保您的代码符合项目的编码规范。

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。 