# GPU监控程序

这是一个用于监控NVIDIA GPU的C++程序，可以实时显示多台GPU的状态信息，包括温度、利用率、内存使用情况、功率等。

## 功能特点

- 支持监控多台NVIDIA GPU
- 显示GPU温度、利用率、内存使用情况、功率等信息
- 显示GPU上运行的进程信息（包括进程ID、显存使用量和进程路径）
- 支持自定义刷新间隔
- 支持单次运行或持续监控模式

## 编译方法

```bash
make
```

## 运行方法

### 使用脚本运行

```bash
# 默认每秒刷新一次
./run_gpu_monitor.sh

# 设置刷新间隔为5秒
./run_gpu_monitor.sh -i 5

# 只运行一次
./run_gpu_monitor.sh --once
```

### 直接运行可执行文件

```bash
# 需要设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

# 默认每秒刷新一次
./gpu_monitor

# 设置刷新间隔为5秒
./gpu_monitor -i 5

# 只运行一次
./gpu_monitor --once
```

## CUDA压力测试程序

为了测试GPU监控程序的功能，我们提供了一个CUDA压力测试程序，它会执行重复的矩阵乘法计算，以便产生GPU负载。

### 功能特点

- 执行大规模矩阵乘法计算（默认2048x2048）
- 分配额外的GPU内存（默认256MB）以增加显存占用
- 显示进程ID和可执行文件路径，便于在监控程序中识别
- 可自定义运行时间

### 编译方法

```bash
make cuda_stress_test
```

### 运行方法

#### 使用脚本运行

```bash
# 默认运行60秒
./run_cuda_stress_test.sh

# 指定运行时间（秒）
./run_cuda_stress_test.sh 120
```

#### 直接运行可执行文件

```bash
# 默认运行60秒
./cuda_stress_test

# 指定运行时间（秒）
./cuda_stress_test 120
```

### 测试方法

1. 在一个终端中启动GPU监控程序：
   ```bash
   ./gpu_monitor
   ```

2. 在另一个终端中启动CUDA压力测试程序：
   ```bash
   ./cuda_stress_test
   ```

3. 观察GPU监控程序是否能够检测到CUDA压力测试程序的进程ID、显存使用情况和进程路径。

## 注意事项

由于权限限制，GPU监控程序可能无法获取某些进程的确切路径。在这种情况下，它会显示"CUDA进程 (可能是cuda_stress_test)"作为进程路径。

## 系统要求

- NVIDIA GPU
- CUDA Toolkit 12.4或更高版本
- C++17兼容的编译器 