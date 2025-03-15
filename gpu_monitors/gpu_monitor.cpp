#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <map>
#include <string.h>
#include </usr/local/cuda-12.4/include/nvml.h>
#include <filesystem>

// 格式化显示内存大小
std::string formatMemory(unsigned long long bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = bytes;
    
    while (size >= 1024 && unitIndex < 4) {
        size /= 1024;
        unitIndex++;
    }
    
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unitIndex]);
    return std::string(buffer);
}

// 格式化显示温度
std::string formatTemperature(unsigned int temp) {
    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%u°C", temp);
    return std::string(buffer);
}

// 格式化显示利用率百分比
std::string formatUtilization(unsigned int util) {
    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%u%%", util);
    return std::string(buffer);
}

// 格式化显示功率
std::string formatPower(unsigned int power) {
    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%.2f W", power / 1000.0);
    return std::string(buffer);
}

// 获取进程的可执行文件路径
std::string getProcessPath(unsigned int pid) {
    std::string path = "未知";
    
    // 尝试通过/proc/PID/cmdline获取命令行
    std::stringstream cmdline_path;
    cmdline_path << "/proc/" << pid << "/cmdline";
    std::ifstream cmdline_file(cmdline_path.str());
    if (cmdline_file.is_open()) {
        std::string cmdline;
        std::getline(cmdline_file, cmdline, '\0');
        if (!cmdline.empty()) {
            path = cmdline;
        }
        cmdline_file.close();
    }
    
    // 如果cmdline为空，尝试通过readlink获取进程路径
    if (path == "未知") {
        std::stringstream exe_path;
        exe_path << "/proc/" << pid << "/exe";
        char buffer[1024];
        ssize_t len = readlink(exe_path.str().c_str(), buffer, sizeof(buffer) - 1);
        if (len != -1) {
            buffer[len] = '\0';
            path = buffer;
        }
    }
    
    // 如果是CUDA进程，尝试猜测路径
    if (path == "未知") {
        path = "CUDA进程 (可能是cuda_stress_test)";
    }
    
    return path;
}

// 获取GPU信息的结构体
struct GPUInfo {
    std::string name;
    std::string pcieBusId;
    unsigned int temperature;
    unsigned int fanSpeed;
    unsigned int utilizationGPU;
    unsigned int utilizationMemory;
    unsigned long long memoryTotal;
    unsigned long long memoryUsed;
    unsigned long long memoryFree;
    unsigned int powerUsage;
    unsigned int powerCapacity;
    std::vector<nvmlProcessInfo_t> processes;
};

// 获取单个GPU的信息
GPUInfo getGPUInfo(nvmlDevice_t device) {
    GPUInfo info;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlPciInfo_t pciInfo;
    nvmlMemory_t memory;
    nvmlUtilization_t utilization;
    unsigned int temperature;
    unsigned int fanSpeed;
    unsigned int powerUsage;
    unsigned int powerCapacity;
    unsigned int processCount = 0;
    
    // 获取GPU名称
    nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    info.name = name;
    
    // 获取PCI总线ID
    nvmlDeviceGetPciInfo(device, &pciInfo);
    info.pcieBusId = pciInfo.busId;
    
    // 获取温度
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
    info.temperature = temperature;
    
    // 尝试获取风扇转速（某些GPU可能不支持）
    nvmlReturn_t result = nvmlDeviceGetFanSpeed(device, &fanSpeed);
    if (result == NVML_SUCCESS) {
        info.fanSpeed = fanSpeed;
    } else {
        info.fanSpeed = 0; // 不支持风扇转速读取
    }
    
    // 获取利用率
    nvmlDeviceGetUtilizationRates(device, &utilization);
    info.utilizationGPU = utilization.gpu;
    info.utilizationMemory = utilization.memory;
    
    // 获取内存信息
    nvmlDeviceGetMemoryInfo(device, &memory);
    info.memoryTotal = memory.total;
    info.memoryUsed = memory.used;
    info.memoryFree = memory.free;
    
    // 获取功率信息
    nvmlDeviceGetPowerUsage(device, &powerUsage);
    info.powerUsage = powerUsage;
    nvmlDeviceGetPowerManagementLimit(device, &powerCapacity);
    info.powerCapacity = powerCapacity;
    
    // 获取进程信息
    nvmlDeviceGetComputeRunningProcesses(device, &processCount, nullptr);
    if (processCount > 0) {
        std::vector<nvmlProcessInfo_t> processes(processCount);
        nvmlDeviceGetComputeRunningProcesses(device, &processCount, processes.data());
        info.processes = processes;
    }
    
    return info;
}

// 显示GPU信息
void displayGPUInfo(const GPUInfo& info, int gpuIndex) {
    std::cout << "===== GPU " << gpuIndex << " (" << info.name << ") =====" << std::endl;
    std::cout << "温度: " << formatTemperature(info.temperature) << std::endl;
    if (info.fanSpeed > 0) {
        std::cout << "风扇转速: " << formatUtilization(info.fanSpeed) << std::endl;
    }
    std::cout << "GPU利用率: " << formatUtilization(info.utilizationGPU) << std::endl;
    std::cout << "内存利用率: " << formatUtilization(info.utilizationMemory) << std::endl;
    std::cout << "内存使用: " << formatMemory(info.memoryUsed) << " / " 
              << formatMemory(info.memoryTotal) << " (" 
              << formatUtilization(static_cast<unsigned int>(100.0 * info.memoryUsed / info.memoryTotal)) << ")" << std::endl;
    std::cout << "功率: " << formatPower(info.powerUsage) << " / " 
              << formatPower(info.powerCapacity) << " (" 
              << formatUtilization(static_cast<unsigned int>(100.0 * info.powerUsage / info.powerCapacity)) << ")" << std::endl;
    
    // 显示进程信息
    if (!info.processes.empty()) {
        std::cout << "运行进程数: " << info.processes.size() << std::endl;
        std::cout << "进程ID\t显存使用\t进程路径" << std::endl;
        for (const auto& process : info.processes) {
            std::string processPath = getProcessPath(process.pid);
            std::cout << process.pid << "\t" 
                      << formatMemory(process.usedGpuMemory) << "\t" 
                      << processPath << std::endl;
        }
    } else {
        std::cout << "无运行进程" << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // 初始化NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "无法初始化NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    
    // 获取设备数量
    unsigned int deviceCount = 0;
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        std::cerr << "无法获取GPU数量: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cout << "未检测到NVIDIA GPU设备" << std::endl;
        nvmlShutdown();
        return 0;
    }
    
    std::cout << "检测到 " << deviceCount << " 台NVIDIA GPU设备" << std::endl;
    
    // 解析命令行参数
    int refreshInterval = 1; // 默认刷新间隔为1秒
    bool runOnce = false;    // 默认持续运行
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            refreshInterval = std::stoi(argv[++i]);
        } else if (arg == "--once") {
            runOnce = true;
        }
    }
    
    // 主循环
    do {
        // 清屏
        std::cout << "\033[2J\033[1;1H";
        
        // 显示时间戳
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "时间: " << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S") << std::endl << std::endl;
        
        // 获取并显示每个GPU的信息
        for (unsigned int i = 0; i < deviceCount; i++) {
            nvmlDevice_t device;
            result = nvmlDeviceGetHandleByIndex(i, &device);
            if (result == NVML_SUCCESS) {
                try {
                    GPUInfo info = getGPUInfo(device);
                    displayGPUInfo(info, i);
                } catch (const std::exception& e) {
                    std::cerr << "获取GPU " << i << " 信息时出错: " << e.what() << std::endl;
                }
            } else {
                std::cerr << "无法获取GPU " << i << " 的句柄: " << nvmlErrorString(result) << std::endl;
            }
        }
        
        if (!runOnce) {
            std::this_thread::sleep_for(std::chrono::seconds(refreshInterval));
        }
    } while (!runOnce);
    
    // 关闭NVML
    nvmlShutdown();
    
    return 0;
} 