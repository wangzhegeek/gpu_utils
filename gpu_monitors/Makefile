CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
CUDA_PATH = /usr/local/cuda-12.4
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/targets/x86_64-linux/lib/stubs -lnvidia-ml

NVCC = nvcc
CUDA_FLAGS = -O2

TARGET = gpu_monitor
SRC = gpu_monitor.cpp

all: $(TARGET) cuda_stress_test

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

cuda_stress_test: cuda_stress_test.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET) cuda_stress_test

.PHONY: all clean 