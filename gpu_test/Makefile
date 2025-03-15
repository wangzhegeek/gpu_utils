CUDA_PATH = /usr/local/cuda-12.4
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -O3
LDFLAGS = -lcublas

TARGET = matrix_multiply
SRC = matrix_multiply.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 