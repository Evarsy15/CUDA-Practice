# Makefile
CXX = g++
NVCC = nvcc

CUDA_VERSION = 11.3
CUDA_DIR = /opt/ohpc/pub/apps/cuda-$(CUDA_VERSION)
CUDA_LIB = $(CUDA_DIR)/lib64
CUDA_INC = $(CUDA_DIR)/include

EXE = MLP

CPP = main.cpp mlp.cpp mlp_cpu.cpp
CUDA = mlp_cuda.cu
HDR = common.h mlp.h mlp_cpu.h mlp_cuda.h

all : $(EXE)

$(EXE) : $(CPP) $(CUDA) $(HDR)
	$(NVCC) $(CPP) $(CUDA) -o $@ -L $(CUDA_LIB) -I $(CUDA_INC)

clean :
	rm -f MLP