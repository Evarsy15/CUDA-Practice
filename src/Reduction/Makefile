NVCC = nvcc
CUDA_VERSION = 11.3

HDR      = reduce_sum_aux.hpp reduce_sum.cuh
CPP_SRC  = main.cpp reduce_sum_aux.cpp
CUDA_SRC = reduce_sum.cu
SRC = $(CPP_SRC) $(CUDA_SRC)

# CUDA_INSTALL_PATH = <your_cuda_path>
CUDA_INSTALL_PATH = /opt/ohpc/pub/apps/cuda-$(CUDA_VERSION)
CUDA_LIBRARY  = $(CUDA_INSTALL_PATH)/lib64
CUDA_INCLUDE  = $(CUDA_INSTALL_PATH)/include
LOCAL_INCLUDE = ../../include
INC = $(CUDA_INCLUDE) $(LOCAL_INCLUDE)
LIB = $(CUDA_LIBRARY)

EXE = reduce_sum

all : $(EXE)

$(EXE) : $(SRC) $(HDR)
	$(NVCC) $(SRC) -o $(EXE) -I $(CUDA_INCLUDE) -I $(LOCAL_INCLUDE) -I ./ -L $(LIB)

clean :
	rm -f $(EXE)