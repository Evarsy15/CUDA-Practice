NVCC = nvcc

HDR      = reduce_sum.hpp reduce_sum_aux.hpp
CPP_SRC  = main.cpp reduce_sum_aux.cpp
CUDA_SRC = reduce_sum.cu
SRC = $(CPP_SRC) $(CUDA_SRC)
EXE = reduce_sum

all : $(EXE)

$(EXE) : $(SRC) $(HDR)
	$(NVCC) $(SRC) -o $(EXE)

clean :
	rm -f $(OBJ) $(EXE)