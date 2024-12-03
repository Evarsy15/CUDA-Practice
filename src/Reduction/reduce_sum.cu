#include "../include/cuda_aux.cuh"

/*
    LocalReduceSum(src, N) : In-Block Reduce-Sum
*/
template <typename dType>
__device__ dType LocalReduceSum(dType *src, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.x;
    extern __shared__ dType aux[];

    if (idx < N)
        aux[offset] = src[idx];
    else
        aux[offset] = (dType) 0; // Additive Identity
    __syncthreads();

    int active, stride; // ReduceSum's active length & stride
    for (; active > 1; active = ceil(active, 2)) {
        int stride = ceil(active, 2);
        if (offset + stride < active) {
            aux[offset] += aux[offset + stride];
        }
        __syncthreads();
    }

    return aux[0];
}

template <typename dType>
__global__ void LocalReduceSumKernel(dType *src, dType *aux, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.x;  // offset in shared memory
    
    extern __shared__ dType temp[];
    if (idx < N)
        temp[offset] = src[idx];
    else
        temp[offset] = (dType) 0; // Additive Identity
    __syncthreads();

    int active, stride; // ReduceSum's active length & stride
    for (; active > 1; active = ceil(active, 2)) {
        int stride = ceil(active, 2);
        if (offset + stride < active)
            temp[offset] += temp[offset + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        aux[blockIdx.x] = temp[0];
}

template <typename dType>
dType ReduceSum(dType *src, dType *aux, int N, int block_size) {
    int working_size = N;
    for (int num_block = ceil(working_size, block_size); working_size >= block_size;
                working_size = num_block, num_block = ceil(num_block, block_size)) {
        // Get local reduce-sum
        int shared_mem_size = sizeof(dType) * block_size;
        LocalReduceSumKernel<<<num_block, block_size, shared_mem_size>>>(
            src, aux, working_size
        );

        // 
    }
}