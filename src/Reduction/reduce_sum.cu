#include "cuda_aux.cuh"
#include <cstdio>

/*
    LocalReduceSum(src, N) : In-Block Reduce-Sum
*/
template <typename data_t>
__device__ data_t LocalReduceSum(data_t *src, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.x;
    extern __shared__ data_t aux[];

    if (idx < N)
        aux[offset] = src[idx];
    else
        aux[offset] = (data_t) 0; // Additive Identity
    __syncthreads();

    int active, stride; // ReduceSum's active length & stride
    for (active = blockDim.x; active > 1; active = ceil(active, 2)) {
        stride = ceil(active, 2);
        if (offset + stride < active) {
            aux[offset] += aux[offset + stride];
        }
        __syncthreads();
    }

    return aux[0];
}

template <typename data_t>
__global__ void LocalReduceSumKernel(data_t *src, data_t *aux, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.x;  // offset in shared memory
    
    extern __shared__ data_t temp[];
    if (idx < N)
        temp[offset] = src[idx];
    else
        temp[offset] = (data_t) 0; // Additive Identity
    __syncthreads();

    int active, stride; // ReduceSum's active length & stride
    for (active = blockDim.x; active > 1; active = ceil(active, 2)) {
        stride = ceil(active, 2);
        if (offset + stride < active)
            temp[offset] += temp[offset + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        aux[blockIdx.x] = temp[0];
}

template <typename data_t>
data_t ReduceSum(data_t *src, data_t *aux, int N, int block_size) {
    int working_size = N;
    int num_block;
  // int iter = 0;  # For debugging

    do {
        // Compute # of GPU Thread-blocks
        num_block = ceil(working_size, block_size);
    /*
        printf("Iteration %d : \n", iter);
        printf("Working-size = %d, Grid-Dim = %d, Block-Dim = %d\n",
                        working_size, num_block, block_size);
    */

        // Get local reduce-sum
        int shared_mem_size = sizeof(data_t) * block_size;
        LocalReduceSumKernel<<<num_block, block_size, shared_mem_size>>>(
            src, aux, working_size
        );

        // Swap pointer
        data_t *tmp = aux; aux = src; src = tmp;

        // Update working-size
        working_size = num_block;
    //  iter++;
    } while (num_block > 1);
    
    data_t res;
    cudaMemcpy(&res, src, sizeof(data_t), cudaMemcpyDeviceToHost);
    return res;
}

// Explicit instantiation
template double ReduceSum<double>(double *src, double *aux, int N, int block_size);