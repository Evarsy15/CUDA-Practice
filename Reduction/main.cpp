#include <iostream>
#include <cstdio>
#include "cuda_aux.cuh"
#include "rand_aux.hpp"
#include "reduce_sum_aux.hpp"

int main(int argc, char *argv[]) {

    // Host section
    int N = 1048576;
    double *arr = new double[N];
    fill_rand_normal(arr, N, 2.0, 2.0);
    double ans = host_reduce_sum<double>(arr, N);
    
    // Device allocation section
    int num_threads = 256;
    double *d_1, *d_2;
    cudaMalloc((void **)&d_1, sizeof(double) * N);
    cudaMalloc((void **)&d_2, sizeof(double) * N);
    cudaMemcpy(d_1, arr, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Device execution section
    double res = ReduceSum(d_1, d_2, N, num_threads);

    // Correctness check
    if (isEqual_float(ans, res, 1e-6)) {
        std::cout << "Reduce_Sum Kernel succeeded.\n";
    } else {
        std::cout << "Reduce_Sum Kernel failed.\n";
        std::cout << "Answer(Host) = " << ans << ", Result(Device) = " << res << "\n";
    }

    delete[] arr;
    cudaFree(d_1); cudaFree(d_2);

    return 0;
}