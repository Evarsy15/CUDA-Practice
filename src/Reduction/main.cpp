#include <iostream>
#include <cstdio>
#include "cuda_aux.cuh"
#include "rand_aux.hpp"
#include "reduce_sum.cuh"
#include "reduce_sum_aux.hpp"

void showHelp() {
    return;
}

int main(int argc, char *argv[]) {

    // Host section
    int N = 1048576;
    double *arr = new double[N];
    fill_rand_normal(arr, N, 0.2, 1.0);
    double ans = host_reduce_sum<double>(arr, N);
    
    // Device allocation section
    int num_threads = 256;
    double *d_1, *d_2;
    cudaMalloc((void **)&d_1, sizeof(double) * N);
    cudaMalloc((void **)&d_2, sizeof(double) * N);
    cudaMemcpy(d_1, arr, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Device execution section
    double res = ReduceSum<double>(d_1, d_2, N, num_threads);

    // Correctness check
    if (isEqual_double(ans, res, (double)1e-10)) {
        std::cout << "ReduceSum Kernel succeeded.\n";
    } else {
        std::cout << "ReduceSum Kernel failed.\n";
        std::cout << "Answer(Host) = " << ans << ", Result(Device) = " << res << "\n";
        std::cout << "Relative Error = " << (ans - res) / ans << "\n";
    }

    // Free resources
    delete[] arr;
    cudaFree(d_1); cudaFree(d_2);

    return 0;
}