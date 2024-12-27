#include <iostream>
#include <cstdio>
#include "option_set.hpp"
#include "cuda_aux.cuh"
#include "rand_aux.hpp"
#include "reduce_sum.cuh"
#include "reduce_sum_aux.hpp"

int main(int argc, char *argv[]) {
    // Define ReduceSum Options
    Nix::OptionSet ReduceSumOptions();
    ReduceSumOptions.add_option<int>("N", "Reduce-Sum Array Length");
    ReduceSumOptions.add_option<int>("block_size", "GPU Threadblock Size");
    ReduceSumOptions.add_option<std::string>("execution_mode", 
        "Execution mode\n - 'Functional' : Functionality test\n - 'Performance' : Performance test");
    ReduceSumOptions.add_option<bool>("help", "Print help message");

    ReduceSumOptions.parse_command_line(argc, argv);

    // Check if 'help' is set
    if ()

    // Get Reduce-Sum Array Length
    int N;
    if (!ReduceSumOptions.get_option_value("N", N)) {
        std::cout << "Array length undefined; Set by 1,048,576.\n";
        N = 1048576;
    } else if (N <= 0) {
        std::cout << "Array length(" << N << ") is invalid.\n";
        return 0;
    }

    // Get GPU Threadblock Size
    int num_threads;
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    if (!ReduceSumOptions.get_option_value("block_size", num_threads)) {
        std::cout << "Block size undefined; Set by 512.\n";
        num_threads = 512;
    } else if (num_threads > max_threads_per_block) {
        std::cout << "Your block size (" << num_threads << ") exceeds maximum threads per block (" << max_threads_per_block << ") of your device.\n";
        std::cout << "Block size set by " << max_threads_per_block << ".\n";
        num_threads = max_threads_per_block;
    }

    // Get Execution Mode
    std::string ExecMode;
    if (!ReduceSumOptions.get_option_value("execution_mode", ExecMode)) {
        std::cout << "Execution mode undefined; Set by 'Functional'.\n";
        ExecMode = "Functional";
    }

    if (ExecMode == "Functional") {
        // Generate Array
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
        if (Nix::isClose<double>(ans, res, (double) 0, (double) 1e-7)) {
            std::cout << "ReduceSum Kernel succeeded.\n";
        /*
            if (verbose) {
                std::cout << "Answer(Host) = " << ans << ", Result(Device) = " << res << "\n";
                std::cout << "Relative Error = " << (ans - res) / ans << "\n";
            }
        */
        } else {
            std::cout << "ReduceSum Kernel failed.\n";
            std::cout << "Answer(Host) = " << ans << ", Result(Device) = " << res << "\n";
            std::cout << "Absolute Error = " << abs(ans - res) << "\n";
            std::cout << "Relative Error = " << (ans - res) / ans << "\n";
        }

        // Free resources
        delete[] arr;
        cudaFree(d_1); cudaFree(d_2);
    }
    else if (ExecMode == "Performance") {

    }
    else {
        std::cout << "Wrong Execution Mode (" << ExecMode << ")\n";
    }

    return 0;    
}