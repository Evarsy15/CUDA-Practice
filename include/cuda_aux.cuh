#ifndef AUX_CUH
#define AUX_CUH

#include <cuda.h>
#include <cuda_runtime.h>

/* 
    ceil(M, N) : Returns ceil(M/N). 
    Main Usage : ceil(...) can be used to compute GPU Kernel Block dimension.
*/
inline __device__ __host__ int ceil(int M, int N) 
    { return (M + N - 1) / N; }

inline __device__ __host__ int floor(int M, int N) 
    { return (M / N); }


/* 
    Correctness check functions
*/
namespace Nix {

template <typename data_t>
inline __host__ __device__ bool isEqual(data_t true_val, data_t measured_val, const data_t tolerance) {
    data_t relative_error = (true_val - measured_val) / true_val;
    return (abs(relative_error) < tolerance);
}

template __host__ __device__ bool isEqual<float>(float true_val, float measured_val, const float tolerance);
template __host__ __device__ bool isEqual<double>(double true_val, double measured_val, const double tolerance);

}

#endif