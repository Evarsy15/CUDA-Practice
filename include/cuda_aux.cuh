#ifndef AUX_CUH
#define AUX_CUH

#include <cstdlib>
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
inline __host__ __device__ bool isClose(data_t true_val, data_t measured_val, double abs_tol, double rel_tol) {
    assert(abs_tol >= 0.0 && rel_tol >= 0.0);
    return (double) abs(true_val - measured_val) < (abs_tol + rel_tol * (double) true_val);
}

template __host__ __device__ bool isClose<half>  (half   true_val, half   measured_val, double abs_tol, double rel_tol);
template __host__ __device__ bool isClose<float> (float  true_val, float  measured_val, double abs_tol, double rel_tol);
template __host__ __device__ bool isClose<double>(double true_val, double measured_val, double abs_tol, double rel_tol);

template <typename data_t>
inline __host__ bool isCloseArray(const data_t *true_val, const data_t *measured_val, int N, double abs_tol, double rel_tol) {
    for (int i = 0; i < N; i++) {
        if (!isClose(true_val[i], measured_val[i], abs_tol, rel_tol))
            return false;
    }
    return true;
}

#endif