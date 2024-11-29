#ifndef AUX_CUH
#define AUX_CUH

/* 
    ceil(M, N) : Returns ceil(M/N). 
    Main Usage : ceil(...) can be used to compute GPU Kernel Block dimension.
*/
inline __device__ __host__ int ceil(int M, int N) 
    { return (M + N - 1) / N; }

inline __device__ __host__ int floor(int M, int N) 
    { return (M / N); }

__host__ bool isEqual_float(float true_val, float measured_val, const float tolerance) {
    float relative_error = (true_val - measured_val) / true_val;
    return (abs(relative_error) < tolerance);
}

__host__ bool isEqual_double(double true_val, double measured_val, const double tolerance) {
    float relative_error = (true_val - measured_val) / true_val;
    return (abs(relative_error) < tolerance);
}

#endif