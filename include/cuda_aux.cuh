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

#endif