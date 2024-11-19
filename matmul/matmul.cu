#include "../include/cuda_aux.cuh"
#define TILE_SIZE 16

__global__ void MatMulKernel(const double *A, const double *B, double *C,
                             const int M, const int K, const int N) {
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int rowoff = threadIdx.y; // row offset in tile
    int coloff = threadIdx.x; // column offset in tile
    
    __shared__ double tA[TILE_SIZE][TILE_SIZE];
    __shared__ double tB[TILE_SIZE][TILE_SIZE];

    double acc = 0.0;
    for (int i = 0; i < K; i += TILE_SIZE) {
        // Load tiles of operand matrices into shared memory
        if (row < M && i + coloff < K)
            tA[rowoff][coloff] = A[row*K + (i+coloff)]; // A[row][i+coloff]
        if (col < N && i + rowoff < K)
            tB[rowoff][coloff] = B[(i+rowoff)*N + col]; // B[i+rowoff][col];
        __syncthreads();

        // Compute matrix multiplication on tile
        for (int l = 0; l < TILE_SIZE; l++) {
            if (row < M && col < N && i + l < K)
                acc += tA[rowoff][l] * tB[l][coloff];
        }
        __syncthreads(); // Guarantee that the operation on tile is done across SM.
    }
    if (row < M && col < N)
        C[row*N + col] = acc;
}

__device__ void DeviceMatMul(const double *A, const double *B, double *C,
                             const int M, const int K, const int N) {
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int rowoff = threadIdx.y; // row offset in tile
    int coloff = threadIdx.x; // column offset in tile
    
    __shared__ double tA[TILE_SIZE][TILE_SIZE];
    __shared__ double tB[TILE_SIZE][TILE_SIZE];

    double acc = 0.0;
    for (int i = 0; i < K; i += TILE_SIZE) {
        // Load tiles of operand matrices into shared memory
        if (row < M && i + coloff < K)
            tA[rowoff][coloff] = A[row*K + (i+coloff)]; // A[row][i+coloff]
        if (col < N && i + rowoff < K)
            tB[rowoff][coloff] = B[(i+rowoff)*N + col]; // B[i+rowoff][col];
        __syncthreads();

        // Compute matrix multiplication on tile
        for (int l = 0; l < TILE_SIZE; l++) {
            if (row < M && col < N && i + l < K)
                acc += tA[rowoff][l] * tB[l][coloff];
        }
        __syncthreads(); // Guarantee that the operation on tile is done across SM.
    }
    if (row < M && col < N)
        C[row*N + col] = acc;
}

__global__ void MatMulNaiveKernel(const double *A, const double *B, double *C,
                                  const int M, const int K, const int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    double acc = 0.0;
    if (row < M && col < N) {
        for (int i = 0; i < K; i++)
            acc += A[row][i] * B[i][col];
    }
}

// Wrapper for GPU Matrix Multiply Kernel
void MatMul(const double *d_A, const double *d_B, double *d_C,
            const int M, const int K, const int N) {
    // GPU threadblock topology
    dim3 GridDim(ceil(N, TILE_SIZE), ceil(M, TILE_SIZE), 1);
    dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);

    MatMulKernel<<<GridDim, BlockDim>>> (
        d_A, d_B, d_C, M, K, N
    );
}

