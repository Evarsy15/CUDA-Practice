#include <torch/types.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define tile_size 32

// Implement naive_attention kernel below
// __global__ void matmul() {}
// __global__ void reduce_sum() {}
// __global__ void reduce_max() {}
// ...

__forceinline__ __host__ __device__ unsigned int ceil(unsigned int M, unsigned int N) {
    // assert(N != 0);
    return (M + N - 1) / N;
}

__forceinline__ __device__ double max_float(const float a, const float b) {
    return (a > b ? a : b);
}

/*
    transpose(A, A_T, M, N) :
        Transpose M×N-matrix A into A_T.
*/
__global__ void transpose(const float *A, float *A_T, const int M, const int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float tileA[tile_size][tile_size+1];
    if (row < M && col < N)
        tileA[threadIdx.y][threadIdx.x] = A[row*N + col];

    if (row < M && col < N)
        A_T[col*M + row] = tileA[threadIdx.y][threadIdx.x];
}

/*
    scaled_dot_product(Q, K, S, a, N, D):
        Compute scaled-dot-product S = a * (Q * K_T); S_ij = a * <q_i, k_j>
    
    N : Sequence Length
    D : Embedding Size

    Q : L×E-matrix of Query token
    K : L×E-matrix of Key token
*/
__global__ void scaled_dot_product(const float *Q, const float *K, float *S, 
                                   const float a, const int N, const int D) {
    // IDs for allocated matrix in the matrix pile, described by torch tensor.
    int pile = blockIdx.z;

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int rowoff = threadIdx.y;
    int coloff = threadIdx.x;

    __shared__ float tile_Q[tile_size][tile_size+1];
    __shared__ float tile_K[tile_size][tile_size+1];

    float acc = 0.0;
    for (int i = 0; i < E; i += tile_size) {
        // Load tiles of operand matrices into shared memory
        if (row < N && i + coloff < D) {
            tile_Q[rowoff][coloff] = Q[pile*(N*D) + row*D + (i+coloff)];
            tile_K[rowoff][coloff] = K[pile*(N*D) + row*D + (i+coloff)];
        }
        __syncthreads();

        // Compute partial mat-mul on tile
        for (int l = 0; l < tile_size; l++) {
            if (row < N && col < N && i+l < D)
                acc += tile_Q[rowoff][l] * tile_K[coloff][l];
        }
        __syncthreads();
    }

    if (row < N && col < N)
        S[pile*(N*N) + row*N + col] = acc * a;
}

/*
    reduce_max_sub_exp(S, N):
        Computes row-wise reduce-max, broadcast-sub and elementwise-exp in a row.
    ※ Assuming that N <= MAX_THREADS_PER_BLOCK. Otherwise the kernel should be re-designed.
*/
__global__ void reduce_max_sub_exp(float *S, int N) {
    // blockIdx.y : pile id = batch_id * num_head + head_id
    int pile = blockIdx.y;
    int row  = blockIdx.x;
    int col  = threadIdx.x;
    int idx  = pile*(N*N) + row*N + col;

    // Load target row-vector into shared memory
    extern __shared__ float aux[];
    if (col < N)
        aux[col] = S[idx];
    __syncthreads();
        
    // Process Reduce-max on shared memory vector 'aux'
    for (; active > 1; active = ceil(active, 2)) {
        int stride = ceil(active, 2);
        if (col + stride < active)
            aux[col] = max_float(aux[col], aux[col + stride]);
        __syncthreads();
    }
        
    // Compute exp(S-m) on each row
    S[idx] = exp(S[idx] - aux[0]);
}

/*
    reduce_sum_div(S, N):
        Compute row-wise reduce-sum and broadcast-div in a row.
    ※ Assuming that N <= MAX_THREADS_PER_BLOCK. Otherwise the kernel should be re-designed.
*/
__global__ void reduce_sum_div(float *S, int N) {
    // blockIdx.y : pile id = batch_id * num_head + head_id
    int pile = blockIdx.y;
    int row  = blockIdx.x;
    int col  = threadIdx.x;
    int idx  = pile*(N*N) + row*N + col;

    // Load target row-vector into shared memory
    extern __shared__ float aux[];
    if (col < N)
        aux[col] = S[idx];
    __syncthreads();
        
    // Process Reduce-max on shared memory vector 'aux'
    for (; active > 1; active = ceil(active, 2)) {
        int stride = ceil(active, 2);
        if (col + stride < active)
            aux[col] += aux[col + stride];
        __syncthreads();
    }
        
    // Compute exp(S-m) on each row
    S[idx] = S[idx] / aux[0];
}

/*
    piled_matrix_multiply(P, V, O, M, K, N):
        Computes matrix-multiply O[b][h] = P[b][h] * V[b][h]
*/
__global__ void matrix_multiply(const float *P, const float *V, float *O,
                                const int M, const int K, const int N) {
    // Matrix Pile ID
    int pile = blockDim.z * blockIdx.z + threadIdx.z;

    // Row and Column
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int rowoff = threadIdx.y; // row offset in tile
    int coloff = threadIdx.x; // column offset in tile
    
    __shared__ float tileA[tile_size][tile_size];
    __shared__ float tileB[tile_size][tile_size];

    float acc = 0.0;
    for (int i = 0; i < K; i += tile_size) {
        // Load tiles of operand matrices into shared memory
        if (row < M && i + coloff < K)
            tileA[rowoff][coloff] = A[pile*(M*K) + row*K + (i+coloff)]; // A[row][i+coloff]
        if (col < N && i + rowoff < K)
            tileB[rowoff][coloff] = B[pile*(K*N) + (i+rowoff)*N + col]; // B[i+rowoff][col];
        __syncthreads();

        // Compute matrix multiplication on tile
        for (int l = 0; l < tile_size; l++) {
            if (row < M && col < N && i + l < K)
                acc += tileA[rowoff][l] * tileB[l][coloff];
        }
        __syncthreads(); // Guarantee that the operation on tile is done across SM.
    }
    if (row < M && col < N)
        C[pile*(M*N) + row*N + col] = acc;
}


/*
    row_wise_softmax(R, S) : 
        Apply softmax each row of matrix R[b][h], save into row of matrix S[b][h] of same position.
*/
inline __device__ double reduce_max(int N, double *src, double *aux) {
    int offset = threadIdx.x;
    int active = N; // Length of active section
    
    // Load source array into shared memory
    if (offset < N)
        aux[offset] = src[offset];

    // Find maximum element of the sub-array
    // ※ Read 'Reduction' in report.
    for (; active > 1; active = ceil(active, 2)) {
        __syncthreads();
        int stride = ceil(active, 2);
        if (offset + stride < active)
            aux[offset] = max_double(aux[offset], aux[offset + stride]);
    }

    return aux[0];
}

inline __device__ double reduce_sum() {
    int offset = threadIdx.x;
    int active = N; // Length of active section
    
    // Load source array into shared memory
    if (offset < N)
        aux[offset] = src[offset];

    // Find sum of elements in sub-array
    for (; active > 1; active = ceil(active, 2)) {
        __syncthreads();
        int stride = ceil(active, 2);
        if (offset + stride < active)
            aux[offset] += aux[offset + stride];        
    }

    return aux[0];
}

torch::Tensor naive_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);	// Batch size
    const int nh = Q.size(1);	// Number of heads
    const int N = Q.size(2);	// Sequence size
    const int d = Q.size(3);	// Embedding size

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto T = torch::zeros({B, nh, N, N}); // Temporal memory for storing QK^T/√d
    // auto l = torch::zeros({B, nh, N});
    // auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    O = O.to(device);
    T = T.to(device);
    // l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    int max_sram_size;
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    printf("Max Threads per Block : %d\n", max_threads_per_block);
    printf("Max shared memory: %d\n", max_sram_size);

    // ============= Call kernels ==================
    // matmul<<>>();
    // softmax<<>>();
    // ..
    // =============================================
    float scale = (float) ((double) 1.0 / sqrt(d));

    int  num_tiles = ceil(N, tile_size);
    dim3 ker1_GridDim(num_tiles, num_tiles, B*nh);
    dim3 ker1_BlockDim(tile_size, tile_size, 1);
    scaled_dot_product<<<ker1_GridDim, ker1_BlockDim>>> (
        Q, K, T, scale, N, d
    );

    dim3 ker2_GridDim(N, B*nh, 1);
    dim3 ker2_BlockDim(N, 1, 1);
    int  ker2_smem_size = sizeof(float) * N;
    reduce_max_sub_exp<<<ker2_GridDim, ker2_BlockDim, ker2_smem_size>>> (T, N);
    reduce_sum_div<<<ker2_GridDim, ker2_BlockDim, ker2_smem_size>>> (T, N);

    dim3 ker3_GridDim(ceil(d, tile_size), ceil(N, tile_size), B*nh);
    dim3 ker3_BlockDim(tile_size, tile_size, 1);
    matrix_multiply<<<ker3_GridDim, ker3_BlockDim>>> (
        T, V, O, N, N, d
    )

    // Return output
    O = O.to(torch::kCPU);
    return O;
}
