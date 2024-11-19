#include "mlp_cuda.h"
#define TILE_SIZE 16  // Config. of Tiled Matrix Multiplication

/*
    Auxiliary functions 
                        */
inline __device__ __host__ int ceil(int M, int N) {
    return (M + N - 1) / N;
}

inline __device__ __host__ double max_double(double x, double y) {
    return (x > y ? x : y);
}

/*
    Transpose(A, A_T, M, N) compute the transpose of M×N matrix A into A_T.
*/
__global__ void Transpose(const double *A, double *A_T, const int M, const int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ double tile[TILE_SIZE][TILE_SIZE+1];
    if (row < M && col < N)
        tile[threadIdx.y][threadIdx.x] = A[row*N + col];

    if (row < M && col < N)
        A_T[col*M + row] = tile[threadIdx.y][threadIdx.x];
}

/****
        Device kernels & functions
                                    ****/

/*
    MatMul(A, B, C, M, K, N) computes the general matrix product C = AB
    where [A : M×K], [B : K×N] are operand matrices and [C : M×N] is result matrix.
*/
__device__ void MatMul(const double *A, const double *B, double *C,
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

/*
    BiasAddition(B, N, idata, bias, odata) computes bias addition row-wise:
        odata[k][i] = idata[k][i] + bias[i]
*/
__device__ void BiasAdditionNaive(int B, int N, double *idata, double *bias, double *odata) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int pos = row * N + col;

    if (row < B && col < N)
        odata[pos] = idata[pos] + bias[col];
}

/* __device__ void BiasAddition(int B, int N, double *idata, double *bias, double *odata) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int pos = row * N + col;

    __shared__ double s_bias[TILE_SIZE];
    __shared__ bool b_s_bias[TILE_SIZE];
    if (atomicExch(&b_s_bias[threadIdx.x], true) == false) {
        s_bias[threadIdx.x] = bias[row];
    }

    if (row < N && col < B)
        odata[pos] = idata[pos] + s_bias[threadIdx.x];
}
*/

/*
    ReLUActivation(B, N, idata, odata) applies ReLU activation function 
    to matrix 'idata' element-wisely, saving results in 'odata'.
*/
__device__ void ReLUActivation(int B, int N, double *idata, double *odata) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int pos = row * N + col;

    if (row < B && col < N)
        odata[pos] = idata[pos] >= 0.0 ? idata[pos] : 0.0;
}

/*
    PerceptronLayerKernel(...) is the integraion of a single layer perceptron.
    ※ It supports batching case (multiple inputs). 
*/
__global__ void PerceptronLayerKernel(int input_dim, int output_dim, int batch_size,
                                      double* inputs, double* weights, double* biases, 
                                      double* intermediates, double* outputs, bool apply_activation) {
    
    // Apply weight into bundle of input vectors
    MatMul(inputs, weights, intermediates, batch_size, input_dim, output_dim);
    
    // Apply bias into intermediate result
    BiasAdditionNaive(batch_size, output_dim, intermediates, biases, intermediates);
        
    if (apply_activation) {
        // Apply activation function element-wise
        ReLUActivation(batch_size, output_dim, intermediates, outputs);
    }
}

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

__global__ void MatMulNaiveKernel(const double *A, const double *B, double *C,
                             const int M, const int K, const int N) {
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    double acc = 0.0;
    if (row < M && col < N) {
        for (int i = 0; i < K; i++) {
            acc += A[row*K + i] * B[i*N + col];
        }
        C[row*N + col] = acc;
    }
}

/*
    ReduceMax(N, src, aux) finds the maximum element of sub-vector allocated to threadblock
    with same level of precision (providing numerical stability).
*/
__device__ double ReduceMax(int N, double *src, double *aux) {
    int offset = threadIdx.x;
    int active = N; // Length of active section
    
    // Load source array into shared memory
    if (offset < N)
        aux[offset] = src[offset];
    __syncthreads();

    // Find maximum element of the sub-array
    // ※ Read 'Reduction' in report.
    for (; active > 1; active = ceil(active, 2)) {
        int stride = ceil(active, 2);
        if (offset + stride < active) {
            aux[offset] = max_double(aux[offset], aux[offset + stride]);
        }
        __syncthreads();
    }

    return aux[0];
}

/*
    BroadcastSub(N, idata, sub, doata) subtracts given vector 'idata'
    by provided value 'sub' and save into 'odata'.
        odata[i] = idata[i] - sub
*/
__device__ void BroadcastSub(int N, double *idata, double sub, double *odata) {
    int idx = threadIdx.x;
    if (idx < N)
        odata[idx] = idata[idx] - sub;
}

/*
    ElementwiseExp(N, idata, odata) applies elementwise exponential function.
        odata[i] = exp(idata[i])
*/
__device__ void ElementwiseExp(int N, double *idata, double *odata) {
    int idx = threadIdx.x;
    if (idx < N)
        odata[idx] = exp(idata[idx]);
}

/*
    ReduceSum(N, src, dst) computes the sum of elements of sub-vector allocated to threadblock.
*/
__device__ double ReduceSum(int N, double *src, double *aux) {
    int offset = threadIdx.x;
    int active = N; // Length of active section
    
    // Load source array into shared memory
    if (offset < N)
        aux[offset] = src[offset];
    __syncthreads();

    // Find sum of elements in sub-array
    for (; active > 1; active = ceil(active, 2)) {
        int stride = ceil(active, 2);
        if (offset + stride < active) {
            aux[offset] += aux[offset + stride];
        }
        __syncthreads();
    }

    return aux[0];
}

/*
    BroadcastDiv(N, idata, div, odata) multiplies [1/div] to each element of idata & save into odata.
*/
__device__ void BroadcastDiv(int N, double *idata, double div, double *odata) {
    int idx = threadIdx.x;
    if (idx < N)
        odata[idx] = idata[idx] / div;
}

/*
    SoftmaxKernel(...)
*/
__global__ void SoftmaxKernel(double *inputs, double *results) {
    int batch_id   = blockIdx.x;
    int vector_dim = blockDim.x;
    
    double *alloc_input  = inputs  + (batch_id * vector_dim);
    double *alloc_result = results + (batch_id * vector_dim);

    extern __shared__ double aux[];

    double max_elt = ReduceMax(vector_dim, alloc_input, aux);
    BroadcastSub  (vector_dim, alloc_input, max_elt, alloc_input);
    ElementwiseExp(vector_dim, alloc_input, alloc_input);
    double exp_sum = ReduceSum(vector_dim, alloc_input, aux);
    BroadcastDiv  (vector_dim, alloc_input, exp_sum, alloc_result);
}



/****
        Host functions
                        ****/

void MLP_CUDA::allocate_gpu_memory() {
    // Start timer
    timer().startCpuTimer();

    int batch_size = params.batch_size;
    for (int i = 0; i < params.layer_count; i++) {
        int input_dim  = (i > 0) ? params.layer_sizes[i-1] : params.input_size;
        int output_dim = params.layer_sizes[i];
        
        double *d_layer_w, *d_layer_b, *d_layer_z;
        
        cudaMalloc((void **) &d_layer_w, input_dim  * output_dim * sizeof(double));
        cudaMalloc((void **) &d_layer_b, output_dim * sizeof(double));
        cudaMalloc((void **) &d_layer_z, batch_size * output_dim * sizeof(double));
        
        d_w.push_back(d_layer_w);
        d_b.push_back(d_layer_b);
        d_z.push_back(d_layer_z);
    }

    for (int i = 0; i <= params.layer_count; i++) {
        int input_dim = (i > 0) ? params.layer_sizes[i-1] : params.input_size;
        double *d_layer_a;
        cudaMalloc((void **) &d_layer_a, batch_size * input_dim * sizeof(double));
        d_a.push_back(d_layer_a);
    }

    // End timer
    timer().endCpuTimer();
}

// TODO : Apply CUDA Stream
void MLP_CUDA::copy_mlp_into_gpu() {
    // Start timer
    timer().startCpuTimer();

    for (int i = 0; i < params.layer_count; i++) {
        int input_dim  = (i > 0) ? params.layer_sizes[i-1] : params.input_size;
        int output_dim = params.layer_sizes[i];

        // Copy weight and bias of i-th layer
        cudaMemcpy(d_w[i], w[i], input_dim * output_dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[i], b[i], output_dim * sizeof(double), cudaMemcpyHostToDevice);
        
        /* 
            Note : Weights in './weights.csv' is written in column-wise format,
                   or equivalently, is already a transposed weight in row-wise format.
                   (See <mlp_cpu.cpp>:MLP_CPU::matmul(...) to deduce it.)
        */
    }
    
    // End timer
    cudaDeviceSynchronize(); // for precise timing estimation
    timer().endCpuTimer();
}

void MLP_CUDA::free_gpu_memory() {
    for (int i = 0; i < params.layer_count; i++) {
        cudaFree(d_w[i]);
        cudaFree(d_b[i]);
        cudaFree(d_z[i]);
        cudaFree(d_a[i]);
    }
    cudaFree(d_a[params.layer_count]);
}

void MLP_CUDA::__forward(vector<double*> &data, double *res, int start, int batch_size) {
    // Start timer
    timer().startCpuTimer();

    int input_size  = params.input_size;
    int output_size = params.output_size;
    int layer_count = params.layer_count;

    // Copy batched input data into GPU
    for (int i = 0; i < batch_size; i++) {
        cudaMemcpy(d_a[0] + (i * input_size), data[i + start],
                   input_size * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Process input vectors into MLP
    for (int i = 0; i < layer_count; i++) {
        int  input_dim  = (i > 0) ? params.layer_sizes[i-1] : input_size;
        int  output_dim = params.layer_sizes[i];
        bool apply_activation = (i < layer_count-1) ? true : false;

        dim3 grid_dim(ceil(output_dim, TILE_SIZE), ceil(batch_size, TILE_SIZE), 1);
        dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);

        // Perceptron Layer Kernel
        PerceptronLayerKernel<<<grid_dim, block_dim>>> (
            input_dim, output_dim, batch_size,
            d_a[i], d_w[i], d_b[i], d_z[i], d_a[i+1], apply_activation
        );
        // cudaDeviceSynchronize(); // Wait for device(GPU) to finish the entire kernel.
    }

    // Convert the result into probability vectors using Softmax
    int shared_mem_size = output_size * sizeof(double);
    SoftmaxKernel<<<output_size, batch_size, shared_mem_size>>> (
        d_z[layer_count-1], d_a[layer_count]
    );
    // cudaDeviceSynchronize();

    // Copy final result in device(GPU) to host(CPU)
    cudaMemcpy(res, d_a[layer_count], batch_size * output_size * sizeof(double), cudaMemcpyDeviceToHost);

    // End timer
    timer().endCpuTimer();

    // Copy intermediates for memory matching purpose
    // ※ This process is excluded for time measuring because 
    //    copying intermediates into host is not necessary in real application of MLP.
    
    /*
    cudaStream_t load_z, load_a;
    cudaStreamCreate(&load_z); cudaStreamCreate(&load_a);
    for (int i = 0; i < layer_count; i++) {
        int input_dim  = (i > 0) ? params.layer_sizes[i-1] : params.input_size;
        int output_dim = params.layer_sizes[i];

        cudaMemcpyAsync(z[i], d_z[i], batch_size * output_dim * sizeof(double), 
                        cudaMemcpyDeviceToHost, load_z);
        cudaMemcpyAsync(a[i+1], d_a[i], batch_size * input_dim  * sizeof(double), 
                        cudaMemcpyDeviceToHost, load_a);
    }
    cudaStreamDestroy(load_z); cudaStreamDestroy(load_a);
    */
}

double* MLP_CUDA::forward(vector<double*> &data, int num_data) {
    int output_size = params.output_size;
    int batch_size  = params.batch_size;
    
    // MLP Result
    double *res = new double[num_data * output_size];
    double *cur_res = res;

    this->total_time_forward = 0.0;

    int start_pos   = 0;
    int num_remains = num_data; // number of remaining data
    for (; start_pos < num_data; start_pos += batch_size, num_remains -= batch_size) {
        // Proceed Batched Forward
        int current_batch_size = num_remains < batch_size ? num_remains : batch_size;
        __forward(data, cur_res, start_pos, current_batch_size);

        // Move cursor of result
        cur_res += current_batch_size * output_size;

        // Accumulate MLP Execution Time
        this->total_time_forward += timer().getCpuElapsedTimeForPreviousOperation();
    }

    return res;
}

double* MLP_CUDA::forward_single(double *data) {
    timer().startCpuTimer();

    int input_size  = params.input_size;
    int output_size = params.output_size;
    int layer_count = params.layer_count;
    int batch_size  = 1; // single forward

    double *res = new double[output_size];

    cudaMemcpy(d_a[0], data, input_size * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < layer_count; i++) {
        int  input_dim  = (i > 0) ? params.layer_sizes[i-1] : input_size;
        int  output_dim = params.layer_sizes[i];
        bool apply_activation = (i < layer_count-1) ? true : false;

        dim3 grid_dim(ceil(output_dim, TILE_SIZE), ceil(batch_size, TILE_SIZE), 1);
        dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);

        // Perceptron Layer Kernel
        PerceptronLayerKernel<<<grid_dim, block_dim>>> (
            input_dim, output_dim, batch_size,
            d_a[i], d_w[i], d_b[i], d_z[i], d_a[i+1], apply_activation
        );
        cudaDeviceSynchronize();
    }

    int shared_mem_size = output_size * sizeof(double);
    SoftmaxKernel<<<batch_size, output_size, shared_mem_size>>> (
        d_z[layer_count-1], d_a[layer_count]
    );

    cudaMemcpy(res, d_a[layer_count], output_size * sizeof(double), cudaMemcpyDeviceToHost);
    timer().endCpuTimer();

    return res;
}