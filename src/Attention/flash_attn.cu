#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__forceinline__ __host__ __device__ unsigned int ceil(unsigned int M, unsigned int N) {
    // assert(N != 0);
    return (M + N - 1) / N;
}

__forceinline__ __device__ double max_float(const float a, const float b) {
    return (a > b ? a : b);
}

/*
    FlashAttention : Memory-efficient Attention implementation provided by
        [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Tri Dao, et al., NeurIPS’22]
*/
__global__ void flash_attention(const float *Q, const float *K, const float *V, float *O, float *m, float *l,
                                int N, int d, int B_r, int B_c, int T_r, int T_c) {
    int pb_mat = blockIdx.z * (N * d); // pile base for (N × d)-matrix
    int pb_vec = blockIdx.z * N;       // pile base for N-dim vector

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    extern __shared__ float SRAM[];
    float *_Q_i_ = SRAM;                 // Q_i : (B_r × d)-matrix
    float *_K_j_ = _Q_i_ + (B_r*d);      // K_j : (B_c × d)-matrix
    float *_V_j_ = _K_j_ + (B_c*d);      // V_j : (B_c × d)-matrix
    float *_O_i_ = _V_j_ + (B_c*d);      // O_i : (B_r × d)-matrix
    float *_m_i_ = _O_i_ + (B_r*d);      // m_i : (B_r)-dim vector
    float *_l_i_ = _m_i_ + (B_r);        // l_i : (B_r)-dim vector
    float *_m_ij_ = _l_i_ + (B_r);       // ~m_ij : (B_r)-dim vector
    float *_l_ij_ = _m_ij_ + (B_r);      // ~l_ij : (B_r)-dim vector
    float *_m_i_new_ = _l_ij_ + (B_r);   // m_i_new : (B_r)-dim vector
    float *_l_i_new_ = _m_i_new_ + (B_r); // l_i_new : (B_r)-dim vector
    float *_S_ij_ = _l_i_new_ + (B_r);    // S_ij : (B_r × B_c)-matrix

    /*
        GPU Threadblock is of size (B_c, B_r, 1) which corresponds to the
        (B_r × B_c)-matrix, which is the size of [(Q_i) * (K_j)^T].
    */
    for (int J = 0; J < T_c; J++) {
        // Load K_j, V_j into SRAM
        for (int i = 0; i < B_c; i += B_r) {   // y-axis
            for (int j = 0; j < d; j += B_c) { // x-axis
                int r = i + ty;
                int c = j + tx;
                if (r < B_c && c < d) {
                    _K_j_[r*d + c] = K[pb_mat + (B_c*J + r)*d + c];
                    _V_j_[r*d + c] = V[pb_mat + (B_c*J + r)*d + c];
                }
            }
        }
        __syncthreads();

        for (int I = 0; I < T_r; I++) {
            // Load Q_i, O_i, l_i, m_i into SRAM
            for (int j = 0; j < d; j += B_c) {
                int r = ty;
                int c = j + tx;
                if (c < d) {
                    _Q_i_[r*d + c] = Q[pb_mat + (B_r*I + r)*d + c];
                    _O_i_[r*d + c] = O[pb_mat + (B_r*I + r)*d + c];
                }
            }
            if (ty == 0 && tx < B_r) // tx-first to reduce warp divergence
                _m_i_[tx] = m[pb_vec + B_r*I + tx];
            if (ty == 1 && tx < B_r)
                _l_i_[tx] = l[pb_vec + B_r*I + tx];
            __syncthreads();

            // Compute S_ij = (Q_i) * (K_j)^T
            float acc = 0.0;
            for (int k = 0; k < d; k++)
                acc += _Q_i_[ty*d + k] * _K_j_[tx*d + k];
            _S_ij_[ty*B_c + tx] = acc / sqrtf(d);
            __syncthreads();

            // Compute ~m_ij = rowmax(S_ij)
            if (ty == 0) {
                if (tx < B_r) {
                    _m_ij_[tx] = _S_ij_[tx*B_c];
                    for (int k = 1; k < B_c; k++)
                        _m_ij_[tx] = max_float(_m_ij_[tx], _S_ij_[tx*B_c + k]);
                }
            }
            __syncthreads();

            // Compute P_ij = exp(S_ij - ~m_ij)
            _S_ij_[ty*B_c + tx] = exp(_S_ij_[ty*B_c + tx] - _m_ij_[ty]);
            __syncthreads();

            // Compute ~l_ij = rowsum(P_ij)
            if (ty == 0) {
                if (tx < B_r) {
                    _l_ij_[tx] = _S_ij_[tx*B_c];
                    for (int k = 1; k < B_c; k++)
                        _l_ij_[tx] = _l_ij_[tx] + _S_ij_[tx*B_c + k];
                }
            }
            __syncthreads();

            // Compute m_i_new = max(m_i, ~m_ij) and
            //         l_i_new = exp(m_i - m_i_new) * l_i + exp(~m_ij - m_i_new) * ~l_ij
            if (ty == 0) {
                if (tx < B_r) {
                    _m_i_new_[tx] = max_float(_m_i_[tx], _m_ij_[tx]);
                    _l_i_new_[tx] = exp(_m_i_[tx] - _m_i_new_[tx]) * _l_i_[tx] 
                                    + exp(_m_ij_[tx] - _m_i_new_[tx]) * _l_ij_[tx];
                }
            }
            __syncthreads();

            // Write O_i <- inv_diag(l_i_new) * [diag(l_i) * exp(m_i - m_i_new) * O_i + exp(~m_ij - m_i_new) * P_ij * V_j]
            // Note : O_i is (B_r × d)-matrix.
            for (int j = 0; j < d; j += B_c) {
                int r = ty;
                int c = j + tx;
                if (c < d) {
                    // diag(l_i) * exp(m_i - m_i_new) * O_i
                    _O_i_[r*d + c] *= _l_i_[r] * exp(_m_i_[r] - _m_i_new_[r]);
                    // exp(~m_ij - m_i_new) * (P_ij * V_j)
                    float acc = 0.0;
                    for (int k = 0; k < B_c; k++)
                        acc += _S_ij_[r*B_c + k] * _V_j_[k*d + c];
                    _O_i_[r*d + c] += exp(_m_ij_[r] - _m_i_new_[r]) * acc;

                    // Write O_i into HBM
                    O[pb_mat + (B_r*I + r)*d + c] = _O_i_[r*d + c] / _l_i_new_[r];
                }
            }
            __syncthreads();

            // Write m_i_new and l_i_new into HBM
            if (ty == 0 && tx < B_r) // tx-first to reduce warp divergence
                m[pb_vec + B_r*I + tx] = _m_i_new_[tx];
            if (ty == 1 && tx < B_r)
                l[pb_vec + B_r*I + tx] = _l_i_new_[tx];
            __syncthreads();
        }
    }
    
}

torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);	// Batch size
    const int nh = Q.size(1);	// Number of heads
    const int N = Q.size(2);	// Sequence size
    const int d = Q.size(3);	// Embedding size

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    O = O.to(device);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d\n", max_sram_size);

    int B_c = 32;
    int B_r = 32;
    int T_c = ceil(N, B_c);
    int T_r = ceil(N, B_r);

    int SharedMemSize = sizeof(float) * (2*(B_r*d) + 2*(B_c*d) + (B_r*B_c) + 6*(B_r));
    printf("Required shared memory: %d\n", SharedMemSize);

    if (SharedMemSize > max_sram_size) {
        printf("Unsupported Embed size(d)\n");
        return O;
    }
        

    // Launch FlashAttention GPU Kernel
    dim3 GridDim(1, 1, B*nh);
    dim3 BlockDim(B_c, B_r, 1);
    flash_attention<<<GridDim, BlockDim, SharedMemSize>>> (
        (float*) Q.data_ptr(), (float*) K.data_ptr(), (float*) V.data_ptr(), 
        (float*) O.data_ptr(), (float*) m.data_ptr(), (float*) l.data_ptr(),
        N, d, B_r, B_c, T_r, T_c
    );

    // Return output
    return O;
}
