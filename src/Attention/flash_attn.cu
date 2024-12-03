#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Implement flash_attention kernel below
__global__ void _flash_attention() {}

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
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d\n", max_sram_size);

    // ============= Call kernel ==================
    // _flash_attention<<>>();
    // =============================================

    // Return output
    return O;
}
