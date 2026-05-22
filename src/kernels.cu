#include "matmul_kernel.cuh"

#include <vector>

namespace {

__global__ void naive_matmul_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = acc;
}

void launch_naive_matmul(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int tile = 16;
  const dim3 block(tile, tile);
  const dim3 grid((n + tile - 1) / tile, (m + tile - 1) / tile);
  naive_matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

}  // namespace

const std::vector<MatmulKernelSpec>& registered_matmul_kernels() {
  static const std::vector<MatmulKernelSpec> kernels = {
      {"naive", &launch_naive_matmul},
      // Register future kernels here.
      // {"shared", &launch_shared_matmul},
      // {"warp_tiling", &launch_warp_tiling_matmul},
  };
  return kernels;
}

