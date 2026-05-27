#include <cuda_runtime.h>

#include "kernel_registry.cuh"
#include "launch_utils.cuh"

namespace {

__global__ void sgemm_v2(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  float val = 0.0f;
  if (row < m && col < n) {
    for (int i = 0; i < k; ++i) {
      val += A[row * k + i] * B[n * i + col];
    }
    C[row * n + col] = alpha * val + beta * C[row * n + col];
  }
}

void launch_sgemm_v2(const SgemmParams& params) {
  dim3 block_dim(32, 32);
  dim3 grid_dim(ceil_div(params.n, 32), ceil_div(params.m, 32));
  sgemm_v2<<<grid_dim, block_dim>>>(
      params.m, params.n, params.k, params.alpha, params.A, params.B, params.beta, params.C);
}

}  // namespace

REGISTER_SGEMM_KERNEL(2, "naive-row-col", launch_sgemm_v2)
