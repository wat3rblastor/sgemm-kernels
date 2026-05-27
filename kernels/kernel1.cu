#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

__global__ void sgemm_v1(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  float val = 0.0f;
  if (x < n && y < m) {
    for (int i = 0; i < k; ++i) {
      val += A[y * k + i] * B[n * i + x];
    }
    C[y * n + x] = alpha * val + beta * C[y * n + x];
  }
}

void launch_sgemm_v1(const SgemmParams& params) {
  dim3 block_dim(32, 32);
  dim3 grid_dim((params.n + 31) / 32, (params.m + 31) / 32);
  sgemm_v1<<<grid_dim, block_dim>>>(
      params.m, params.n, params.k, params.alpha, params.A, params.B, params.beta, params.C);
}

}  // namespace

REGISTER_SGEMM_KERNEL(1, "naive-matmul", launch_sgemm_v1)
