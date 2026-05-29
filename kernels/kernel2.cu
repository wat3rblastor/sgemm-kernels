#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int BLOCK_SIZE = 32;

__global__ void sgemm_v2(int m, int n, int k, float alpha, float* A, float* B,
                         float beta, float* C) {
  __shared__ float Ads[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bds[BLOCK_SIZE][BLOCK_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * BLOCK_SIZE + ty;
  int col = bx * BLOCK_SIZE + tx;

  float Pvalue = 0.0f;
  for (int ph = 0; ph < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++ph) {
    if (row < m && ph * BLOCK_SIZE + tx < k) {
      Ads[ty][tx] = A[row * k + ph * BLOCK_SIZE + tx];
    } else {
      Ads[ty][tx] = 0.0f;
    }

    if (ty + ph * BLOCK_SIZE < k && col < n) {
      Bds[ty][tx] = B[n * (ty + ph * BLOCK_SIZE) + col];
    } else {
      Bds[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; ++i) {
      Pvalue += Ads[ty][i] * Bds[i][tx];
    }

    __syncthreads();
  }
  if (row < m && col < n) {
    C[row * n + col] = alpha * Pvalue + beta * C[row * n + col];
  }
}

void launch_sgemm_v2(const SgemmParams& params) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((params.n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (params.m + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sgemm_v2<<<grid_dim, block_dim>>>(params.m, params.n, params.k, params.alpha,
                                    params.A, params.B, params.beta, params.C);
}

}  // namespace

REGISTER_SGEMM_KERNEL(2, "shared-mem", launch_sgemm_v2)
