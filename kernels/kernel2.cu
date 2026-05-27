#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int kTileWidth = 32;

__global__ void sgemm_v2(int m, int n, int k, float alpha, float* A, float* B,
                         float beta, float* C) {
  __shared__ float Ads[kTileWidth][kTileWidth];
  __shared__ float Bds[kTileWidth][kTileWidth];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * kTileWidth + ty;
  int col = bx * kTileWidth + tx;

  float Pvalue = 0.0f;
  for (int ph = 0; ph < (k + kTileWidth - 1) / kTileWidth; ++ph) {
    if (row < m && ph * kTileWidth + tx < k) {
      Ads[ty][tx] = A[row * k + ph * kTileWidth + tx];
    } else {
      Ads[ty][tx] = 0.0f;
    }

    if (ty + ph * kTileWidth < k && col < n) {
      Bds[ty][tx] = B[n * (ty + ph * kTileWidth) + col];
    } else {
      Bds[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < kTileWidth; ++i) {
      Pvalue += Ads[ty][i] * Bds[i][tx];
    }

    __syncthreads();
  }
  if (row < m && col < n) {
    C[row * n + col] = alpha * Pvalue + beta * C[row * n + col];
  }
}

void launch_sgemm_v2(const SgemmParams& params) {
  dim3 block_dim(kTileWidth, kTileWidth);
  dim3 grid_dim((params.n + kTileWidth - 1) / kTileWidth,
                (params.m + kTileWidth - 1) / kTileWidth);
  sgemm_v2<<<grid_dim, block_dim>>>(params.m, params.n, params.k, params.alpha,
                                    params.A, params.B, params.beta, params.C);
}

}  // namespace

REGISTER_SGEMM_KERNEL(2, "shared-mem", launch_sgemm_v2)
