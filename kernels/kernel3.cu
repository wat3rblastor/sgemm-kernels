#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {


template<
  const int BLOCK_SIZE,
  const int TM,
  const int TN
>
__global__ void sgemm_v3(int m, int n, int k, float alpha, float* A, float* B,
                         float beta, float* C) {
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  float acc[TM][TN] = {0.0f};

  for (int ph = 0; ph < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++ph) {
    
    // Load shared memory
    for (int tr = ty; tr < BLOCK_SIZE; tr += blockDim.y) {
      for (int tc = tx; tc < BLOCK_SIZE; tc += blockDim.x) {
        int Arow = by * BLOCK_SIZE + tr;

        if (Arow < m && ph * BLOCK_SIZE + tc < k) {
          As[tr][tc] = A[Arow * k + ph * BLOCK_SIZE + tc];
        } else {
          As[tr][tc] = 0.0f;
        }

        int Bcol = bx * BLOCK_SIZE + tc;

        if (ph * BLOCK_SIZE + tr < k && Bcol < n) {
          Bs[tr][tc] = B[(ph * BLOCK_SIZE + tr) * n + Bcol];
        } else {
          Bs[tr][tc] = 0.0f;
        }

      }
    }

    __syncthreads();

    for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
      float regA[TM];
      float regB[TN];

      for (int i = 0; i < TM; ++i) {
        regA[i] = As[ty * TM + i][kk];
      }

      for (int j = 0; j < TN; ++j) {
        regB[j] = Bs[kk][tx * TN + j];
      }

      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += regA[i] * regB[j];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int row = by * BLOCK_SIZE + ty * TM + i;
      int col = bx * BLOCK_SIZE + tx * TN + j;

      if (row < m && col < n) {
        C[row * n + col] = alpha * acc[i][j] + beta * C[row * n + col];
      }
    }
  }
}

void launch_sgemm_v3(const SgemmParams& params) {
  constexpr int BLOCK_SIZE = 32;

  // TM has to divide into BLOCK_SIZE evenly
  // Similarly for TN
  constexpr int TM = 4;
  constexpr int TN = 4;

  dim3 block_dim(BLOCK_SIZE / TN, BLOCK_SIZE / TM);
  dim3 grid_dim((params.n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (params.m + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sgemm_v3<BLOCK_SIZE, TM, TN>
    <<<grid_dim, block_dim>>>(params.m, params.n, params.k, params.alpha,
                              params.A, params.B, params.beta, params.C);
}

}  // namespace

REGISTER_SGEMM_KERNEL(3, "reg-tiling", launch_sgemm_v3)
