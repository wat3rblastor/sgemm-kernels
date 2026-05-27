#pragma once

#include <cuda_runtime.h>

__global__ void sgemm_v1(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  float val = 0.0; 
  if (row < m && col < n) {
    for (int i = 0; i < k; ++i) {
      val += A[row * k + i] * B[n * i + col];
    }
    C[row * n + col] = alpha * val + beta * C[row * n + col];
  }
}

