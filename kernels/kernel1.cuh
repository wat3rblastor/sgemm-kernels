#pragma once

#include <cuda_runtime.h>

__global__ void sgemm_v1(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  float val = 0.0; 
  if (x < n && y < m) {
    for (int i = 0; i < k; ++i) {
      val += A[y * k + i] * B[n * i + x];
    }
    C[y * n + x] = alpha * val + beta * C[y * n + x];
  }
}

