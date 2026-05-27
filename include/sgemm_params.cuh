#pragma once

#include <cublas_v2.h>

struct SgemmParams {
  int m;
  int n;
  int k;
  float alpha;
  float* A;
  float* B;
  float beta;
  float* C;
  cublasHandle_t handle;
};
