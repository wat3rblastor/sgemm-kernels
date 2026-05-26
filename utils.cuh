#pragma once 

#include <array>

#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int kFirstKernel = 0;
constexpr int kLastKernel = 7;
constexpr int kWarmupCount = 3;
constexpr int kRepeatTimes = 10;
constexpr int kSizeCount = 16;
constexpr int kSizeStep = 256;

/*
CUDA Operations
*/
void check_cuda(cudaError_t, const char*);
void check_cublas(cublasStatus_t, const char*);

/*
Matrix Operations
*/
void randomize_matrix(float*, int);
void copy_matrix(float*, float*, int);
bool verify_matrix(float*, float*, int);

/*
Kernel Operations
*/
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

void launch_kernel(int, const SgemmParams&)

/*
Misc. Operations
*/
std::array<int, kSizeCount> make_sizes();
int parse_kernel_num(const char*);