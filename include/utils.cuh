#pragma once

#include <array>

#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int kWarmupCount = 3;
constexpr int kRepeatTimes = 10;
constexpr int kSizeCount = 16;
constexpr int kSizeStep = 256;

void check_cuda(cudaError_t error, const char* message);
void check_cublas(cublasStatus_t status, const char* message);

void randomize_matrix(float* matrix, int n);
void copy_matrix(float* src, float* dst, int n);
bool verify_matrix(float* lhs, float* rhs, int n);

std::array<int, kSizeCount> make_sizes();
int parse_kernel_num(const char* arg);
