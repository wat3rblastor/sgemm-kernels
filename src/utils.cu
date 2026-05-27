#include "utils.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "kernel_registry.cuh"

void check_cuda(cudaError_t error, const char* message) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
  }
}

void check_cublas(cublasStatus_t status, const char* message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(message);
  }
}

void randomize_matrix(float* matrix, int n) {
  std::random_device seed;
  std::mt19937 generator(seed());
  std::uniform_int_distribution<int> whole_dist(0, 4);
  std::uniform_int_distribution<int> frac_dist(0, 4);
  std::bernoulli_distribution sign_dist(0.5);

  for (int i = 0; i < n; ++i) {
    float value = static_cast<float>(whole_dist(generator) + 0.01f * static_cast<float>(frac_dist(generator)));
    matrix[i] = sign_dist(generator) ? value : -value;
  }
}

void copy_matrix(float* src, float* dst, int n) {
  if (!src || !dst) {
    throw std::invalid_argument("copy_matrix requires non-null src and dst");
  }
  std::copy_n(src, n, dst);
}

bool verify_matrix(float* lhs, float* rhs, int n) {
  if (!lhs || !rhs) {
    throw std::invalid_argument("verify_matrix requires non-null lhs and rhs");
  }

  for (int i = 0; i < n; ++i) {
    const double diff = std::fabs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
    if (diff > 1e-2) {
      std::cerr << "error: " << lhs[i] << ',' << rhs[i] << ',' << i << '\n';
      return false;
    }
  }
  return true;
}

std::array<int, kSizeCount> make_sizes() {
  std::array<int, kSizeCount> sizes{};
  for (int i = 0; i < kSizeCount; ++i) {
    sizes[i] = kSizeStep * (i + 1);
  }
  return sizes;
}

int parse_kernel_num(const char* arg) {
  const int kernel_num = std::stoi(arg);
  (void)get_sgemm_kernel(kernel_num);
  return kernel_num;
}
