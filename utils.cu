#include "utils.cuh"
#include "kernel.cuh"

#include <stdexcept>
#include <string>

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

/*
CUDA Operations
*/
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

/*
Matrix Operations
*/
void randomize_matrix(float* M, int n) {
  std::random_device seed;
  std::mt19937 generator(seed());
  std::uniform_int_distribution<int> whole_dist(0, 4);
  std::uniform_int_distribution<int> frac_dist(0, 4);
  std::bernoulli_distribution sign_dist(0.5);

  for (int i = 0; i < n; ++i) {
    float value = static_cast<float>(whole_dist(generator) + 0.01f * static_cast<float>(frac_dist(generator)));
    M[i] = sign_dist(generator) ? value : -value;
  }
}

void copy_matrix(float* src, float* dst, int n) {
  if (!src || !dst) {
    throw std::invalid_argument("copy_matrix requires non_null src and dst");
  }
  std::copy_n(src, n, dst);
}

bool verify_matrix(float* M1, float* M2, int n) {
  if (!M1 || !M2) {
    throw std::invalid_argument("copy_matrix requires non_null src and dst");
  }
  
  for (int i = 0; i < n; ++i) {
    const double diff = std::fabs(static_cast<double>(M1[i]) - static_cast<double>(M2[i]));
    if (diff > 1e-2) {
      std::cerr << "error: " << M1[i] << ',' << M2[i] << ',' << i << '\n';
      return false;
    }
  }
  return true;
}

/*
Kernel Operations
*/
static void launch_cublas(const SgemmParams &params) {
  cublasSgemm(
    params.handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    params.n,
    params.m,
    params.k,
    &params.alpha,
    params.B,
    params.n,
    params.A,
    params.k,
    &params.beta,
    params.C,
    params.n);
}

static void launch_sgemm_v1(const SgemmParams &params) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(params.n, 32), CEIL_DIV(params.m, 32));
    sgemm_v1<<<gridDim, blockDim>>>(params.m, params.n, params.k, params.alpha, params.A, params.B, params.beta, params.C);
}

void launch_kernel(int kernel_num, const SgemmParams& params) {
  switch (kernel_num) {
    case 0:
      launch_cublas(params);
      break;
    case 1:
      launch_sgemm_v1(params);
      break;
    default:
      break;
  }
}

/*
Misc. Operations
*/
std::array<int, kSizeCount> make_sizes() {
  std::array<int, kSizeCount> sizes{};
  for (int i = 0; i < kSizeCount; ++i) {
    sizes[i] = kSizeStep * (i + 1);
  }
  return sizes;
}

int parse_kernel_num(const char* arg) {
  const int kernel_num = std::stoi(arg);
  if (kernel_num < kFirstKernel || kernel_num > kLastKernel) {
    throw std::invalid_argument("Please enter a valid kernel number");
  }
  return kernel_num;
}