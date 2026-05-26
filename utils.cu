#include <utils.cuh>

#include <stdexcept>
#include <string>

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

std::array<int, kSizeCount> make_sizes() {
  std::array<int, kSizeCount> sizes{};
  for (int i = 0; i < kSizeCount; ++i) {
    sizes[i] = kSizeStep * (i + 1);
  }
}

int parse_kernel_num(const char* arg) {
  const int kernel_num = std::stoi(arg);
  if (kernel_num < kFirstKernel || kernel_num > kLastKernel) {
    throw std::invalid_argument("Please enter a valid kernel number");
  }
  return kernel_num;
}