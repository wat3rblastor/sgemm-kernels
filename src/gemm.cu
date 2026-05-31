#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_registry.cuh"
#include "utils.cuh"

namespace {

void print_kernel_list(bool ids_only) {
  for (const auto& kernel : registered_sgemm_kernels()) {
    if (ids_only) {
      std::cout << kernel.id << '\n';
    } else {
      std::cout << kernel.id << ": " << kernel.name << '\n';
    }
  }
}

void benchmark_kernel(int kernel_num) {
  const auto& kernel = get_sgemm_kernel(kernel_num);
  std::cout << "Select kernel " << kernel.id << " (" << kernel.name << ")" << std::endl;

  const auto sizes = make_sizes();
  const int max_size = sizes.back();
  const std::size_t element_count = static_cast<std::size_t>(max_size) * max_size;

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::vector<float> host_a(element_count);
  std::vector<float> host_b(element_count);
  std::vector<float> host_c(element_count);
  std::vector<float> host_c_ref(element_count);

  randomize_matrix(host_a.data(), static_cast<int>(element_count));
  randomize_matrix(host_b.data(), static_cast<int>(element_count));
  randomize_matrix(host_c.data(), static_cast<int>(element_count));
  copy_matrix(host_c.data(), host_c_ref.data(), static_cast<int>(element_count));

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  float* device_c_ref = nullptr;

  cublasHandle_t handle = nullptr;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;

  check_cublas(cublasCreate(&handle), "Create cublas handle error");
  check_cuda(cudaEventCreate(&begin), "cudaEventCreate(begin) failed");
  check_cuda(cudaEventCreate(&end), "cudaEventCreate(end) failed");

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), sizeof(float) * element_count), "cudaMalloc(A) failed");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), sizeof(float) * element_count), "cudaMalloc(B) failed");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c), sizeof(float) * element_count), "cudaMalloc(C) failed");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c_ref), sizeof(float) * element_count), "cudaMalloc(C_ref) failed");

  check_cuda(cudaMemcpy(device_a, host_a.data(), sizeof(float) * element_count, cudaMemcpyHostToDevice), "cudaMemcpy(A) failed");
  check_cuda(cudaMemcpy(device_b, host_b.data(), sizeof(float) * element_count, cudaMemcpyHostToDevice), "cudaMemcpy(B) failed");
  check_cuda(cudaMemcpy(device_c, host_c.data(), sizeof(float) * element_count, cudaMemcpyHostToDevice), "cudaMemcpy(C) failed");
  check_cuda(cudaMemcpy(device_c_ref, host_c_ref.data(), sizeof(float) * element_count, cudaMemcpyHostToDevice), "cudaMemcpy(C_ref) failed");

  std::cout << "max_size=" << max_size << std::endl;

  for (int size : sizes) {
    const int m = size;
    const int n = size;
    const int k = size;
    SgemmParams params = {m, n, k, alpha, device_a, device_b, beta, device_c, handle};
    SgemmParams ref_params = {m, n, k, alpha, device_a, device_b, beta, device_c_ref, handle};

    std::cout << "m=n=k=" << size << std::endl;

    if (kernel_num != 0) {
      launch_kernel(0, ref_params);
      launch_kernel(kernel_num, params);
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after correctness check failed");

      check_cuda(cudaMemcpy(host_c.data(), device_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost), "cudaMemcpy(C result) failed");
      check_cuda(cudaMemcpy(host_c_ref.data(), device_c_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost), "cudaMemcpy(C_ref result) failed");

      if (!verify_matrix(host_c.data(), host_c_ref.data(), m * n)) {
        throw std::runtime_error("Failed correctness verification against NVIDIA cuBLAS");
      }
    }

    for (int warmup = 0; warmup < kWarmupCount; ++warmup) {
      launch_kernel(kernel_num, params);
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timing failed");
    check_cuda(cudaEventRecord(begin), "cudaEventRecord(begin) failed");

    for (int repeat = 0; repeat < kRepeatTimes; ++repeat) {
      launch_kernel(kernel_num, params);
    }

    check_cuda(cudaEventRecord(end), "cudaEventRecord(end) failed");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end) failed");

    float elapsed_time_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_time_ms, begin, end), "cudaEventElapsedTime failed");

    const float elapsed_time_s = elapsed_time_ms / 1000.0f;
    std::cout << "Average elapsed time: (" << elapsed_time_s / kRepeatTimes
              << ") seconds, performance : (" << 2.0f * 1.e-9f * kRepeatTimes * m * n * k / elapsed_time_s
              << ") GFLOPs/s. size: (" << m << ")." << std::endl;

    copy_matrix(host_c_ref.data(), host_c.data(), m * n);
  }

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cublasDestroy(handle);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaFree(device_c_ref);
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc == 2 && std::string(argv[1]) == "--list") {
    print_kernel_list(false);
    return EXIT_SUCCESS;
  }

  if (argc == 2 && std::string(argv[1]) == "--list-ids") {
    print_kernel_list(true);
    return EXIT_SUCCESS;
  }

  if (argc != 2) {
    throw std::invalid_argument("Usage: ./gemm <kernel_num> | --list | --list-ids");
  }

  benchmark_kernel(parse_kernel_num(argv[1]));
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::cerr << error.what() << std::endl;
  return EXIT_FAILURE;
}
