#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "utils.cu"

int main(int argc, char** argv) try {
  if (argc != 2) {
    throw std::invalid_argument("Usage: ./gemm <kernel_num>");
  }

  const int kernel_num = parse_kernel_num(argv[1]);
  std::cout << "Select kernel " << kernel_num << std::endl;

  const auto sizes = make_sizes();
  const int max_size = sizes.back();
  // Square matrices
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

    // Correctness Check
    // cuBLAS is assumed to be correct
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

    // Warmup to prevent cold start
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
              << ") FGLOPs/s. size: (" << m << ")." << std::endl;

    // Sync C with C_ref for next iteration
    // No longer testing with the same C/C_ref, but that's okay
    copy_matrix(host_c_ref.data(), host_c.data(), m * n);
  }

  // Note that the idiomatic C++ way is to wrap all these resources with RAII
  // However, I don't want to write all of the boilerplate
  // Notice that if an error is thrown, we'll have a memory leak, but luckily
  // CUDA cleans everything up once the process exits
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cublasDestroy(handle);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaFree(device_c_ref);

} catch (const std::exception& error) {
  std::cerr << error.what() << std::endl;
  return EXIT_FAILURE;
}