#pragma once

#include <cuda_runtime.h>

#include <string_view>
#include <vector>

using MatmulKernelFn = void (*)(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

struct MatmulKernelSpec {
  std::string_view name;
  MatmulKernelFn launch;
};

const std::vector<MatmulKernelSpec>& registered_matmul_kernels();

