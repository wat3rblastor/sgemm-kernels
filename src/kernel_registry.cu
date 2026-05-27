#include "kernel_registry.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<SgemmKernel>& mutable_sgemm_kernels() {
  static std::vector<SgemmKernel> kernels;
  return kernels;
}

bool register_sgemm_kernel(SgemmKernel kernel) {
  if (!kernel.launch) {
    throw std::invalid_argument("Cannot register an SGEMM kernel with a null launcher");
  }

  auto& kernels = mutable_sgemm_kernels();
  const auto duplicate = std::find_if(
      kernels.begin(), kernels.end(),
      [id = kernel.id](const SgemmKernel& existing) { return existing.id == id; });
  if (duplicate != kernels.end()) {
    throw std::invalid_argument("Duplicate SGEMM kernel id: " + std::to_string(kernel.id));
  }

  kernels.push_back(kernel);
  return true;
}

std::vector<SgemmKernel> registered_sgemm_kernels() {
  auto kernels = mutable_sgemm_kernels();
  std::sort(kernels.begin(), kernels.end(),
            [](const SgemmKernel& lhs, const SgemmKernel& rhs) { return lhs.id < rhs.id; });
  return kernels;
}

const SgemmKernel& get_sgemm_kernel(int id) {
  const auto& kernels = mutable_sgemm_kernels();
  const auto iter = std::find_if(
      kernels.begin(), kernels.end(),
      [id](const SgemmKernel& kernel) { return kernel.id == id; });
  if (iter == kernels.end()) {
    throw std::invalid_argument("Unknown SGEMM kernel id: " + std::to_string(id));
  }
  return *iter;
}

void launch_kernel(int id, const SgemmParams& params) {
  get_sgemm_kernel(id).launch(params);
}
