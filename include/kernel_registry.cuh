#pragma once

#include <vector>

#include "sgemm_params.cuh"

class SgemmKernel {
 public:
  virtual ~SgemmKernel() = default;

  virtual int id() const = 0;
  virtual const char* name() const = 0;
  virtual void launch(const SgemmParams& params) const = 0;
};

bool register_sgemm_kernel(const SgemmKernel& kernel);
std::vector<const SgemmKernel*> registered_sgemm_kernels();
const SgemmKernel& get_sgemm_kernel(int id);
void launch_kernel(int id, const SgemmParams& params);

#define SGEMM_CONCAT_IMPL(a, b) a##b
#define SGEMM_CONCAT(a, b) SGEMM_CONCAT_IMPL(a, b)

#define REGISTER_SGEMM_KERNEL(KERNEL_TYPE)                         \
  namespace {                                                      \
  const KERNEL_TYPE SGEMM_CONCAT(kSgemmKernel, __LINE__);          \
  const bool SGEMM_CONCAT(kRegisteredSgemmKernel, __LINE__) =      \
      register_sgemm_kernel(SGEMM_CONCAT(kSgemmKernel, __LINE__)); \
  }
