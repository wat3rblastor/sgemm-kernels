#pragma once

#include <vector>

#include "sgemm_params.cuh"

struct SgemmKernel {
  int id;
  const char* name;
  void (*launch)(const SgemmParams&);
};

bool register_sgemm_kernel(SgemmKernel kernel);
std::vector<SgemmKernel> registered_sgemm_kernels();
const SgemmKernel& get_sgemm_kernel(int id);
void launch_kernel(int id, const SgemmParams& params);

#define SGEMM_CONCAT_IMPL(a, b) a##b
#define SGEMM_CONCAT(a, b) SGEMM_CONCAT_IMPL(a, b)

#define REGISTER_SGEMM_KERNEL(ID, NAME, LAUNCH_FN)                 \
  namespace {                                                      \
  const bool SGEMM_CONCAT(kRegisteredSgemmKernel, __LINE__) =      \
      register_sgemm_kernel(SgemmKernel{ID, NAME, LAUNCH_FN});     \
  }
