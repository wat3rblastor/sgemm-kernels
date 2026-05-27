#include "kernel_registry.cuh"

namespace {

void launch_cublas(const SgemmParams& params) {
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

}  // namespace

REGISTER_SGEMM_KERNEL(0, "cuBLAS", launch_cublas)
