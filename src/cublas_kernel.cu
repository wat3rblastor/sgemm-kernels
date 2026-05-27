#include "kernel_registry.cuh"

namespace {

class CublasKernel final : public SgemmKernel {
 public:
  int id() const override { return 0; }
  const char* name() const override { return "cuBLAS"; }

  void launch(const SgemmParams& params) const override {
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
};

}  // namespace

REGISTER_SGEMM_KERNEL(CublasKernel)
