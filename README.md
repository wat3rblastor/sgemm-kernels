# matmul-kernels

This project builds a CUDA SGEMM benchmark with self-registering kernels:

- `0`: cuBLAS reference
- `1`: custom CUDA kernel
- `2`: custom CUDA kernel

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed and `nvcc` available in `PATH`
- CMake 3.18 or newer

For an NVIDIA A10 GPU, use CUDA architecture `86`.

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

This produces the executable at `./build/gemm`.

## Benchmarking

For fair benchmarking, pin the GPU graphics clocks so the GPU does not
dynamically change clock rate between kernel runs:

```bash
sudo nvidia-smi -lgc 1650,1650
```

## Add a Kernel

Kernels are self-registering. To add a new implementation, you only need to add
one `.cu` file under `./kernels/`. Do not edit `src/gemm.cu`, `src/utils.cu`,
or `CMakeLists.txt`.

Each kernel file should:

- include `kernel_registry.cuh`
- define a CUDA `__global__` kernel
- define a host launcher with the signature `void(const SgemmParams&)`
- register that launcher with `REGISTER_SGEMM_KERNEL`

Example `kernels/kernel3.cu`:

```cpp
#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

__global__ void sgemm_v3(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc += A[row * k + i] * B[i * n + col];
  }
  C[row * n + col] = alpha * acc + beta * C[row * n + col];
}

void launch_sgemm_v3(const SgemmParams& params) {
  dim3 block_dim(32, 32);
  dim3 grid_dim((params.n + 31) / 32, (params.m + 31) / 32);
  sgemm_v3<<<grid_dim, block_dim>>>(
      params.m,
      params.n,
      params.k,
      params.alpha,
      params.A,
      params.B,
      params.beta,
      params.C);
}

}  // namespace

REGISTER_SGEMM_KERNEL(3, "my-new-kernel", launch_sgemm_v3)
```

The `SgemmParams` fields are:

- `m`, `n`, `k`: matrix dimensions for `C = alpha * A * B + beta * C`
- `A`: row-major `m x k` input matrix
- `B`: row-major `k x n` input matrix
- `C`: row-major `m x n` output matrix
- `alpha`, `beta`: SGEMM scale factors
- `handle`: cuBLAS handle, available if your implementation needs cuBLAS helpers

Choose a unique integer ID. Kernel `0` is reserved for the cuBLAS reference.
`run.sh` and `./build/gemm --list` use the registered IDs, so there
is no separate launcher table to update.

After adding the file, rebuild:

```bash
cmake --build build -j
```

To see available kernels:

```bash
./build/gemm --list
```

To run only your kernel:

```bash
./build/gemm 3
```

To run all registered kernels:

```bash
bash run.sh
```

## Run With `run.sh`

After building, run:

```bash
bash run.sh
```

The script will:

- create `./test/` if it does not exist
- run every kernel returned by `./build/gemm --list-ids`
- write outputs to:
  - `./test/test_kernel_0.txt`
  - `./test/test_kernel_<id>.txt`

## Plot Results

After `run.sh` finishes, generate a comparison plot with:

```bash
python3 plot.py 0 1
```

This reads:

- `./test/test_kernel_0.txt`
- `./test/test_kernel_1.txt`

and writes:

- `./images/kernel_cublas_vs_1.png`

## Run Manually

You can also run the benchmark directly:

```bash
./build/gemm 0
./build/gemm 1
```

## Clean Rebuild

If you want to rebuild from scratch:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```
