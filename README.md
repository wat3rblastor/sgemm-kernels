# CUDA Matmul Benchmark Harness

This project is a minimal benchmarking pipeline for CUDA matrix multiplication kernels. It assumes:

- `float32` inputs and outputs
- row-major `A`, `B`, and `C`
- kernel contract `C = A x B`
- one local NVIDIA GPU

## What it does

- benchmarks any registered matmul kernel with a shared launch signature
- validates each kernel against a CPU reference
- runs warmup launches before timing to reduce cold-start noise
- times kernels with CUDA events
- prints a terminal summary
- writes CSV and JSON for later plotting

## Kernel interface

Every benchmarked kernel must expose this launch signature:

```cpp
void launch_your_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);
```

The matrices are laid out as:

- `A`: `m x k`
- `B`: `k x n`
- `C`: `m x n`

All three are row-major.

## Adding kernels

Add your kernels in [src/kernels.cu](/Users/bowencheng/Projects/college_third_year/mlsys/matmul-kernels/src/kernels.cu) and register them in `registered_matmul_kernels()`:

```cpp
{"naive", &launch_naive_matmul},
{"shared", &launch_shared_matmul},
{"warp_tiling", &launch_warp_tiling_matmul},
```

The benchmark harness will pick them up automatically.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

If you need to force an architecture:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Run

Default sizes:

```bash
./build/matmul_bench
```

Custom arbitrary sizes:

```bash
./build/matmul_bench \
  --size 512x768x256 \
  --size 1024x1024x1536 \
  --csv results.csv \
  --json results.json
```

Single kernel:

```bash
./build/matmul_bench --kernel naive --size 2048x1024x512
```

Custom warmup and timing counts:

```bash
./build/matmul_bench --warmup 3 --iters 20 --size 1024x1024x1024
```

## Output fields

Each CSV/JSON record includes:

- kernel name
- `m`, `n`, `k`
- correctness pass/fail
- max absolute error
- min, max, mean, median, and standard deviation in milliseconds
- mean and best-case GFLOP/s
- effective input/output GB/s based on one read of `A`, one read of `B`, and one write of `C`

## Notes

- timing measures kernel execution only, not host-device transfers
- correctness runs before timing
- the included `naive` kernel is just a reference baseline for the harness
