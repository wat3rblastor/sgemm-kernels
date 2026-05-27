# matmul-kernels

This project builds a CUDA SGEMM benchmark and runs two kernels:

- `0`: cuBLAS reference
- `1`: custom CUDA kernel

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

## Run With `run.sh`

After building, run:

```bash
bash run.sh
```

The script will:

- create `./test/` if it does not exist
- run `./build/gemm 0`
- run `./build/gemm 1`
- write outputs to:
  - `./test/test_kernel_0.txt`
  - `./test/test_kernel_1.txt`

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
