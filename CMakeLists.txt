cmake_minimum_required(VERSION 3.24)

project(matmul_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(MATMUL_BENCH_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

add_executable(matmul_bench
  src/main.cu
  src/kernels.cu
)

target_include_directories(matmul_bench PRIVATE include)

target_compile_options(matmul_bench
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(MATMUL_BENCH_WARNINGS_AS_ERRORS)
  target_compile_options(matmul_bench
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
  )
endif()

set_target_properties(matmul_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY Shared
)

