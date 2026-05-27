cmake_minimum_required(VERSION 3.18)
project(matmul_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(ENABLE_FAST_MATH "Enable --use_fast_math for CUDA builds" ON)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
)

add_executable(gemm
  src/gemm.cu
  src/utils.cu
  src/kernel_registry.cu
  src/cublas_kernel.cu
  ${KERNEL_SOURCES}
)

target_include_directories(gemm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

if(ENABLE_FAST_MATH)
  target_compile_options(gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET gemm PROPERTY CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm PRIVATE CUDA::cublas)
