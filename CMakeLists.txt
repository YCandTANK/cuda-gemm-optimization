cmake_minimum_required(VERSION 3.22)

project(cuda_gemm_opt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)

function(configure_cuda_target target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v;--ptxas-options=-warn-spills>)
endfunction()

add_executable(gemm_fp32 src/gemm.cu)
target_link_libraries(gemm_fp32 PRIVATE CUDA::cublas)
configure_cuda_target(gemm_fp32)

add_executable(tensor_core_gemm src/tensor_core_gemm.cu)
target_link_libraries(tensor_core_gemm PRIVATE CUDA::cublas)
configure_cuda_target(tensor_core_gemm)

add_executable(v3_random_correctness tests/v3_random_correctness.cu)
target_link_libraries(v3_random_correctness PRIVATE CUDA::cublas)
configure_cuda_target(v3_random_correctness)

add_executable(fp8_gemm_v1_test
    tests/fp8_gemm_v1_test.cu
    src/fp8_gemm.cu)
target_include_directories(fp8_gemm_v1_test PRIVATE src)
target_link_libraries(fp8_gemm_v1_test PRIVATE CUDA::cudart)
configure_cuda_target(fp8_gemm_v1_test)

add_executable(fp8_gemm_v2_compare
    tests/fp8_gemm_v2_compare.cu
    src/fp8_gemm.cu)
target_include_directories(fp8_gemm_v2_compare PRIVATE src)
target_link_libraries(fp8_gemm_v2_compare PRIVATE CUDA::cudart)
configure_cuda_target(fp8_gemm_v2_compare)

add_executable(fp8_gemm_v3_compare
    tests/fp8_gemm_v3_compare.cu
    src/fp8_gemm.cu)
target_include_directories(fp8_gemm_v3_compare PRIVATE src)
target_link_libraries(fp8_gemm_v3_compare PRIVATE CUDA::cudart)
configure_cuda_target(fp8_gemm_v3_compare)

add_executable(fp8_gemm_v4_compare
    tests/fp8_gemm_v4_compare.cu
    src/fp8_gemm.cu)
target_include_directories(fp8_gemm_v4_compare PRIVATE src)
target_link_libraries(fp8_gemm_v4_compare PRIVATE CUDA::cudart)
configure_cuda_target(fp8_gemm_v4_compare)

add_executable(fp8_gemm_v5_compare
    tests/fp8_gemm_v5_compare.cu
    src/fp8_gemm.cu)
target_include_directories(fp8_gemm_v5_compare PRIVATE src)
target_link_libraries(fp8_gemm_v5_compare PRIVATE CUDA::cudart)
configure_cuda_target(fp8_gemm_v5_compare)
