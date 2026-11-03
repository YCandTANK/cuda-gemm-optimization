#pragma once

#include <cuda_fp8.h>

// V1 contract:
//   A: M x K, row-major
//   B: K x N, column-major
//   C: M x N, row-major
//   M % 16 == 0, N % 8 == 0, K % 32 == 0
// Launch with block(32) and grid(N / 8, M / 16).
__global__ void fp8_gemm_v1(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M,
    int N,
    int K);
