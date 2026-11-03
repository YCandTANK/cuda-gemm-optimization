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

// V2 has the same contract and launch configuration as V1. Its only kernel
// change is loading each four-byte A/B fragment with one ld.global.b32.
__global__ void fp8_gemm_v2(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M,
    int N,
    int K);

// V3 keeps V2's fragment layout and 32-bit global loads, but groups four
// warps in a 2 x 2 layout. Launch with block(128) and
// grid(N / 16, M / 32); M, N and K must be multiples of 32, 16 and 32.
__global__ void fp8_gemm_v3(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M,
    int N,
    int K);

// V4 keeps V3's 4-warp 32 x 16 CTA tile and stages one 32-element K tile
// through 1536 bytes of shared memory for A/B reuse. Launch with block(128)
// and grid(N / 16, M / 32); M, N and K must be multiples of 32, 16 and 32.
__global__ void fp8_gemm_v4(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M,
    int N,
    int K);
