#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace std;
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = 128;

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void wmmaGemmBaseline(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
// Warp mapping
    //warp/lane id
    int lane_id = threadIdx.x % 32;
    int warp_id_in_block = threadIdx.x / 32;

    //global warp id
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    //C Tile 数量
    int num_tile_rows = M / WMMA_M;
    int num_tile_cols = N / WMMA_N;

    int total_tiles = num_tile_rows * num_tile_cols;

    //Warp 级越界保护
    if (global_warp_id >= total_tiles) {
        return;
    }

    //global warp id → C中所处的tile_row / tile_col位置
    int tile_row = global_warp_id / num_tile_cols;
    int tile_col = global_warp_id % num_tile_cols;

    //c_row / c_col
    int c_row = tile_row * WMMA_M; 
    int c_col = tile_col * WMMA_N;

//声明 fragments
    wmma::fragment<
        wmma::matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        half,
        wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        half,
        wmma::row_major
    > b_frag;

    wmma::fragment<
        wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float
    > c_frag;

    //accumulator 清零
    wmma::fill_fragment(c_frag, 0.0f);

//    K loop
    for (int k0 = 0; k0 < K; k0 += WMMA_K){
        //    A pointer
        const half* A_tile = A + c_row * K + k0;

        //    B pointer
        const half* B_tile = B + k0 * N + c_col;

        //    load_matrix_sync
        wmma::load_matrix_sync(
            a_frag,
            A_tile,
            K
        );

        wmma::load_matrix_sync(
            b_frag,
            B_tile,
            N
        );

        //    mma_sync
        wmma::mma_sync(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );
    }

//结果写回C
    // C pointer
    float* C_tile = C + c_row * N + c_col;
    
    wmma::store_matrix_sync(
        C_tile,
        c_frag,
        N,
        wmma::mem_row_major
    );
}

// CPU FP32 reference. The same routine is used for both original FP32 inputs
// and FP16-quantized inputs converted back to FP32.
void gemmCPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

struct ErrorStats {
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    bool finite = true;
    bool within_tolerance = true;
};

ErrorStats compareResults(
    const vector<float>& reference,
    const vector<float>& result,
    float abs_tolerance,
    float rel_tolerance)
{
    ErrorStats stats;
    constexpr float RELATIVE_ERROR_FLOOR = 1.0e-6f;

    for (size_t i = 0; i < reference.size(); ++i) {
        const float abs_error = fabsf(reference[i] - result[i]);
        const float denominator = max(fabsf(reference[i]), RELATIVE_ERROR_FLOOR);
        const float rel_error = abs_error / denominator;

        if (!isfinite(reference[i]) || !isfinite(result[i]) ||
            !isfinite(abs_error) || !isfinite(rel_error)) {
            stats.finite = false;
            stats.within_tolerance = false;
            stats.max_abs_error = numeric_limits<float>::infinity();
            stats.max_rel_error = numeric_limits<float>::infinity();
            continue;
        }

        stats.max_abs_error = max(stats.max_abs_error, abs_error);
        stats.max_rel_error = max(stats.max_rel_error, rel_error);

        // A mixed absolute/relative check avoids treating results near zero as
        // failures solely because their relative error is large.
        if (abs_error > abs_tolerance + rel_tolerance * fabsf(reference[i])) {
            stats.within_tolerance = false;
        }
    }

    return stats;
}

int main(int argc, char** argv)
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (argc != 1 && argc != 4) {
        cerr << "Usage: " << argv[0] << " [M N K]" << endl;
        return EXIT_FAILURE;
    }
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    if (M <= 0 || N <= 0 || K <= 0 ||
        M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        cerr << "M, N, and K must be positive multiples of 16; got "
             << M << " x " << N << " x " << K << endl;
        return EXIT_FAILURE;
    }

    const int warmup_runs = 10;
    const int benchmark_runs = 100;
    constexpr float KERNEL_ABS_TOLERANCE = 1.0e-2f;
    constexpr float KERNEL_REL_TOLERANCE = 1.0e-2f;

    cudaDeviceProp device_prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
    if (device_prop.major < 7) {
        cerr << "WMMA baseline requires a GPU with compute capability 7.0 or newer; got "
             << device_prop.major << "." << device_prop.minor << endl;
        return EXIT_FAILURE;
    }

    const size_t A_elements = static_cast<size_t>(M) * K;
    const size_t B_elements = static_cast<size_t>(K) * N;
    const size_t C_elements = static_cast<size_t>(M) * N;

    // Match src/gemm.cu: deterministic FP32 source data in small ranges.
    vector<float> h_A_fp32(A_elements);
    vector<float> h_B_fp32(B_elements);
    for (size_t i = 0; i < A_elements; ++i) {
        h_A_fp32[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 10.0f;
    }
    for (size_t i = 0; i < B_elements; ++i) {
        h_B_fp32[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 10.0f;
    }

    // Quantization is deliberately completed before any timed region.
    vector<half> h_A_fp16(A_elements);
    vector<half> h_B_fp16(B_elements);
    vector<float> h_A_quantized_fp32(A_elements);
    vector<float> h_B_quantized_fp32(B_elements);
    for (size_t i = 0; i < A_elements; ++i) {
        h_A_fp16[i] = __float2half(h_A_fp32[i]);
        h_A_quantized_fp32[i] = __half2float(h_A_fp16[i]);
    }
    for (size_t i = 0; i < B_elements; ++i) {
        h_B_fp16[i] = __float2half(h_B_fp32[i]);
        h_B_quantized_fp32[i] = __half2float(h_B_fp16[i]);
    }

    vector<float> h_reference_original(C_elements);
    vector<float> h_reference_quantized(C_elements);
    vector<float> h_wmma_C(C_elements);

    cout << "Computing FP32 references..." << endl;
    gemmCPU(h_A_fp32.data(), h_B_fp32.data(), h_reference_original.data(),
            M, N, K);
    gemmCPU(h_A_quantized_fp32.data(), h_B_quantized_fp32.data(),
            h_reference_quantized.data(), M, N, K);

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    const size_t A_bytes = A_elements * sizeof(half);
    const size_t B_bytes = B_elements * sizeof(half);
    const size_t C_bytes = C_elements * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, C_bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_fp16.data(), A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_fp16.data(), B_bytes, cudaMemcpyHostToDevice));

    const int total_tiles = (M / WMMA_M) * (N / WMMA_N);
    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid((total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    for (int i = 0; i < warmup_runs; ++i) {
        wmmaGemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        wmmaGemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    const float average_ms = total_ms / benchmark_runs;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = flops / (average_ms / 1000.0) / 1.0e12;

    CHECK_CUDA(cudaMemcpy(h_wmma_C.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));

    // Reference 1 measures the combined FP16 input quantization and WMMA error.
    const ErrorStats original_stats = compareResults(
        h_reference_original, h_wmma_C,
        numeric_limits<float>::infinity(), numeric_limits<float>::infinity());

    // Reference 2 isolates kernel arithmetic from input quantization. Its
    // threshold is intentionally looser than the FP32 kernel's 1e-5 check.
    const ErrorStats quantized_stats = compareResults(
        h_reference_quantized, h_wmma_C,
        KERNEL_ABS_TOLERANCE, KERNEL_REL_TOLERANCE);
    const bool correct = quantized_stats.finite && quantized_stats.within_tolerance;

    cout << fixed << setprecision(6);
    cout << "WMMA baseline matrix size = " << M << " x " << N << " x " << K << endl;
    cout << "WMMA baseline launch = " << grid.x << " blocks x "
         << block.x << " threads (" << WARPS_PER_BLOCK << " warps/block)" << endl;
    cout << "WMMA baseline warmup runs = " << warmup_runs
         << ", benchmark runs = " << benchmark_runs << endl;
    cout << "WMMA baseline average latency = " << average_ms << " ms" << endl;
    cout << "WMMA baseline TFLOPS = " << tflops << endl;

    cout << scientific << setprecision(6);
    cout << "Reference 1 (original FP32 inputs) max absolute error = "
         << original_stats.max_abs_error << endl;
    cout << "Reference 1 (original FP32 inputs) max relative error = "
         << original_stats.max_rel_error << endl;
    cout << "Reference 2 (FP16-quantized inputs) max absolute error = "
         << quantized_stats.max_abs_error << endl;
    cout << "Reference 2 (FP16-quantized inputs) max relative error = "
         << quantized_stats.max_rel_error << endl;
    cout << "Reference 2 mixed tolerance: abs_error <= "
         << KERNEL_ABS_TOLERANCE << " + " << KERNEL_REL_TOLERANCE
         << " * abs(reference)" << endl;
    cout << "WMMA baseline: " << (correct ? "Correct!" : "Mismatch!") << endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
