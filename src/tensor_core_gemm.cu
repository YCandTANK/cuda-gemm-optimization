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


constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 16;
constexpr int B_SHARED_STRIDE = 40;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;

static_assert(WARPS_M * WARPS_N == WARPS_PER_BLOCK,
              "The 2D warp layout must match the block warp count");
static_assert(BLOCK_M == WARPS_M * WMMA_M &&
              BLOCK_N == WARPS_N * WMMA_N && BLOCK_K == WMMA_K,
              "V1 block and warp tiles are inconsistent");
static_assert(B_SHARED_STRIDE >= BLOCK_N && B_SHARED_STRIDE % 8 == 0,
              "The padded half WMMA leading dimension must be >= BLOCK_N "
              "and a multiple of 8 elements");

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

__global__ void wmmaGemmSharedV1(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
    int tid = threadIdx.x;

    // Block 对应的 C Tile 左上角
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    // Warp mapping
    int warp_id = tid / 32;

    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;

    int warp_c_row = block_row + warp_row * WMMA_M;
    int warp_c_col = block_col + warp_col * WMMA_N;

    // Shared Tiles
    __shared__ __align__(32) half As[BLOCK_M][BLOCK_K];
    __shared__ __align__(32) half Bs[BLOCK_K][BLOCK_N];

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

    //k循环
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        //Cooperative Load A
        for(int idx = tid; idx < BLOCK_M * BLOCK_K; idx += blockDim.x){
            int row_As = idx / BLOCK_K;
            int col_As = idx % BLOCK_K;

            int row_A = block_row + row_As;
            int col_A = k0 + col_As;

            As[row_As][col_As] = A[row_A * K + col_A];
        }

        //Cooperative Load B
        for(int idx = tid; idx < BLOCK_K * BLOCK_N; idx += blockDim.x){
            int row_Bs = idx / BLOCK_N;
            int col_Bs = idx % BLOCK_N;

            int row_B = k0 + row_Bs;
            int col_B = block_col + col_Bs;

            Bs[row_Bs][col_Bs] = B[row_B * N + col_B];
        }

        //等待整个 Block 完成搬运
        __syncthreads();

        //Shared → WMMA fragments
        //    A pointer
        const half* A_frag_ptr = &As[warp_row * WMMA_M][0];

        //    B pointer
        const half* B_frag_ptr = &Bs[0][warp_col * WMMA_N];

        //    load_matrix_sync
        wmma::load_matrix_sync(
            a_frag,
            A_frag_ptr,
            BLOCK_K
        );

        wmma::load_matrix_sync(
            b_frag,
            B_frag_ptr,
            BLOCK_N
        );

        //    mma_sync
        wmma::mma_sync(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );

        //确保所有 Warp 已使用完当前 Shared Tile
        __syncthreads();
    }

//结果写回C
    // C pointer
    float* C_frag_ptr = C + warp_c_row * N + warp_c_col;

    wmma::store_matrix_sync(
        C_frag_ptr,
        c_frag,
        N,
        wmma::mem_row_major
    );
}

// V1.1 isolates one layout change from V1: B has an eight-half padding at the
// end of each shared-memory row. The logical tile and cooperative load remain
// BLOCK_K x BLOCK_N.
__global__ void wmmaGemmSharedV1_1(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
    int tid = threadIdx.x;

    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    int warp_id = tid / 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;

    int warp_c_row = block_row + warp_row * WMMA_M;
    int warp_c_col = block_col + warp_col * WMMA_N;

    __shared__ __align__(32) half As[BLOCK_M][BLOCK_K];
    __shared__ __align__(32) half Bs[BLOCK_K][B_SHARED_STRIDE];

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

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
            int row_As = idx / BLOCK_K;
            int col_As = idx % BLOCK_K;

            int row_A = block_row + row_As;
            int col_A = k0 + col_As;

            As[row_As][col_As] = A[row_A * K + col_A];
        }

        // Indexing is intentionally based on BLOCK_N, not the padded stride,
        // so exactly the same 16 x 32 logical B elements as V1 are loaded.
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += blockDim.x) {
            int row_Bs = idx / BLOCK_N;
            int col_Bs = idx % BLOCK_N;

            int row_B = k0 + row_Bs;
            int col_B = block_col + col_Bs;

            Bs[row_Bs][col_Bs] = B[row_B * N + col_B];
        }

        __syncthreads();

        const half* A_frag_ptr = &As[warp_row * WMMA_M][0];
        const half* B_frag_ptr = &Bs[0][warp_col * WMMA_N];

        wmma::load_matrix_sync(
            a_frag,
            A_frag_ptr,
            BLOCK_K
        );

        wmma::load_matrix_sync(
            b_frag,
            B_frag_ptr,
            B_SHARED_STRIDE
        );

        wmma::mma_sync(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );

        __syncthreads();
    }

    float* C_frag_ptr = C + warp_c_row * N + warp_c_col;

    wmma::store_matrix_sync(
        C_frag_ptr,
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

enum class KernelVersion { V0, V1, V1_1 };

struct BenchmarkResult {
    vector<float> latencies_ms;
    float average_ms = 0.0f;
    float median_ms = 0.0f;
    double average_tflops = 0.0;
    double median_tflops = 0.0;
};

struct CorrectnessResult {
    ErrorStats original_stats;
    ErrorStats quantized_stats;
    bool correct = false;
};

const char* kernelName(KernelVersion version)
{
    switch (version) {
        case KernelVersion::V0:
            return "V0 baseline";
        case KernelVersion::V1:
            return "V1 shared";
        case KernelVersion::V1_1:
            return "V1.1 B-padding-40";
    }
    return "unknown";
}

void launchKernel(
    KernelVersion version,
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M,
    int N,
    int K)
{
    const dim3 block(THREADS_PER_BLOCK);
    if (version == KernelVersion::V0) {
        const int total_tiles = (M / WMMA_M) * (N / WMMA_N);
        const dim3 grid((total_tiles + WARPS_PER_BLOCK - 1) /
                        WARPS_PER_BLOCK);
        wmmaGemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else if (version == KernelVersion::V1) {
        const dim3 grid(N / BLOCK_N, M / BLOCK_M);
        wmmaGemmSharedV1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else {
        const dim3 grid(N / BLOCK_N, M / BLOCK_M);
        wmmaGemmSharedV1_1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
}

CorrectnessResult checkKernel(
    KernelVersion version,
    const half* d_A,
    const half* d_B,
    float* d_C,
    vector<float>& h_C,
    const vector<float>& h_reference_original,
    const vector<float>& h_reference_quantized,
    int M,
    int N,
    int K,
    size_t C_bytes)
{
    constexpr float KERNEL_ABS_TOLERANCE = 1.0e-2f;
    constexpr float KERNEL_REL_TOLERANCE = 1.0e-2f;

    // A NaN fill makes a missing tile/store visible to the finite-value check.
    CHECK_CUDA(cudaMemset(d_C, 0xff, C_bytes));
    launchKernel(version, d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));

    CorrectnessResult result;
    // Reference 1 includes FP32 -> FP16 input quantization error and is
    // reported for diagnosis. Reference 2 is used for the pass/fail decision.
    result.original_stats = compareResults(
        h_reference_original, h_C,
        numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
    result.quantized_stats = compareResults(
        h_reference_quantized, h_C,
        KERNEL_ABS_TOLERANCE, KERNEL_REL_TOLERANCE);
    result.correct = result.quantized_stats.finite &&
                     result.quantized_stats.within_tolerance;
    return result;
}

BenchmarkResult benchmarkKernel(
    KernelVersion version,
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup_runs,
    int benchmark_runs,
    int measurement_rounds)
{
    BenchmarkResult result;
    result.latencies_ms.reserve(measurement_rounds);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int round = 0; round < measurement_rounds; ++round) {
        for (int i = 0; i < warmup_runs; ++i) {
            launchKernel(version, d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            launchKernel(version, d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        result.latencies_ms.push_back(total_ms / benchmark_runs);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    for (float latency : result.latencies_ms) {
        result.average_ms += latency;
    }
    result.average_ms /= result.latencies_ms.size();

    vector<float> sorted = result.latencies_ms;
    sort(sorted.begin(), sorted.end());
    result.median_ms = sorted[sorted.size() / 2];
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    result.average_tflops = flops / (result.average_ms / 1000.0) / 1.0e12;
    result.median_tflops = flops / (result.median_ms / 1000.0) / 1.0e12;
    return result;
}

void printCorrectness(
    KernelVersion version,
    const CorrectnessResult& result)
{
    cout << scientific << setprecision(6);
    cout << kernelName(version)
         << " Reference 1 (original FP32 inputs) max absolute error = "
         << result.original_stats.max_abs_error << endl;
    cout << kernelName(version)
         << " Reference 1 (original FP32 inputs) max relative error = "
         << result.original_stats.max_rel_error << endl;
    cout << kernelName(version)
         << " Reference 2 (FP16-quantized inputs) max absolute error = "
         << result.quantized_stats.max_abs_error << endl;
    cout << kernelName(version)
         << " Reference 2 (FP16-quantized inputs) max relative error = "
         << result.quantized_stats.max_rel_error << endl;
    cout << kernelName(version) << ": "
         << (result.correct ? "Correct!" : "Mismatch!") << endl;
}

void printBenchmark(
    KernelVersion version,
    const BenchmarkResult& result,
    int warmup_runs,
    int benchmark_runs)
{
    cout << fixed << setprecision(6);
    cout << kernelName(version) << " warmup runs per measurement = "
         << warmup_runs << ", benchmark runs per measurement = "
         << benchmark_runs << endl;
    cout << kernelName(version) << " independent average latencies (ms) =";
    for (float latency : result.latencies_ms) {
        cout << " " << latency;
    }
    cout << endl;
    // Keep the original average-latency output and add the robust median.
    cout << kernelName(version) << " average latency = "
         << result.average_ms << " ms" << endl;
    cout << kernelName(version) << " average TFLOPS = "
         << result.average_tflops << endl;
    cout << kernelName(version) << " median latency = "
         << result.median_ms << " ms" << endl;
    cout << kernelName(version) << " median TFLOPS = "
         << result.median_tflops << endl;
}

bool runCase(
    int M,
    int N,
    int K,
    bool run_benchmark,
    const char* case_name)
{
    const bool v1_supported = M % BLOCK_M == 0 && N % BLOCK_N == 0 &&
                              K % BLOCK_K == 0;
    const int warmup_runs = 10;
    const int benchmark_runs = 100;
    const int measurement_rounds = 5;

    const size_t A_elements = static_cast<size_t>(M) * K;
    const size_t B_elements = static_cast<size_t>(K) * N;
    const size_t C_elements = static_cast<size_t>(M) * N;
    vector<float> h_A_fp32(A_elements);
    vector<float> h_B_fp32(B_elements);
    for (size_t i = 0; i < A_elements; ++i) {
        h_A_fp32[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 10.0f;
    }
    for (size_t i = 0; i < B_elements; ++i) {
        h_B_fp32[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 10.0f;
    }

    // Conversion, both CPU references, and transfers are outside all timed
    // regions. The same allocations and inputs are reused by all kernels.
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
    vector<float> h_C(C_elements);
    cout << "\n[" << case_name << "] matrix size = "
         << M << " x " << N << " x " << K << endl;
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
    const dim3 v0_grid((total_tiles + WARPS_PER_BLOCK - 1) /
                       WARPS_PER_BLOCK);
    cout << "V0 baseline launch = (" << v0_grid.x << ", 1) blocks x "
         << THREADS_PER_BLOCK << " threads" << endl;
    CorrectnessResult v0_correctness = checkKernel(
        KernelVersion::V0, d_A, d_B, d_C, h_C,
        h_reference_original, h_reference_quantized, M, N, K, C_bytes);
    printCorrectness(KernelVersion::V0, v0_correctness);

    CorrectnessResult v1_correctness;
    CorrectnessResult v1_1_correctness;
    if (v1_supported) {
        cout << "V1 shared launch = (" << N / BLOCK_N << ", "
             << M / BLOCK_M << ") blocks x " << THREADS_PER_BLOCK
             << " threads (2 x 2 warps/block)" << endl;
        v1_correctness = checkKernel(
            KernelVersion::V1, d_A, d_B, d_C, h_C,
            h_reference_original, h_reference_quantized, M, N, K, C_bytes);
        printCorrectness(KernelVersion::V1, v1_correctness);
        cout << "V1.1 B-padding-40 launch = (" << N / BLOCK_N << ", "
             << M / BLOCK_M << ") blocks x " << THREADS_PER_BLOCK
             << " threads (2 x 2 warps/block)" << endl;
        v1_1_correctness = checkKernel(
            KernelVersion::V1_1, d_A, d_B, d_C, h_C,
            h_reference_original, h_reference_quantized, M, N, K, C_bytes);
        printCorrectness(KernelVersion::V1_1, v1_1_correctness);
    } else {
        cout << "V1 shared: skipped (requires M/N multiples of 32 and K "
                "a multiple of 16); V0 remains supported for multiples of 16."
             << endl;
    }

    if (run_benchmark) {
        BenchmarkResult v0_benchmark = benchmarkKernel(
            KernelVersion::V0, d_A, d_B, d_C, M, N, K,
            warmup_runs, benchmark_runs, measurement_rounds);
        printBenchmark(KernelVersion::V0, v0_benchmark,
                       warmup_runs, benchmark_runs);
        if (v1_supported) {
            BenchmarkResult v1_benchmark = benchmarkKernel(
                KernelVersion::V1, d_A, d_B, d_C, M, N, K,
                warmup_runs, benchmark_runs, measurement_rounds);
            printBenchmark(KernelVersion::V1, v1_benchmark,
                           warmup_runs, benchmark_runs);
            BenchmarkResult v1_1_benchmark = benchmarkKernel(
                KernelVersion::V1_1, d_A, d_B, d_C, M, N, K,
                warmup_runs, benchmark_runs, measurement_rounds);
            printBenchmark(KernelVersion::V1_1, v1_1_benchmark,
                           warmup_runs, benchmark_runs);
            cout << fixed << setprecision(4);
            cout << "V1/V0 speedup (median latency) = "
                 << v0_benchmark.median_ms / v1_benchmark.median_ms
                 << "x" << endl;
            cout << "V1.1/V1 speedup (median latency) = "
                 << v1_benchmark.median_ms / v1_1_benchmark.median_ms
                 << "x" << endl;
        }
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return v0_correctness.correct &&
           (!v1_supported ||
            (v1_correctness.correct && v1_1_correctness.correct));
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

    cudaDeviceProp device_prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
    if (device_prop.major < 7) {
        cerr << "WMMA requires compute capability 7.0 or newer; got "
             << device_prop.major << "." << device_prop.minor << endl;
        return EXIT_FAILURE;
    }
    cout << "GPU = " << device_prop.name << ", compute capability = "
         << device_prop.major << "." << device_prop.minor << endl;
    cout << "Reference 2 pass criterion: abs_error <= 1e-2 + 1e-2 * "
            "abs(reference)" << endl;

    bool correct = runCase(M, N, K, true, "benchmark");
    // This non-square grid catches swapped/missing entries in the 2 x 2 warp
    // mapping while keeping CPU-reference work small.
    if (M != 64 || N != 96 || K != 32) {
        correct = runCase(64, 96, 32, false, "small 2x2 warp-mapping test") &&
                  correct;
    }
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
