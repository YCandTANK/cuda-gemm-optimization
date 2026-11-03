#include "fp8_gemm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (false)

constexpr float kAbsoluteTolerance = 1.0e-4f;
constexpr float kRelativeTolerance = 1.0e-4f;
constexpr int kGuardElements = 64;
constexpr float kGuardValue = 1234567.0f;

enum class Pattern {
    Zero,
    SparseIdentity,
    Distinct,
    SignedMixed,
    RandomSigned,
};

struct TestCase {
    int M;
    int N;
    int K;
    Pattern pattern;
    std::uint32_t seed;
    const char* name;
};

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t max_index = 0;
    std::size_t out_of_tolerance = 0;
    std::size_t nonfinite = 0;
};

const char* pattern_name(Pattern pattern)
{
    switch (pattern) {
        case Pattern::Zero: return "zero";
        case Pattern::SparseIdentity: return "sparse_identity";
        case Pattern::Distinct: return "distinct_values";
        case Pattern::SignedMixed: return "signed_mixed";
        case Pattern::RandomSigned: return "fixed_seed_random_signed";
    }
    return "unknown";
}

void fill_inputs(
    const TestCase& test,
    std::vector<float>& a,
    std::vector<float>& b_column_major)
{
    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-0.875f, 0.875f);

    for (int m = 0; m < test.M; ++m) {
        for (int k = 0; k < test.K; ++k) {
            float value = 0.0f;
            switch (test.pattern) {
                case Pattern::Zero:
                    break;
                case Pattern::SparseIdentity:
                    value = (k == (m * 7) % test.K)
                        ? ((m & 1) ? -1.0f : 1.0f) : 0.0f;
                    break;
                case Pattern::Distinct:
                    value = static_cast<float>(((m * 13 + k * 3) % 29) - 14)
                        / 16.0f;
                    break;
                case Pattern::SignedMixed:
                    value = static_cast<float>(((m * 5 + k * 11) % 23) - 11)
                        / 16.0f;
                    if ((m + k) % 17 == 0) value = 0.0f;
                    break;
                case Pattern::RandomSigned:
                    value = distribution(generator);
                    if ((m * test.K + k) % 97 == 0) value = 0.0f;
                    break;
            }
            a[static_cast<std::size_t>(m) * test.K + k] = value;
        }
    }

    for (int n = 0; n < test.N; ++n) {
        for (int k = 0; k < test.K; ++k) {
            float value = 0.0f;
            switch (test.pattern) {
                case Pattern::Zero:
                    break;
                case Pattern::SparseIdentity:
                    value = (k == (n * 11) % test.K)
                        ? ((n & 1) ? 0.5f : -0.75f) : 0.0f;
                    break;
                case Pattern::Distinct:
                    value = static_cast<float>(((n * 7 + k * 5) % 31) - 15)
                        / 16.0f;
                    break;
                case Pattern::SignedMixed:
                    value = static_cast<float>(((n * 9 + k * 7) % 27) - 13)
                        / 16.0f;
                    if ((n + k) % 19 == 0) value = 0.0f;
                    break;
                case Pattern::RandomSigned:
                    value = distribution(generator);
                    if ((n * test.K + k) % 89 == 0) value = 0.0f;
                    break;
            }
            b_column_major[static_cast<std::size_t>(n) * test.K + k] = value;
        }
    }
}

void quantize_e4m3(
    const std::vector<float>& input,
    std::vector<__nv_fp8_e4m3>& fp8,
    std::vector<float>& quantized)
{
    for (std::size_t i = 0; i < input.size(); ++i) {
        fp8[i] = __nv_fp8_e4m3(input[i]);
        quantized[i] = static_cast<float>(fp8[i]);
    }
}

void reference_gemm(
    const std::vector<float>& a,
    const std::vector<float>& b_column_major,
    std::vector<float>& c,
    int M,
    int N,
    int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum = std::fma(
                    a[static_cast<std::size_t>(m) * K + k],
                    b_column_major[static_cast<std::size_t>(n) * K + k],
                    sum);
            }
            c[static_cast<std::size_t>(m) * N + n] = sum;
        }
    }
}

ErrorStats compare(
    const std::vector<float>& reference,
    const std::vector<float>& actual)
{
    ErrorStats stats;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
            ++stats.nonfinite;
            ++stats.out_of_tolerance;
            continue;
        }
        const double abs_error = std::abs(
            static_cast<double>(actual[i]) - reference[i]);
        const double rel_error = abs_error /
            std::max(std::abs(static_cast<double>(reference[i])), 1.0e-7);
        if (abs_error > stats.max_abs) {
            stats.max_abs = abs_error;
            stats.max_index = i;
        }
        stats.max_rel = std::max(stats.max_rel, rel_error);
        const double allowed = kAbsoluteTolerance
            + kRelativeTolerance * std::abs(static_cast<double>(reference[i]));
        if (abs_error > allowed) ++stats.out_of_tolerance;
    }
    return stats;
}

void print_failure_location(
    const TestCase& test,
    const ErrorStats& stats,
    const std::vector<float>& reference,
    const std::vector<float>& actual)
{
    const int row = static_cast<int>(stats.max_index / test.N);
    const int col = static_cast<int>(stats.max_index % test.N);
    const int local_row = row % 16;
    const int local_col = col % 8;
    const int group = local_row % 8;
    const int tid = local_col / 2;
    const int lane = group * 4 + tid;
    const int accumulator = (local_row >= 8 ? 2 : 0) + local_col % 2;
    std::cout << "  worst C(" << row << ',' << col << ") expected="
              << reference[stats.max_index] << " actual="
              << actual[stats.max_index] << " -> lane=" << lane
              << " acc[" << accumulator << "] k_tiles=" << test.K / 32
              << '\n';
}

bool run_case(const TestCase& test)
{
    const std::size_t a_count = static_cast<std::size_t>(test.M) * test.K;
    const std::size_t b_count = static_cast<std::size_t>(test.N) * test.K;
    const std::size_t c_count = static_cast<std::size_t>(test.M) * test.N;

    std::vector<float> a_source(a_count);
    std::vector<float> b_source(b_count);
    fill_inputs(test, a_source, b_source);

    std::vector<__nv_fp8_e4m3> a_fp8(a_count);
    std::vector<__nv_fp8_e4m3> b_fp8(b_count);
    std::vector<float> a_quantized(a_count);
    std::vector<float> b_quantized(b_count);
    quantize_e4m3(a_source, a_fp8, a_quantized);
    quantize_e4m3(b_source, b_fp8, b_quantized);

    std::vector<float> reference(c_count);
    reference_gemm(
        a_quantized, b_quantized, reference, test.M, test.N, test.K);

    __nv_fp8_e4m3* d_a = nullptr;
    __nv_fp8_e4m3* d_b = nullptr;
    float* d_c_all = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, a_count * sizeof(*d_a)));
    CHECK_CUDA(cudaMalloc(&d_b, b_count * sizeof(*d_b)));
    CHECK_CUDA(cudaMalloc(
        &d_c_all, (c_count + 2 * kGuardElements) * sizeof(*d_c_all)));
    CHECK_CUDA(cudaMemcpy(
        d_a, a_fp8.data(), a_count * sizeof(*d_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, b_fp8.data(), b_count * sizeof(*d_b), cudaMemcpyHostToDevice));

    std::vector<float> initialized(c_count + 2 * kGuardElements, kGuardValue);
    std::fill(
        initialized.begin() + kGuardElements,
        initialized.begin() + kGuardElements + c_count,
        std::numeric_limits<float>::quiet_NaN());
    CHECK_CUDA(cudaMemcpy(
        d_c_all, initialized.data(), initialized.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    float* d_c = d_c_all + kGuardElements;
    const dim3 block(32);
    const dim3 grid(test.N / 8, test.M / 16);
    fp8_gemm_v1<<<grid, block>>>(
        d_a, d_b, d_c, test.M, test.N, test.K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> copied(initialized.size());
    CHECK_CUDA(cudaMemcpy(
        copied.data(), d_c_all, copied.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    std::vector<float> actual(
        copied.begin() + kGuardElements,
        copied.begin() + kGuardElements + c_count);

    bool guards_ok = true;
    for (int i = 0; i < kGuardElements; ++i) {
        guards_ok = guards_ok && copied[i] == kGuardValue;
        guards_ok = guards_ok
            && copied[kGuardElements + c_count + i] == kGuardValue;
    }

    const ErrorStats stats = compare(reference, actual);
    const bool passed = stats.out_of_tolerance == 0
        && stats.nonfinite == 0 && guards_ok;
    std::cout << std::left << std::setw(24) << test.name
              << " shape=" << test.M << 'x' << test.N << 'x' << test.K
              << " pattern=" << std::setw(24) << pattern_name(test.pattern)
              << std::scientific << std::setprecision(6)
              << " max_abs=" << stats.max_abs
              << " max_rel=" << stats.max_rel
              << " over_tol=" << stats.out_of_tolerance
              << " nonfinite=" << stats.nonfinite
              << " guards=" << (guards_ok ? "OK" : "CORRUPT")
              << " result=" << (passed ? "PASS" : "FAIL") << '\n';
    if (!passed && stats.max_index < reference.size()) {
        print_failure_location(test, stats, reference, actual);
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_all));
    return passed;
}

bool run_correctness()
{
    const std::vector<TestCase> tests = {
        {16, 8, 32, Pattern::Zero, 1001u, "zero_single_tile"},
        {16, 8, 32, Pattern::SparseIdentity, 1002u, "sparse_single_tile"},
        {16, 8, 32, Pattern::Distinct, 1003u, "distinct_single_tile"},
        {16, 8, 96, Pattern::SignedMixed, 2001u, "mixed_three_k_tiles"},
        {32, 16, 64, Pattern::RandomSigned, 3001u, "random_multi_cta"},
        {64, 64, 128, Pattern::RandomSigned, 4001u, "random_64"},
        {128, 128, 256, Pattern::RandomSigned, 5001u, "random_128"},
        {256, 256, 512, Pattern::RandomSigned, 6001u, "random_256"},
    };

    std::cout << "Reference: actual E4M3 values widened to FP32; sequential "
                 "FP32 std::fma accumulation\n"
              << "Tolerance: abs_error <= " << kAbsoluteTolerance << " + "
              << kRelativeTolerance << " * abs(reference)\n";
    bool all_passed = true;
    for (const TestCase& test : tests) {
        all_passed = run_case(test) && all_passed;
    }
    std::cout << "CORRECTNESS_RESULT=" << (all_passed ? "PASS" : "FAIL")
              << '\n';
    return all_passed;
}

bool benchmark_size(int size, int warmups, int iterations)
{
    const std::size_t matrix_elements = static_cast<std::size_t>(size) * size;
    std::vector<__nv_fp8_e4m3> a(matrix_elements);
    std::vector<__nv_fp8_e4m3> b(matrix_elements);
    std::mt19937 generator(7001u + static_cast<std::uint32_t>(size));
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (std::size_t i = 0; i < matrix_elements; ++i) {
        a[i] = __nv_fp8_e4m3(distribution(generator));
        b[i] = __nv_fp8_e4m3(distribution(generator));
    }

    __nv_fp8_e4m3* d_a = nullptr;
    __nv_fp8_e4m3* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, matrix_elements * sizeof(*d_a)));
    CHECK_CUDA(cudaMalloc(&d_b, matrix_elements * sizeof(*d_b)));
    CHECK_CUDA(cudaMalloc(&d_c, matrix_elements * sizeof(*d_c)));
    CHECK_CUDA(cudaMemcpy(
        d_a, a.data(), matrix_elements * sizeof(*d_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, b.data(), matrix_elements * sizeof(*d_b), cudaMemcpyHostToDevice));

    const dim3 block(32);
    const dim3 grid(size / 8, size / 16);
    for (int i = 0; i < warmups; ++i) {
        fp8_gemm_v1<<<grid, block>>>(d_a, d_b, d_c, size, size, size);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        fp8_gemm_v1<<<grid, block>>>(d_a, d_b, d_c, size, size, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = elapsed_ms / iterations;
    const double operations = 2.0 * size * size * static_cast<double>(size);
    const double tflops = operations / (average_ms * 1.0e9);
    std::cout << "BENCH shape=" << size << '^' << 3
              << " warmups=" << warmups << " iterations=" << iterations
              << std::fixed << std::setprecision(6)
              << " latency_ms=" << average_ms
              << " tflops=" << tflops << '\n';

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return true;
}

bool run_benchmark()
{
    bool passed = true;
    passed = benchmark_size(512, 10, 100) && passed;
    passed = benchmark_size(1024, 10, 50) && passed;
    passed = benchmark_size(2048, 5, 20) && passed;
    return passed;
}

}  // namespace

int main(int argc, char** argv)
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA device found\n";
        return EXIT_FAILURE;
    }
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    std::cout << "GPU=" << properties.name << " compute_capability="
              << properties.major << '.' << properties.minor
              << " global_memory_bytes=" << properties.totalGlobalMem << '\n';
    if (properties.major != 12 || properties.minor != 0) {
        std::cerr << "This validation target expects SM120\n";
        return EXIT_FAILURE;
    }

    const std::string mode = argc > 1 ? argv[1] : "--correctness";
    if (mode == "--correctness") {
        return run_correctness() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (mode == "--benchmark") {
        return run_benchmark() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    std::cerr << "Usage: " << argv[0]
              << " [--correctness|--benchmark]\n";
    return EXIT_FAILURE;
}
