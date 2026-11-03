#include "fp8_gemm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
constexpr int kBenchmarkRounds = 7;

enum class KernelVersion { V1, V2, V3 };

enum class Pattern {
    Zero,
    Sparse,
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

struct KernelOutput {
    std::vector<float> values;
    bool guards_ok = false;
};

const char* version_name(KernelVersion version)
{
    switch (version) {
        case KernelVersion::V1: return "V1";
        case KernelVersion::V2: return "V2";
        case KernelVersion::V3: return "V3";
    }
    return "unknown";
}

const char* pattern_name(Pattern pattern)
{
    switch (pattern) {
        case Pattern::Zero: return "zero";
        case Pattern::Sparse: return "sparse";
        case Pattern::Distinct: return "distinct_nonuniform";
        case Pattern::SignedMixed: return "signed_mixed";
        case Pattern::RandomSigned: return "fixed_seed_random_signed";
    }
    return "unknown";
}

void launch_kernel(
    KernelVersion version,
    const __nv_fp8_e4m3* a,
    const __nv_fp8_e4m3* b,
    float* c,
    int M,
    int N,
    int K)
{
    if (version == KernelVersion::V3) {
        fp8_gemm_v3<<<dim3(N / 16, M / 32), dim3(128)>>>(
            a, b, c, M, N, K);
    } else if (version == KernelVersion::V2) {
        fp8_gemm_v2<<<dim3(N / 8, M / 16), dim3(32)>>>(
            a, b, c, M, N, K);
    } else {
        fp8_gemm_v1<<<dim3(N / 8, M / 16), dim3(32)>>>(
            a, b, c, M, N, K);
    }
}

void fill_inputs(
    const TestCase& test,
    std::vector<float>& a,
    std::vector<float>& b_column_major)
{
    std::mt19937 generator(test.seed);
    std::uniform_real_distribution<float> distribution(-0.9375f, 0.9375f);

    for (int m = 0; m < test.M; ++m) {
        for (int k = 0; k < test.K; ++k) {
            float value = 0.0f;
            switch (test.pattern) {
                case Pattern::Zero:
                    break;
                case Pattern::Sparse:
                    if (k == (m * 7 + 3) % test.K) {
                        value = (m & 1) ? -1.0f : 0.75f;
                    } else if (k == (m * 13 + 5) % test.K) {
                        value = (m % 3 == 0) ? 0.5f : -0.25f;
                    }
                    break;
                case Pattern::Distinct:
                    value = static_cast<float>(
                        ((m * 17 + k * 5 + (m ^ k)) % 61) - 30) / 32.0f;
                    break;
                case Pattern::SignedMixed:
                    value = static_cast<float>(
                        ((m * 11 + k * 7 + 3) % 37) - 18) / 16.0f;
                    if ((m * 3 + k) % 23 == 0) value = 0.0f;
                    break;
                case Pattern::RandomSigned:
                    value = distribution(generator);
                    if ((static_cast<std::size_t>(m) * test.K + k) % 101 == 0) {
                        value = 0.0f;
                    }
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
                case Pattern::Sparse:
                    if (k == (n * 11 + 1) % test.K) {
                        value = (n & 1) ? 0.625f : -0.875f;
                    } else if (k == (n * 19 + 7) % test.K) {
                        value = (n % 3 == 0) ? -0.5f : 0.25f;
                    }
                    break;
                case Pattern::Distinct:
                    value = static_cast<float>(
                        ((n * 13 + k * 9 + (n ^ k)) % 59) - 29) / 32.0f;
                    break;
                case Pattern::SignedMixed:
                    value = static_cast<float>(
                        ((n * 5 + k * 13 + 1) % 41) - 20) / 16.0f;
                    if ((n + k * 2) % 29 == 0) value = 0.0f;
                    break;
                case Pattern::RandomSigned:
                    value = distribution(generator);
                    if ((static_cast<std::size_t>(n) * test.K + k) % 97 == 0) {
                        value = 0.0f;
                    }
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

KernelOutput run_output(
    KernelVersion version,
    const __nv_fp8_e4m3* d_a,
    const __nv_fp8_e4m3* d_b,
    float* d_c_all,
    std::size_t c_count,
    int M,
    int N,
    int K)
{
    std::vector<float> initialized(c_count + 2 * kGuardElements, kGuardValue);
    std::fill(
        initialized.begin() + kGuardElements,
        initialized.begin() + kGuardElements + c_count,
        std::numeric_limits<float>::quiet_NaN());
    CHECK_CUDA(cudaMemcpy(
        d_c_all, initialized.data(), initialized.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    launch_kernel(
        version, d_a, d_b, d_c_all + kGuardElements, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> copied(initialized.size());
    CHECK_CUDA(cudaMemcpy(
        copied.data(), d_c_all, copied.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    KernelOutput output;
    output.values.assign(
        copied.begin() + kGuardElements,
        copied.begin() + kGuardElements + c_count);
    output.guards_ok = true;
    for (int i = 0; i < kGuardElements; ++i) {
        output.guards_ok = output.guards_ok && copied[i] == kGuardValue;
        output.guards_ok = output.guards_ok
            && copied[kGuardElements + c_count + i] == kGuardValue;
    }
    return output;
}

void print_stats(const char* label, const ErrorStats& stats)
{
    std::cout << label
              << " max_abs=" << stats.max_abs
              << " max_rel=" << stats.max_rel
              << " over_tol=" << stats.out_of_tolerance
              << " nonfinite=" << stats.nonfinite;
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

    const KernelOutput v2 = run_output(
        KernelVersion::V2, d_a, d_b, d_c_all, c_count,
        test.M, test.N, test.K);
    const KernelOutput v3 = run_output(
        KernelVersion::V3, d_a, d_b, d_c_all, c_count,
        test.M, test.N, test.K);

    const ErrorStats v2_reference = compare(reference, v2.values);
    const ErrorStats v3_reference = compare(reference, v3.values);
    const ErrorStats v3_v2 = compare(v2.values, v3.values);
    const bool passed = v2.guards_ok && v3.guards_ok
        && v2_reference.out_of_tolerance == 0
        && v3_reference.out_of_tolerance == 0
        && v3_v2.out_of_tolerance == 0;

    std::cout << std::scientific << std::setprecision(6)
              << "CASE " << std::left << std::setw(25) << test.name
              << " shape=" << test.M << 'x' << test.N << 'x' << test.K
              << " pattern=" << pattern_name(test.pattern) << '\n' << "  ";
    print_stats("V2_vs_ref", v2_reference);
    std::cout << " guard=" << (v2.guards_ok ? "OK" : "CORRUPT") << '\n'
              << "  ";
    print_stats("V3_vs_ref", v3_reference);
    std::cout << " guard=" << (v3.guards_ok ? "OK" : "CORRUPT") << '\n'
              << "  ";
    print_stats("V3_vs_V2", v3_v2);
    std::cout << " result=" << (passed ? "PASS" : "FAIL") << '\n';

    if (!passed) {
        const std::size_t index = v3_reference.max_index;
        const int row = static_cast<int>(index / test.N);
        const int col = static_cast<int>(index % test.N);
        const int warp_m = (row % 32) / 16;
        const int warp_n = (col % 16) / 8;
        const int warp_id = warp_m * 2 + warp_n;
        const int lane = (row % 8) * 4 + (col % 8) / 2;
        const int accumulator = ((row % 16) >= 8 ? 2 : 0) + col % 2;
        std::cout << "  worst C(" << row << ',' << col << ") ref="
                  << reference[index] << " V2=" << v2.values[index]
                  << " V3=" << v3.values[index] << " warp=" << warp_id
                  << " lane=" << lane << " acc[" << accumulator << "]\n";
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_all));
    return passed;
}

bool run_correctness()
{
    const std::vector<TestCase> tests = {
        {32, 16, 32, Pattern::Zero, 8101u, "zero_single_cta"},
        {32, 16, 32, Pattern::Sparse, 8102u, "sparse_single_cta"},
        {32, 16, 32, Pattern::Distinct, 8103u, "distinct_single_cta"},
        {32, 16, 96, Pattern::SignedMixed, 8201u, "mixed_three_k_tiles"},
        {64, 32, 64, Pattern::RandomSigned, 8301u, "random_multi_cta_a"},
        {96, 48, 160, Pattern::Distinct, 8401u, "rectangular_nonuniform"},
        {64, 64, 128, Pattern::RandomSigned, 8501u, "random_multi_cta_b"},
        {128, 128, 256, Pattern::RandomSigned, 8601u, "random_128"},
        {256, 256, 512, Pattern::RandomSigned, 8701u, "random_256"},
    };

    std::cout << "V2 and V3 share the same quantized E4M3 inputs; tolerance: "
              << "abs_error <= " << kAbsoluteTolerance << " + "
              << kRelativeTolerance << " * abs(reference)\n";
    bool passed = true;
    for (const TestCase& test : tests) {
        passed = run_case(test) && passed;
    }
    std::cout << "CORRECTNESS_RESULT=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double time_kernel(
    KernelVersion version,
    const __nv_fp8_e4m3* d_a,
    const __nv_fp8_e4m3* d_b,
    float* d_c,
    int size,
    int warmups,
    int iterations)
{
    for (int i = 0; i < warmups; ++i) {
        launch_kernel(version, d_a, d_b, d_c, size, size, size);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_kernel(version, d_a, d_b, d_c, size, size, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsed_ms / iterations;
}

bool benchmark_size(int size, int warmups, int iterations)
{
    const std::size_t count = static_cast<std::size_t>(size) * size;
    std::vector<__nv_fp8_e4m3> a(count);
    std::vector<__nv_fp8_e4m3> b(count);
    std::mt19937 generator(9001u + static_cast<std::uint32_t>(size));
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (std::size_t i = 0; i < count; ++i) {
        a[i] = __nv_fp8_e4m3(distribution(generator));
        b[i] = __nv_fp8_e4m3(distribution(generator));
    }

    __nv_fp8_e4m3* d_a = nullptr;
    __nv_fp8_e4m3* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(*d_a)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(*d_b)));
    CHECK_CUDA(cudaMalloc(&d_c, count * sizeof(*d_c)));
    CHECK_CUDA(cudaMemcpy(
        d_a, a.data(), count * sizeof(*d_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, b.data(), count * sizeof(*d_b), cudaMemcpyHostToDevice));

    std::vector<double> v2_times;
    std::vector<double> v3_times;
    const double operations = 2.0 * size * size * static_cast<double>(size);
    for (int round = 0; round < kBenchmarkRounds; ++round) {
        double v2_ms = 0.0;
        double v3_ms = 0.0;
        if ((round & 1) == 0) {
            v2_ms = time_kernel(
                KernelVersion::V2, d_a, d_b, d_c,
                size, warmups, iterations);
            v3_ms = time_kernel(
                KernelVersion::V3, d_a, d_b, d_c,
                size, warmups, iterations);
        } else {
            v3_ms = time_kernel(
                KernelVersion::V3, d_a, d_b, d_c,
                size, warmups, iterations);
            v2_ms = time_kernel(
                KernelVersion::V2, d_a, d_b, d_c,
                size, warmups, iterations);
        }
        v2_times.push_back(v2_ms);
        v3_times.push_back(v3_ms);
        std::cout << "ROUND shape=" << size << '^' << 3
                  << " round=" << round + 1
                  << " order=" << ((round & 1) == 0 ? "V2,V3" : "V3,V2")
                  << std::fixed << std::setprecision(6)
                  << " V2_ms=" << v2_ms
                  << " V2_tflops=" << operations / (v2_ms * 1.0e9)
                  << " V3_ms=" << v3_ms
                  << " V3_tflops=" << operations / (v3_ms * 1.0e9)
                  << " speedup_V3_over_V2=" << v2_ms / v3_ms << '\n';
    }

    const double v2_median = median(v2_times);
    const double v3_median = median(v3_times);
    std::cout << "MEDIAN shape=" << size << '^' << 3
              << " rounds=" << kBenchmarkRounds
              << " warmups=" << warmups << " iterations=" << iterations
              << std::fixed << std::setprecision(6)
              << " V2_ms=" << v2_median
              << " V2_tflops=" << operations / (v2_median * 1.0e9)
              << " V3_ms=" << v3_median
              << " V3_tflops=" << operations / (v3_median * 1.0e9)
              << " speedup_V3_over_V2=" << v2_median / v3_median << '\n';

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

bool run_profile_workload(KernelVersion version)
{
    constexpr int size = 1024;
    constexpr int launches = 10;
    const std::size_t count = static_cast<std::size_t>(size) * size;
    std::vector<__nv_fp8_e4m3> a(count);
    std::vector<__nv_fp8_e4m3> b(count);
    std::mt19937 generator(9001u + size);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (std::size_t i = 0; i < count; ++i) {
        a[i] = __nv_fp8_e4m3(distribution(generator));
        b[i] = __nv_fp8_e4m3(distribution(generator));
    }

    __nv_fp8_e4m3* d_a = nullptr;
    __nv_fp8_e4m3* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(*d_a)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(*d_b)));
    CHECK_CUDA(cudaMalloc(&d_c, count * sizeof(*d_c)));
    CHECK_CUDA(cudaMemcpy(
        d_a, a.data(), count * sizeof(*d_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, b.data(), count * sizeof(*d_b), cudaMemcpyHostToDevice));
    for (int i = 0; i < launches; ++i) {
        launch_kernel(version, d_a, d_b, d_c, size, size, size);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "PROFILE_WORKLOAD version=" << version_name(version)
              << " shape=1024^3 launches=" << launches << '\n';
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return true;
}

bool run_v1_timing_audit()
{
    constexpr int size = 1024;
    constexpr int warmups = 10;
    constexpr int iterations = 50;
    constexpr int repeats = 5;
    const std::size_t count = static_cast<std::size_t>(size) * size;
    std::vector<__nv_fp8_e4m3> a(count, __nv_fp8_e4m3(0.25f));
    std::vector<__nv_fp8_e4m3> b(count, __nv_fp8_e4m3(-0.375f));
    __nv_fp8_e4m3* d_a = nullptr;
    __nv_fp8_e4m3* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(*d_a)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(*d_b)));
    CHECK_CUDA(cudaMalloc(&d_c, count * sizeof(*d_c)));
    CHECK_CUDA(cudaMemcpy(
        d_a, a.data(), count * sizeof(*d_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, b.data(), count * sizeof(*d_b), cudaMemcpyHostToDevice));

    std::vector<double> event_times;
    std::vector<double> wall_times;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int i = 0; i < warmups; ++i) {
            launch_kernel(
                KernelVersion::V1, d_a, d_b, d_c, size, size, size);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        const auto wall_start = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            launch_kernel(
                KernelVersion::V1, d_a, d_b, d_c, size, size, size);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        const auto wall_stop = std::chrono::steady_clock::now();

        float event_total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&event_total_ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        const double event_ms = event_total_ms / iterations;
        const double wall_total_ms =
            std::chrono::duration<double, std::milli>(
                wall_stop - wall_start).count();
        const double wall_per_timed_launch = wall_total_ms / iterations;
        event_times.push_back(event_ms);
        wall_times.push_back(wall_per_timed_launch);
        std::cout << std::fixed << std::setprecision(6)
                  << "V1_TIMING_AUDIT repeat=" << repeat + 1
                  << " event_ms=" << event_ms
                  << " wall_ms_per_timed_launch=" << wall_per_timed_launch
                  << " wall_total_ms=" << wall_total_ms << '\n';
    }
    std::cout << std::fixed << std::setprecision(6)
              << "V1_TIMING_AUDIT_MEDIAN event_ms=" << median(event_times)
              << " wall_ms_per_timed_launch=" << median(wall_times)
              << " protocol=10_warmups_then_same_50_launch_interval\n";

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return true;
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
    if (mode == "--profile-v2") {
        return run_profile_workload(KernelVersion::V2)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (mode == "--profile-v3") {
        return run_profile_workload(KernelVersion::V3)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (mode == "--v1-timing-audit") {
        return run_v1_timing_audit() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    std::cerr << "Usage: " << argv[0]
              << " [--correctness|--benchmark|--profile-v2|--profile-v3|"
                 "--v1-timing-audit]\n";
    return EXIT_FAILURE;
}
