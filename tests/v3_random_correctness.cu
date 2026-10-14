#include <chrono>
#include <cstdint>
#include <random>
#include <string>

// Reuse the exact kernels, launch path, CPU reference, and tolerance checks
// from the production validation runner without modifying that source file.
#define main tensor_core_gemm_original_main
#include "../src/tensor_core_gemm.cu"
#undef main

enum class InputDistribution {
    SignedUniform,
    PositiveUniform,
    NegativeUniform,
    NearZeroUniform,
    MixedNormal,
};

struct RandomCase {
    int M;
    int N;
    int K;
    uint32_t seed;
    InputDistribution distribution;
    const char* name;
};

const char* distributionName(InputDistribution distribution)
{
    switch (distribution) {
        case InputDistribution::SignedUniform:
            return "signed_uniform_with_zero_and_near_zero";
        case InputDistribution::PositiveUniform:
            return "positive_uniform_with_zero";
        case InputDistribution::NegativeUniform:
            return "negative_uniform_with_zero";
        case InputDistribution::NearZeroUniform:
            return "near_zero_signed_uniform";
        case InputDistribution::MixedNormal:
            return "mixed_normal_with_zero_and_near_zero";
    }
    return "unknown";
}

void fillRandomValues(
    vector<float>& values,
    mt19937& generator,
    InputDistribution distribution)
{
    uniform_real_distribution<float> signed_uniform(-0.5f, 0.5f);
    uniform_real_distribution<float> positive_uniform(0.0f, 0.25f);
    uniform_real_distribution<float> negative_uniform(-0.25f, 0.0f);
    uniform_real_distribution<float> near_zero_uniform(-1.0e-3f, 1.0e-3f);
    normal_distribution<float> mixed_normal(0.0f, 0.1f);

    for (size_t i = 0; i < values.size(); ++i) {
        float value = 0.0f;
        switch (distribution) {
            case InputDistribution::SignedUniform:
                value = signed_uniform(generator);
                break;
            case InputDistribution::PositiveUniform:
                value = positive_uniform(generator);
                break;
            case InputDistribution::NegativeUniform:
                value = negative_uniform(generator);
                break;
            case InputDistribution::NearZeroUniform:
                value = near_zero_uniform(generator);
                break;
            case InputDistribution::MixedNormal:
                value = max(-0.5f, min(0.5f, mixed_normal(generator)));
                break;
        }

        // Inject values that random sampling alone would almost never cover.
        if (i % 97 == 0) {
            value = 0.0f;
        } else if (i % 89 == 0) {
            if (distribution == InputDistribution::PositiveUniform) {
                value = 1.0e-6f;
            } else if (distribution == InputDistribution::NegativeUniform) {
                value = -1.0e-6f;
            } else {
                value = (i & 1) ? 1.0e-6f : -1.0e-6f;
            }
        }
        values[i] = value;
    }
}

void printStats(const char* implementation, const CorrectnessResult& result)
{
    cout << scientific << setprecision(9)
         << implementation << " original_ref max_abs="
         << result.original_stats.max_abs_error << " max_rel="
         << result.original_stats.max_rel_error << '\n'
         << implementation << " quantized_ref max_abs="
         << result.quantized_stats.max_abs_error << " max_rel="
         << result.quantized_stats.max_rel_error << " result="
         << (result.correct ? "PASS" : "FAIL") << endl;
}

void printPairStats(
    const char* pair_name,
    const vector<float>& reference,
    const vector<float>& result)
{
    const ErrorStats stats = compareResults(
        reference, result,
        numeric_limits<float>::infinity(),
        numeric_limits<float>::infinity());
    cout << scientific << setprecision(9) << pair_name
         << " max_abs=" << stats.max_abs_error
         << " max_rel=" << stats.max_rel_error
         << " finite=" << (stats.finite ? "yes" : "no") << endl;
}

bool runRandomCase(const RandomCase& test, cublasHandle_t cublas_handle)
{
    const size_t A_elements = static_cast<size_t>(test.M) * test.K;
    const size_t B_elements = static_cast<size_t>(test.K) * test.N;
    const size_t C_elements = static_cast<size_t>(test.M) * test.N;
    const size_t A_bytes = A_elements * sizeof(half);
    const size_t B_bytes = B_elements * sizeof(half);
    const size_t C_bytes = C_elements * sizeof(float);

    vector<float> h_A_fp32(A_elements);
    vector<float> h_B_fp32(B_elements);
    mt19937 generator(test.seed);
    fillRandomValues(h_A_fp32, generator, test.distribution);
    fillRandomValues(h_B_fp32, generator, test.distribution);

    vector<half> h_A_fp16(A_elements);
    vector<half> h_B_fp16(B_elements);
    vector<float> h_A_quantized(A_elements);
    vector<float> h_B_quantized(B_elements);
    for (size_t i = 0; i < A_elements; ++i) {
        h_A_fp16[i] = __float2half(h_A_fp32[i]);
        h_A_quantized[i] = __half2float(h_A_fp16[i]);
    }
    for (size_t i = 0; i < B_elements; ++i) {
        h_B_fp16[i] = __float2half(h_B_fp32[i]);
        h_B_quantized[i] = __half2float(h_B_fp16[i]);
    }

    vector<float> original_reference(C_elements);
    vector<float> quantized_reference(C_elements);
    const auto reference_start = chrono::steady_clock::now();
    gemmCPU(h_A_fp32.data(), h_B_fp32.data(), original_reference.data(),
            test.M, test.N, test.K);
    gemmCPU(h_A_quantized.data(), h_B_quantized.data(),
            quantized_reference.data(), test.M, test.N, test.K);
    const double reference_seconds = chrono::duration<double>(
        chrono::steady_clock::now() - reference_start).count();

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, C_bytes));
    CHECK_CUDA(cudaMemcpy(
        d_A, h_A_fp16.data(), A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_B, h_B_fp16.data(), B_bytes, cudaMemcpyHostToDevice));

    vector<float> h_C(C_elements);
    const CorrectnessResult v2_1 = checkKernel(
        KernelVersion::V2_1, d_A, d_B, d_C, h_C,
        original_reference, quantized_reference,
        test.M, test.N, test.K, C_bytes, cublas_handle);
    const vector<float> h_C_v2_1 = h_C;

    const CorrectnessResult v3 = checkKernel(
        KernelVersion::V3, d_A, d_B, d_C, h_C,
        original_reference, quantized_reference,
        test.M, test.N, test.K, C_bytes, cublas_handle);
    const vector<float> h_C_v3 = h_C;

    const CorrectnessResult cublas = checkKernel(
        KernelVersion::CUBLAS, d_A, d_B, d_C, h_C,
        original_reference, quantized_reference,
        test.M, test.N, test.K, C_bytes, cublas_handle);

    cout << "\nCASE " << test.name << " shape="
         << test.M << "x" << test.N << "x" << test.K
         << " seed=" << test.seed
         << " distribution=" << distributionName(test.distribution)
         << " cpu_reference_seconds=" << fixed << setprecision(3)
         << reference_seconds << endl;
    printStats("V2.1", v2_1);
    printStats("V3", v3);
    printStats("cuBLAS", cublas);
    printPairStats("V2.1_vs_V3", h_C_v2_1, h_C_v3);
    printPairStats("V3_vs_cuBLAS", h_C_v3, h_C);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return v2_1.correct && v3.correct && cublas.correct;
}

int main()
{
    const vector<RandomCase> cases = {
        {128, 192, 64, 1101u, InputDistribution::SignedUniform,
         "signed_small_nonsquare"},
        {256, 256, 256, 2202u, InputDistribution::PositiveUniform,
         "positive_square"},
        {512, 512, 512, 3303u, InputDistribution::NegativeUniform,
         "negative_square"},
        {512, 1024, 256, 4404u, InputDistribution::NearZeroUniform,
         "near_zero_wide"},
        {1024, 512, 1024, 5505u, InputDistribution::MixedNormal,
         "mixed_normal_tall"},
    };

    cudaDeviceProp device_prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
    cout << "GPU=" << device_prop.name << " compute_capability="
         << device_prop.major << "." << device_prop.minor << endl;
    cout << "Pass criterion: abs_error <= 1e-2 + 1e-2 * abs(quantized_ref)"
         << endl;

    cublasHandle_t cublas_handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, nullptr));
    CHECK_CUBLAS(cublasSetPointerMode(
        cublas_handle, CUBLAS_POINTER_MODE_HOST));

    bool all_passed = true;
    for (const RandomCase& test : cases) {
        all_passed = runRandomCase(test, cublas_handle) && all_passed;
    }
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    cout << "\nFINAL_RESULT=" << (all_passed ? "PASS" : "FAIL") << endl;
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
