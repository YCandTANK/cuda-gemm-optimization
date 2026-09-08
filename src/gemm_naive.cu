#include<bits/stdc++.h>
#include<cuda_runtime.h>

using namespace std;

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void gemmNaive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    if(col < N && row < M){

        for(int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }

}

//CPU Reference
void gemmCPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    for(int row = 0; row < M; row++){
        for(int col = 0; col < N; col++){
            float sum = 0.0f;
            for(int k = 0; k < K; k++){
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main(){
    //初始化数据
    int M = 1024;
    int K = 1024;
    int N = 1024;

    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;

    const int warmup_runs = 10;
    const int benchmark_runs = 100;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    //准备Host数据
    vector<float> h_A(M * K);
    vector<float> h_B(K * N);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>((i % 17) - 8) / 10.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>((i % 13) - 6) / 10.0f;
    }

    vector<float> h_cpu_C(M * N);
    vector<float> h_gpu_C(M * N);

    //CPU Reference
    gemmCPU(h_A.data(), h_B.data(), h_cpu_C.data(), M, N, K);

    //准备Device数组
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t A_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t B_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t C_bytes = static_cast<size_t>(M) * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, C_bytes));

    //H2D
    CHECK_CUDA(cudaMemcpy(
        d_A,
        h_A.data(),
        A_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_B,
        h_B.data(),
        B_bytes,
        cudaMemcpyHostToDevice
    ));

    //warm up
    for(int i = 0; i < warmup_runs; ++i) {
        gemmNaive<<<grid, block>>>(
            d_A, d_B, d_C,
            M, N, K
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //创建event来计时
    cudaEvent_t start, stop;
    
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    //kernel launch
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < benchmark_runs; ++i) {
        gemmNaive<<<grid, block>>>(
            d_A, d_B, d_C,
            M, N, K
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    //计时
    float total_ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    float avg_ms = total_ms / benchmark_runs;

    //计算GFLOPS
    //floating-point operations（浮点运算次数）
    double flops =2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double seconds = avg_ms / 1000.0;//毫秒转换成秒
    double gflops = flops / seconds / 1e9; 

    cout << "Average latency = "
        << avg_ms << " ms" << endl;

    cout << "Performance = "
        << gflops << " GFLOPS" << endl;

    //D2H
    CHECK_CUDA(cudaMemcpy(
        h_gpu_C.data(),
        d_C,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    //正确性检验
    float max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_gpu_C[i]);

        max_abs_error =
            max(max_abs_error, error);
    }

    cout << "Max absolute error = "
        << max_abs_error << endl;

    if (max_abs_error < 1e-5f) {
        cout << "Correct!" << endl;
    } else {
        cout << "Mismatch!" << endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
