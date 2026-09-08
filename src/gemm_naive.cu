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
    int M = 2;
    int K = 3;
    int N = 4;

    //准备Host数据
    vector<float> h_A = {
        1, 2, 3,
        4, 5, 6
    };

    vector<float> h_B = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

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

    //kernel launch
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    gemmNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

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

    return 0;
}
