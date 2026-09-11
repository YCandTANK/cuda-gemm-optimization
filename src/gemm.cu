#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                           \
do {                                                                 \
    cublasStatus_t status = (call);                                  \
    if (status != CUBLAS_STATUS_SUCCESS) {                           \
        fprintf(stderr, "cuBLAS error: %s (%s:%d)\n",                \
                cublasGetStatusString(status), __FILE__, __LINE__);  \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

constexpr int TILE_SIZE = 16;
constexpr int BM = 16;//一个 Block 负责 C 的 16 行
constexpr int BN = 64;//一个 Block 负责 C 的 64 列
constexpr int BK = 16;//BK = 16   每轮沿 K 方向计算 16 个元素
constexpr int TN = 4;//TN = 4    一个 Thread 横向负责 4 个 C

//V0
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

//V1
__global__ void gemmTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        int row_A = row;
        int col_A = tile * TILE_SIZE + tx;
        int row_B = tile * TILE_SIZE + ty;
        int col_B = col;

        //存入
        if(row_A < M && col_A < K){
            As[ty][tx] = A[row_A * K + col_A];
        }else{
            As[ty][tx] = 0.0f;
        }

        if(row_B < K && col_B < N){
            Bs[ty][tx] = B[row_B * N + col_B];
        }else{
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        //计算sum
        for(int k = 0; k < TILE_SIZE; k++){
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    //写结果
    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

//V2
__global__ void gemmThreadTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //计算global_row
    int global_row = blockIdx.y * BM + ty;

    int block_col = blockIdx.x * BN;//计算所处的块的起始坐标，之后的global_col就是block_col + local_col

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sum[TN] = {0.0f};

    for(int tile = 0; tile < (K + BK - 1) / BK; tile++){
        //计算A的坐标，和B的行坐标
        int row_A = global_row;
        int col_A = tile * BK + tx;

        int row_B = tile * BK + ty;

        //存入
        //存入A
        if(row_A < M && col_A < K){
            As[ty][tx] = A[row_A * K + col_A];
        }else{
            As[ty][tx] = 0.0f;
        }

        for(int i = 0; i < TN; i++){
            //计算当前的Bs列坐标,和其计算得出的C的列坐标，也即B的列坐标
            int local_col = tx + i * blockDim.x;
            int global_col = block_col + local_col;
            int col_B = global_col;//虽然多此一举，但是这样我方便理解
            //存入B
            if(row_B < K && col_B < N){
                Bs[ty][local_col] = B[row_B * N + col_B];
            }else{
                Bs[ty][local_col] = 0.0f;
            }
        }

        __syncthreads();

        //计算乘加
        for(int k = 0; k < BK; k++){
            float a = As[ty][k];
            for(int i = 0; i < TN; i++){
                int local_col = tx + i * blockDim.x;

                sum[i] += a * Bs[k][local_col];
            }
        }

        __syncthreads();
    }

    //写结果到C
    for(int i = 0; i < TN; i++){
        int local_col = tx + i * blockDim.x;
        int global_col = block_col + local_col;

        if(global_row < M && global_col < N){
            C[global_row * N + global_col] = sum[i];
        }
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
    int N = 1024;
    int K = 1024;

    // int M = 130;
    // int N = 129;
    // int K = 137;

    const int warmup_runs = 10;
    const int benchmark_runs = 100;

    //计算block和grid
    dim3 block_v1(TILE_SIZE, TILE_SIZE);
    dim3 grid_v1(
        (N + block_v1.x - 1) / block_v1.x,
        (M + block_v1.y - 1) / block_v1.y
    );

    dim3 block_v2(
        BN / TN,//因为一个线程会计算C的TN个列，所以block里线程的列数要用BN / TN
        BM
    );
    dim3 grid_v2(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
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
    vector<float> h_tiled_C(M * N);
    vector<float> h_thread_tiled_C(M * N);
    vector<float> h_cublas_C(M * N);

    //CPU Reference
    gemmCPU(h_A.data(), h_B.data(), h_cpu_C.data(), M, N, K);

    //准备Device数组
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_naive = nullptr;
    float* d_C_tiled = nullptr;
    float* d_C_thread_tiled = nullptr;
    float* d_C_cublas = nullptr;

    size_t A_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t B_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t C_bytes = static_cast<size_t>(M) * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_naive, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_tiled, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_thread_tiled, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_cublas, C_bytes));

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
        gemmNaive<<<grid_v1, block_v1>>>(
            d_A, d_B, d_C_naive,
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
        gemmNaive<<<grid_v1, block_v1>>>(
            d_A, d_B, d_C_naive,
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

    cout << "Naive average latency = "
        << avg_ms << " ms" << endl;

    cout << "Naive GFLOPS = "
        << gflops << endl;
    
    cout<<endl;
    
/************************************************************************************ */

    //Tiled warm up
    for(int i = 0; i < warmup_runs; ++i) {
        gemmTiled<<<grid_v1, block_v1>>>(
            d_A, d_B, d_C_tiled,
            M, N, K
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //创建Tiled event来计时
    cudaEvent_t tiled_start, tiled_stop;

    CHECK_CUDA(cudaEventCreate(&tiled_start));
    CHECK_CUDA(cudaEventCreate(&tiled_stop));

    CHECK_CUDA(cudaEventRecord(tiled_start));

    for (int i = 0; i < benchmark_runs; ++i) {
        gemmTiled<<<grid_v1, block_v1>>>(
            d_A, d_B, d_C_tiled,
            M, N, K
        );
    }

    CHECK_CUDA(cudaEventRecord(tiled_stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(tiled_stop));

    float tiled_total_ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &tiled_total_ms,
            tiled_start,
            tiled_stop
        )
    );

    float tiled_avg_ms =
        tiled_total_ms / benchmark_runs;

    double tiled_gflops =
        flops / (tiled_avg_ms / 1000.0) / 1e9;

    cout << "Tiled average latency = "
        << tiled_avg_ms << " ms" << endl;

    cout << "Tiled GFLOPS = "
        << tiled_gflops << endl;

    double tiled_naive_speedup =
        tiled_gflops / gflops;

    cout << "Tiled / Naive speedup = "
        << tiled_naive_speedup << endl;

    cout<<endl;
    
/************************************************************************************ */

    //Thread Tiled warm up
    for(int i = 0; i < warmup_runs; ++i) {
        gemmThreadTiled<<<grid_v2, block_v2>>>(
            d_A,
            d_B,
            d_C_thread_tiled,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //创建Thread Tiled event来计时
    cudaEvent_t thread_tiled_start, thread_tiled_stop;

    CHECK_CUDA(cudaEventCreate(&thread_tiled_start));
    CHECK_CUDA(cudaEventCreate(&thread_tiled_stop));

    CHECK_CUDA(cudaEventRecord(thread_tiled_start));

    for (int i = 0; i < benchmark_runs; ++i) {
        gemmThreadTiled<<<grid_v2, block_v2>>>(
            d_A,
            d_B,
            d_C_thread_tiled,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaEventRecord(thread_tiled_stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(thread_tiled_stop));

    float thread_tiled_total_ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &thread_tiled_total_ms,
            thread_tiled_start,
            thread_tiled_stop
        )
    );

    float thread_tiled_avg_ms =
        thread_tiled_total_ms / benchmark_runs;

    double thread_tiled_gflops =
        flops / (thread_tiled_avg_ms / 1000.0) / 1e9;

    cout << "Thread Tiled average latency = "
        << thread_tiled_avg_ms << " ms" << endl;

    cout << "Thread Tiled GFLOPS = "
        << thread_tiled_gflops << endl;

    double thread_tiled_naive_speedup =
        thread_tiled_gflops / gflops;

    cout << "Thread Tiled / Naive speedup = "
        << thread_tiled_naive_speedup << endl;

    double thread_tiled_tiled_speedup =
        thread_tiled_gflops / tiled_gflops;

    cout << "Thread Tiled / Tiled speedup = "
        << thread_tiled_tiled_speedup << endl;


    cout<<endl;
    
/************************************************************************************ */

    //创建 cuBLAS handle，执行cuBLAS内的工业级GEMM，以此为参照
    cublasHandle_t handle;

    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    //warm up
    for (int i = 0; i < warmup_runs; ++i) {
        CHECK_CUBLAS(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C_cublas, N
            )
        );
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    //正式测试
    cudaEvent_t cublas_start, cublas_stop;

    CHECK_CUDA(cudaEventCreate(&cublas_start));
    CHECK_CUDA(cudaEventCreate(&cublas_stop));


    CHECK_CUDA(cudaEventRecord(cublas_start));

    for (int i = 0; i < benchmark_runs; ++i) {
        CHECK_CUBLAS(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C_cublas, N
            )
        );
    }

    CHECK_CUDA(cudaEventRecord(cublas_stop));
    CHECK_CUDA(cudaEventSynchronize(cublas_stop));

    //计算cublasSgemm的性能
    float cublas_total_ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &cublas_total_ms,
            cublas_start,
            cublas_stop
        )
    );

    float cublas_avg_ms =
        cublas_total_ms / benchmark_runs;

    double cublas_gflops =
        flops / (cublas_avg_ms / 1000.0) / 1e9;

    cout << "cuBLAS average latency = "
        << cublas_avg_ms << " ms" << endl;

    cout << "cuBLAS GFLOPS = "
        << cublas_gflops << endl;

    double naive_cublas_percentage =
        gflops / cublas_gflops * 100.0;

    cout << "Naive GFLOPS / cuBLAS GFLOPS = "
        << naive_cublas_percentage << "%" << endl;

    double tiled_cublas_percentage =
        tiled_gflops / cublas_gflops * 100.0;

    cout << "Tiled GFLOPS / cuBLAS GFLOPS = "
        << tiled_cublas_percentage << "%" << endl;

    double thread_tiled_cublas_percentage =
        thread_tiled_gflops / cublas_gflops * 100.0;

    cout << "Thread Tiled GFLOPS / cuBLAS GFLOPS = "
        << thread_tiled_cublas_percentage << " %" << endl;

/******************************************************************************** */

    //D2H
    CHECK_CUDA(cudaMemcpy(
        h_gpu_C.data(),
        d_C_naive,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        h_tiled_C.data(),
        d_C_tiled,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        h_thread_tiled_C.data(),
        d_C_thread_tiled,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        h_cublas_C.data(),
        d_C_cublas,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    //Naive正确性检验
    float naive_max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_gpu_C[i]);

        naive_max_abs_error =
            max(naive_max_abs_error, error);
    }

    cout << "Naive max absolute error = "
        << naive_max_abs_error << endl;

    if (naive_max_abs_error < 1e-5f) {
        cout << "Naive: Correct!" << endl;
    } else {
        cout << "Naive: Mismatch!" << endl;
    }

    cout<<endl;

    //Tiled正确性检验
    float tiled_max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_tiled_C[i]);

        tiled_max_abs_error =
            max(tiled_max_abs_error, error);
    }

    cout << "Tiled max absolute error = "
        << tiled_max_abs_error << endl;

    if (tiled_max_abs_error < 1e-5f) {
        cout << "Tiled: Correct!" << endl;
    } else {
        cout << "Tiled: Mismatch!" << endl;
    }

    cout<<endl;

    //Thread Tiled正确性检验
    float thread_tiled_max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_thread_tiled_C[i]);

        thread_tiled_max_abs_error =
            max(thread_tiled_max_abs_error, error);
    }

    cout << "Thread Tiled max absolute error = "
        << thread_tiled_max_abs_error << endl;

    if (thread_tiled_max_abs_error < 1e-5f) {
        cout << "Thread Tiled: Correct!" << endl;
    } else {
        cout << "Thread Tiled: Mismatch!" << endl;
    }

    cout<<endl;
    
    //cuBLAS正确性检验
    float cublas_max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_cublas_C[i]);

        cublas_max_abs_error =
            max(cublas_max_abs_error, error);
    }

    cout << "cuBLAS max absolute error = "
        << cublas_max_abs_error << endl;

    if (cublas_max_abs_error < 1e-5f) {
        cout << "cuBLAS: Correct!" << endl;
    } else {
        cout << "cuBLAS: Mismatch!" << endl;
    }

    cout<<endl;
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_naive));
    CHECK_CUDA(cudaFree(d_C_tiled));
    CHECK_CUDA(cudaFree(d_C_thread_tiled));
    CHECK_CUDA(cudaFree(d_C_cublas));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(tiled_start));
    CHECK_CUDA(cudaEventDestroy(tiled_stop));
    CHECK_CUDA(cudaEventDestroy(thread_tiled_start));
    CHECK_CUDA(cudaEventDestroy(thread_tiled_stop));
    CHECK_CUDA(cudaEventDestroy(cublas_start));
    CHECK_CUDA(cudaEventDestroy(cublas_stop));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
