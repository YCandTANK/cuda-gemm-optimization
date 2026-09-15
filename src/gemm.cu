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
constexpr int BM_V2 = 16;//一个 Block 负责 C 的 16 行
constexpr int BN_V2 = 64;//一个 Block 负责 C 的 64 列
constexpr int BK_V2 = 16;//BK = 16   每轮沿 K 方向计算 16 个元素
constexpr int TN_V2 = 4;//TN = 4    一个 Thread 横向负责 4 个 C

constexpr int BM_V3 = 64;
constexpr int BN_V3 = 64;
constexpr int BK_V3 = 16;
constexpr int TM_V3 = 4;
constexpr int TN_V3 = 4;

constexpr int BM_V4 = 64;
constexpr int BN_V4 = 64;
constexpr int BK_V4 = 16;
constexpr int TM_V4 = 4;
constexpr int TN_V4 = 4;
constexpr int VEC = 4;

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
    int global_row = blockIdx.y * BM_V2 + ty;

    int block_col = blockIdx.x * BN_V2;//计算所处的块的起始坐标，之后的global_col就是block_col + local_col

    __shared__ float As[BM_V2][BK_V2];
    __shared__ float Bs[BK_V2][BN_V2];

    float sum[TN_V2] = {0.0f};

    for(int tile = 0; tile < (K + BK_V2 - 1) / BK_V2; tile++){
        //计算A的坐标，和B的行坐标
        int row_A = global_row;
        int col_A = tile * BK_V2 + tx;

        int row_B = tile * BK_V2 + ty;

        //存入
        //存入A
        if(row_A < M && col_A < K){
            As[ty][tx] = A[row_A * K + col_A];
        }else{
            As[ty][tx] = 0.0f;
        }

        for(int i = 0; i < TN_V2; i++){
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
        for(int k = 0; k < BK_V2; k++){
            float a = As[ty][k];
            for(int i = 0; i < TN_V2; i++){
                int local_col = tx + i * blockDim.x;

                sum[i] += a * Bs[k][local_col];
            }
        }

        __syncthreads();
    }

    //写结果到C
    for(int i = 0; i < TN_V2; i++){
        int local_col = tx + i * blockDim.x;
        int global_col = block_col + local_col;

        if(global_row < M && global_col < N){
            C[global_row * N + global_col] = sum[i];
        }
    }
}

//V3
__global__ void gemm2DThreadTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;//thread 在 block 内线性编号
    int num_threads = blockDim.x * blockDim.y;

    __shared__ float As[BM_V3][BK_V3];
    __shared__ float Bs[BK_V3][BN_V3];

    float sum[TM_V3][TN_V3] = {0.0f};

    //local_row/local_col → 访问 Shared Memory
    int local_row = ty * TM_V3;
    int local_col = tx * TN_V3;

    //C Tile中左上角数据坐标
    //global_row/global_col → 写 Global C
    int global_row = blockIdx.y * BM_V3 + local_row;
    int global_col = blockIdx.x * BN_V3 + local_col;

    for(int tile = 0; tile < (K + BK_V3 - 1) / BK_V3; tile++){
        //将显存中的A，B数据写到共享内存As，Bs
        //A -> As
        for(int idx = tid; idx < BM_V3 * BK_V3; idx += num_threads){
            //计算As的row和col
            int row_As = idx / BK_V3;
            int col_As = idx % BK_V3;
            //计算对应的A的坐标
            int row_A = blockIdx.y * BM_V3 + row_As;
            int col_A = tile * BK_V3 + col_As;

            if(row_A < M && col_A < K){
                As[row_As][col_As] = A[row_A * K + col_A];
            }else{
                As[row_As][col_As] = 0.0f;
            }
            
        }
        //B -> Bs
        for(int idx = tid; idx < BK_V3 * BN_V3; idx += num_threads){
            //计算Bs的行列坐标
            int row_Bs = idx / BN_V3;
            int col_Bs = idx % BN_V3;
            //计算对应的B的行列坐标
            int row_B = tile * BK_V3 + row_Bs;
            int col_B = blockIdx.x * BN_V3 + col_Bs;

            if(row_B < K && col_B < N){
                Bs[row_Bs][col_Bs] = B[row_B * N + col_B];
            }else{
                Bs[row_Bs][col_Bs] = 0.0f;
            }
        }

        __syncthreads();

        //计算乘加
        for(int k = 0; k < BK_V3; k++){
            float reg_A[TM_V3];
            float reg_B[TN_V3];
            //As -> reg_A
            for(int i = 0; i < TM_V3; i++){
                reg_A[i] = As[local_row + i][k];
            }
            //Bs -> reg_B
            for(int i = 0; i < TN_V3; i++){
                reg_B[i] = Bs[k][local_col + i];
            }
            //计算sum
            for(int i = 0; i < TM_V3; i++){
                for(int j = 0; j < TN_V3; j++){
                    sum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    //结果写回C
    for(int i = 0; i < TM_V3; i++){
        for(int j = 0; j < TN_V3; j++){
            //计算C的行列
            int row_C = global_row + i;
            int col_C = global_col + j;
            if(row_C < M && col_C < N){
                C[row_C * N + col_C] = sum[i][j];
            }
        }
    }
}

//V4
__global__ void gemm2DThreadTiledVectorized(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;//thread 在 block 内线性编号

    __shared__ float As[BM_V4][BK_V4];
    __shared__ float Bs[BK_V4][BN_V4];

    float sum[TM_V4][TN_V4] = {0.0f};

    //local_row/local_col → 访问 Shared Memory
    int local_row = ty * TM_V4;
    int local_col = tx * TN_V4;

    //C Tile中左上角数据坐标
    //global_row/global_col → 写 Global C
    int global_row = blockIdx.y * BM_V4 + local_row;
    int global_col = blockIdx.x * BN_V4 + local_col;

    //将显存中的A，B数据写到共享内存As，Bs
    for(int tile = 0; tile < (K + BK_V4 - 1) / BK_V4; tile++){
        //A -> As
        int a_vecs_per_row = BK_V4 / VEC;

        //计算As的起始行列坐标
        int row_As = tid / a_vecs_per_row;
        int col_As = (tid % a_vecs_per_row) * VEC;

        //计算A的起始行列坐标,为了计算linear_idx_A
        int row_A = blockIdx.y * BM_V4 + row_As;
        int col_A = tile * BK_V4 + col_As;

        //计算A起始坐标的线性坐标，用于判断是否16bytes对齐
        int linear_idx_A = row_A * K + col_A;

        //搬取数据
        if(row_A < M && col_A + VEC - 1 < K && linear_idx_A % VEC == 0){//边界合法且16字节对齐
            float4 vec = *reinterpret_cast<const float4*>(&A[linear_idx_A]);//使用float4一次性取4个数据
            As[row_As][col_As + 0] = vec.x;
            As[row_As][col_As + 1] = vec.y;
            As[row_As][col_As + 2] = vec.z;
            As[row_As][col_As + 3] = vec.w;
        }else{
            for(int i = 0; i < VEC; i++){
                int current_col_A = col_A + i;

                if(row_A < M && current_col_A < K){
                    As[row_As][col_As + i] = A[linear_idx_A + i];
                }else{
                    As[row_As][col_As + i] = 0.0f;
                }
            }
        }

        //B -> Bs
        int b_vecs_per_row = BN_V4 / VEC;

        //Bs坐标
        int row_Bs = tid / b_vecs_per_row;
        int col_Bs = (tid % b_vecs_per_row) * VEC;

        //B坐标
        int row_B = tile * BK_V4 + row_Bs;
        int col_B = blockIdx.x * BN_V4 + col_Bs;

        int linear_idx_B = row_B * N + col_B;

        if(row_B < K && col_B + VEC - 1 < N && linear_idx_B % VEC == 0){
            float4 vec = *reinterpret_cast<const float4*>(&B[linear_idx_B]);
            Bs[row_Bs][col_Bs + 0] = vec.x;
            Bs[row_Bs][col_Bs + 1] = vec.y;
            Bs[row_Bs][col_Bs + 2] = vec.z;
            Bs[row_Bs][col_Bs + 3] = vec.w;
        }else{
            for(int i = 0; i < VEC; i++){
                int current_col_B = col_B + i;

                if(row_B < K && current_col_B < N){
                    Bs[row_Bs][col_Bs + i] =
                        B[row_B * N + current_col_B];
                }else{
                    Bs[row_Bs][col_Bs + i] = 0.0f;
                }
            }
        }

        __syncthreads();

        //计算乘加
        for(int k = 0; k < BK_V4; k++){
            float reg_A[TM_V4];
            float reg_B[TN_V4];
            //As -> reg_A
            for(int i = 0; i < TM_V4; i++){
                reg_A[i] = As[local_row + i][k];
            }
            //Bs -> reg_B
            for(int i = 0; i < TN_V4; i++){
                reg_B[i] = Bs[k][local_col + i];
            }
            //计算sum
            for(int i = 0; i < TM_V4; i++){
                for(int j = 0; j < TN_V4; j++){
                    sum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    //结果写回C
    for(int i = 0; i < TM_V4; i++){
        for(int j = 0; j < TN_V4; j++){
            //计算C的行列
            int row_C = global_row + i;
            int col_C = global_col + j;
            if(row_C < M && col_C < N){
                C[row_C * N + col_C] = sum[i][j];
            }
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
        BN_V2 / TN_V2,//因为一个线程会计算C的TN个列，所以block里线程的列数要用BN / TN
        BM_V2
    );
    dim3 grid_v2(
        (N + BN_V2 - 1) / BN_V2,
        (M + BM_V2 - 1) / BM_V2
    );

    dim3 block_v3(
        BN_V3 / TN_V3,
        BM_V3 / TM_V3
    );
    dim3 grid_v3(
        (N + BN_V3 - 1) / BN_V3,
        (M + BM_V3 - 1) / BM_V3
    );

    dim3 block_v4(
        BN_V4 / TN_V4,
        BM_V4 / TM_V4
    );
    dim3 grid_v4(
        (N + BN_V4 - 1) / BN_V4,
        (M + BM_V4 - 1) / BM_V4
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
    vector<float> h_2d_thread_tiled_C(M * N);
    vector<float> h_vectorized_C(M * N);
    vector<float> h_cublas_C(M * N);

    //CPU Reference
    gemmCPU(h_A.data(), h_B.data(), h_cpu_C.data(), M, N, K);

    //准备Device数组
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_naive = nullptr;
    float* d_C_tiled = nullptr;
    float* d_C_thread_tiled = nullptr;
    float* d_C_2d_thread_tiled = nullptr;
    float* d_C_vectorized = nullptr;
    float* d_C_cublas = nullptr;

    size_t A_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t B_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t C_bytes = static_cast<size_t>(M) * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_naive, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_tiled, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_thread_tiled, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_2d_thread_tiled, C_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_vectorized, C_bytes));
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

    //2D Thread Tiled warm up
    for(int i = 0; i < warmup_runs; ++i) {
        gemm2DThreadTiled<<<grid_v3, block_v3>>>(
            d_A,
            d_B,
            d_C_2d_thread_tiled,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //创建2D Thread Tiled event来计时
    cudaEvent_t thread_tiled_2d_start, thread_tiled_2d_stop;

    CHECK_CUDA(cudaEventCreate(&thread_tiled_2d_start));
    CHECK_CUDA(cudaEventCreate(&thread_tiled_2d_stop));

    CHECK_CUDA(cudaEventRecord(thread_tiled_2d_start));

    for (int i = 0; i < benchmark_runs; ++i) {
        gemm2DThreadTiled<<<grid_v3, block_v3>>>(
            d_A,
            d_B,
            d_C_2d_thread_tiled,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaEventRecord(thread_tiled_2d_stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(thread_tiled_2d_stop));

    float thread_tiled_2d_total_ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &thread_tiled_2d_total_ms,
            thread_tiled_2d_start,
            thread_tiled_2d_stop
        )
    );

    float thread_tiled_2d_avg_ms =
        thread_tiled_2d_total_ms / benchmark_runs;

    double thread_tiled_2d_gflops =
        flops / (thread_tiled_2d_avg_ms / 1000.0) / 1e9;

    cout << "2D Thread Tiled average latency = "
        << thread_tiled_2d_avg_ms << " ms" << endl;

    cout << "2D Thread Tiled GFLOPS = "
        << thread_tiled_2d_gflops << endl;

    double thread_tiled_2d_naive_speedup =
        thread_tiled_2d_gflops / gflops;

    cout << "2D Thread Tiled / Naive speedup = "
        << thread_tiled_2d_naive_speedup << endl;

    double thread_tiled_2d_tiled_speedup =
        thread_tiled_2d_gflops / tiled_gflops;

    cout << "2D Thread Tiled / Tiled speedup = "
        << thread_tiled_2d_tiled_speedup << endl;

    double thread_tiled_2d_thread_tiled_speedup =
        thread_tiled_2d_gflops / thread_tiled_gflops;

    cout << "2D Thread Tiled / Thread Tiled speedup = "
        << thread_tiled_2d_thread_tiled_speedup << endl;

    cout<<endl;

/************************************************************************************ */

    //Vectorized warm up
    for(int i = 0; i < warmup_runs; ++i) {
        gemm2DThreadTiledVectorized<<<grid_v4, block_v4>>>(
            d_A,
            d_B,
            d_C_vectorized,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    //创建Vectorized event来计时
    cudaEvent_t vectorized_start, vectorized_stop;

    CHECK_CUDA(cudaEventCreate(&vectorized_start));
    CHECK_CUDA(cudaEventCreate(&vectorized_stop));

    CHECK_CUDA(cudaEventRecord(vectorized_start));

    for (int i = 0; i < benchmark_runs; ++i) {
        gemm2DThreadTiledVectorized<<<grid_v4, block_v4>>>(
            d_A,
            d_B,
            d_C_vectorized,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaEventRecord(vectorized_stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(vectorized_stop));

    float vectorized_total_ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &vectorized_total_ms,
            vectorized_start,
            vectorized_stop
        )
    );

    float vectorized_avg_ms =
        vectorized_total_ms / benchmark_runs;

    double vectorized_gflops =
        flops / (vectorized_avg_ms / 1000.0) / 1e9;

    cout << "Vectorized average latency = "
        << vectorized_avg_ms << " ms" << endl;

    cout << "Vectorized GFLOPS = "
        << vectorized_gflops << endl;

    double vectorized_naive_speedup =
        vectorized_gflops / gflops;

    cout << "Vectorized / Naive speedup = "
        << vectorized_naive_speedup << endl;

    double vectorized_tiled_speedup =
        vectorized_gflops / tiled_gflops;

    cout << "Vectorized / Tiled speedup = "
        << vectorized_tiled_speedup << endl;

    double vectorized_thread_tiled_speedup =
        vectorized_gflops / thread_tiled_gflops;

    cout << "Vectorized / Thread Tiled speedup = "
        << vectorized_thread_tiled_speedup << endl;

    double vectorized_thread_tiled_2d_speedup =
        vectorized_gflops / thread_tiled_2d_gflops;

    cout << "Vectorized / 2D Thread Tiled speedup = "
        << vectorized_thread_tiled_2d_speedup << endl;

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

    double thread_tiled_2d_cublas_percentage =
        thread_tiled_2d_gflops / cublas_gflops * 100.0;

    cout << "2D Thread Tiled GFLOPS / cuBLAS GFLOPS = "
        << thread_tiled_2d_cublas_percentage << " %" << endl;

    double vectorized_cublas_percentage =
        vectorized_gflops / cublas_gflops * 100.0;

    cout << "Vectorized GFLOPS / cuBLAS GFLOPS = "
        << vectorized_cublas_percentage << " %" << endl;

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
        h_2d_thread_tiled_C.data(),
        d_C_2d_thread_tiled,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        h_vectorized_C.data(),
        d_C_vectorized,
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

    //2D Thread Tiled正确性检验
    float thread_tiled_2d_max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_2d_thread_tiled_C[i]);

        thread_tiled_2d_max_abs_error =
            max(thread_tiled_2d_max_abs_error, error);
    }

    cout << "2D Thread Tiled max absolute error = "
        << thread_tiled_2d_max_abs_error << endl;

    if (thread_tiled_2d_max_abs_error < 1e-5f) {
        cout << "2D Thread Tiled: Correct!" << endl;
    } else {
        cout << "2D Thread Tiled: Mismatch!" << endl;
    }

    cout<<endl;

    //Vectorized正确性检验
    float vectorized_max_abs_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float error =
            fabsf(h_cpu_C[i] - h_vectorized_C[i]);

        vectorized_max_abs_error =
            max(vectorized_max_abs_error, error);
    }

    cout << "Vectorized max absolute error = "
        << vectorized_max_abs_error << endl;

    if (vectorized_max_abs_error < 1e-5f) {
        cout << "Vectorized: Correct!" << endl;
    } else {
        cout << "Vectorized: Mismatch!" << endl;
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
    CHECK_CUDA(cudaFree(d_C_2d_thread_tiled));
    CHECK_CUDA(cudaFree(d_C_vectorized));
    CHECK_CUDA(cudaFree(d_C_cublas));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(tiled_start));
    CHECK_CUDA(cudaEventDestroy(tiled_stop));
    CHECK_CUDA(cudaEventDestroy(thread_tiled_start));
    CHECK_CUDA(cudaEventDestroy(thread_tiled_stop));
    CHECK_CUDA(cudaEventDestroy(thread_tiled_2d_start));
    CHECK_CUDA(cudaEventDestroy(thread_tiled_2d_stop));
    CHECK_CUDA(cudaEventDestroy(vectorized_start));
    CHECK_CUDA(cudaEventDestroy(vectorized_stop));
    CHECK_CUDA(cudaEventDestroy(cublas_start));
    CHECK_CUDA(cudaEventDestroy(cublas_stop));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
