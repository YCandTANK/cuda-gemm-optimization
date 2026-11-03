#include "fp8_gemm.cuh"
#include <cstdint>

//基础版
//这一版设置一个 CTA 只有一个 Warp（32 线程），负责输出 C 的 16×8 Tile
__global__ void fp8_gemm_v1(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K)
{
    int lane = threadIdx.x;

    int group = lane / 4;
    int tid   = lane % 4;

    // 当前 CTA 负责的 C Tile 左上角
    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 8;

    // FP32 Accumulator
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    // FP8 原始字节编码
    // const uint8_t* A_bytes = reinterpret_cast<const uint8_t*>(A);
    // const uint8_t* B_bytes = reinterpret_cast<const uint8_t*>(B);

    for(int k0 = 0; k0 < K; k0 += 32){

        //A要4个32bit的寄存器，B要2个
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        // 根据 Lane/Fragment 映射，从 A/B 加载 FP8，打包进寄存器。
        //A
        #pragma unroll
        for(int r = 0; r < 4; r++){
            //计算第一个数据的位置
            int row = group + 8 * (r % 2);
            int col = 4 * tid + 16 * (r / 2);

            // 转换为 Global Memory 索引
            int global_row = m0 + row;
            int global_col = k0 + col;

            int index = global_row * K + global_col;

            //读取四个 FP8 原始编码
            uint32_t x0 = A[index + 0].__x;
            uint32_t x1 = A[index + 1].__x;
            uint32_t x2 = A[index + 2].__x;
            uint32_t x3 = A[index + 3].__x;

            //将四个 FP8 打包到一个 32-bit 寄存器
            a_frag[r] =
                x0
                | (x1 << 8)
                | (x2 << 16)
                | (x3 << 24);
        }

        //B
        #pragma unroll
        for(int r = 0; r < 2; r++){
            int row = 4 * tid + 16 * r;
            int col = group;

            // 转换为 Global Memory 索引
            int global_row = k0 + row;
            int global_col = n0 + col;

            int index = global_col * K + global_row;

            //读取四个 FP8 原始编码
            uint32_t x0 = B[index + 0].__x;
            uint32_t x1 = B[index + 1].__x;
            uint32_t x2 = B[index + 2].__x;
            uint32_t x3 = B[index + 3].__x;

            //将四个 FP8 打包到一个 32-bit 寄存器
            b_frag[r] =
                x0
                | (x1 << 8)
                | (x2 << 16)
                | (x3 << 24);
        }


        // 使用 mma.sync.m16n8k32，更新 acc[0..3]
        //使用next数组存每次mma计算得到的结果
        float next[4];

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"

            // 四个 FP32 输出
            : "=f"(next[0]),
            "=f"(next[1]),
            "=f"(next[2]),
            "=f"(next[3])

            // 填入输入操作数
            //A的4个寄存器
            :"r"(a_frag[0]), 
            "r"(a_frag[1]),
            "r"(a_frag[2]),
            "r"(a_frag[3]),

            //B的2个寄存器
            "r"(b_frag[0]), 
            "r"(b_frag[1]),

            //4个累加
            "f"(acc[0]),
            "f"(acc[1]),
            "f"(acc[2]),
            "f"(acc[3])
        );

        // 更新累加器
        acc[0] = next[0];
        acc[1] = next[1];
        acc[2] = next[2];
        acc[3] = next[3];       
    }


    // 根据 C Fragment 映射，将 acc[0..3] 写回 C
    #pragma unroll
    for(int i = 0; i < 4; i++){
        int row = group + 8 * (i / 2);
        int col = 2 * tid + 1 * (i % 2);

        int global_row = row + m0;
        int global_col = col + n0;

        int index = global_row * N + global_col;

        C[index] = acc[i];
    }
}


//32-bit Vectorized Fragment Loading
__global__ void fp8_gemm_v2(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K)
{
    int lane = threadIdx.x;

    int group = lane / 4;
    int tid   = lane % 4;

    // 当前 CTA 负责的 C Tile 左上角
    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 8;

    // FP32 Accumulator
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for(int k0 = 0; k0 < K; k0 += 32){

        //A要4个32bit的寄存器，B要2个
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        // 根据 Lane/Fragment 映射，使用 32-bit Global Load 直接加载四个连续 FP8 原始编码
        //A
        #pragma unroll
        for(int r = 0; r < 4; r++){
            //计算第一个数据的位置
            int row = group + 8 * (r % 2);
            int col = 4 * tid + 16 * (r / 2);

            // 转换为 Global Memory 索引
            int global_row = m0 + row;
            int global_col = k0 + col;

            int index = global_row * K + global_col;

            //声明一个 32-bit 变量，用来保存四个 FP8 的原始编码。
            uint32_t packed;

            //计算我们希望读取的第一个 FP8 元素的地址
            const __nv_fp8_e4m3* ptr = A + index;

            asm volatile(
                "ld.global.b32 %0, [%1];"
                : "=r"(packed)
                : "l"(ptr)//"l" 表示64-bit 寄存器输入，因为我的RTX 5070 使用 64-bit 设备指针
                : "memory"//"memory" 是编译器层面的 Memory Clobber，告诉编译器这段汇编存在内存访问，限制周围的内存优化或重排
            );

            //将四个 FP8 打包到一个 32-bit 寄存器
            a_frag[r] = packed;
        }

        //B
        #pragma unroll
        for(int r = 0; r < 2; r++){
            int row = 4 * tid + 16 * r;
            int col = group;

            // 转换为 Global Memory 索引
            int global_row = k0 + row;
            int global_col = n0 + col;

            int index = global_col * K + global_row;

            uint32_t packed;

            const __nv_fp8_e4m3* ptr = B + index;

            asm volatile(
                "ld.global.b32 %0, [%1];"
                : "=r"(packed)
                : "l"(ptr)
                : "memory"
            );

            //将四个 FP8 打包到一个 32-bit 寄存器
            b_frag[r] = packed;
        }


        // 使用 mma.sync.m16n8k32，更新 acc[0..3]
        //使用next数组存每次mma计算得到的结果
        float next[4];

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"

            // 四个 FP32 输出
            : "=f"(next[0]),
            "=f"(next[1]),
            "=f"(next[2]),
            "=f"(next[3])

            // 填入输入操作数
            //A的4个寄存器
            :"r"(a_frag[0]), 
            "r"(a_frag[1]),
            "r"(a_frag[2]),
            "r"(a_frag[3]),

            //B的2个寄存器
            "r"(b_frag[0]), 
            "r"(b_frag[1]),

            //4个累加
            "f"(acc[0]),
            "f"(acc[1]),
            "f"(acc[2]),
            "f"(acc[3])
        );

        // 更新累加器
        acc[0] = next[0];
        acc[1] = next[1];
        acc[2] = next[2];
        acc[3] = next[3];       
    }


    // 根据 C Fragment 映射，将 acc[0..3] 写回 C
    #pragma unroll
    for(int i = 0; i < 4; i++){
        int row = group + 8 * (i / 2);
        int col = 2 * tid + 1 * (i % 2);

        int global_row = row + m0;
        int global_col = col + n0;

        int index = global_row * N + global_col;

        C[index] = acc[i];
    }
}



//block内warp增加，现在一个block会有2 * 2一共4个warp
__global__ void fp8_gemm_v3(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int group = lane / 4;
    int tid   = lane % 4;

    int warp_m = warp_id / 2;//因为warp的排布形状是2 * 2，一行2个warp
    int warp_n = warp_id % 2;

    // 计算当前 Warp 的输出 Tile 起点
    int m0 = blockIdx.y * 32 + warp_m * 16;
    int n0 = blockIdx.x * 16 + warp_n * 8;

    // FP32 Accumulator
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for(int k0 = 0; k0 < K; k0 += 32){

        //A要4个32bit的寄存器，B要2个
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        // 根据 Lane/Fragment 映射，使用 32-bit Global Load 直接加载四个连续 FP8 原始编码
        //A
        #pragma unroll
        for(int r = 0; r < 4; r++){
            //计算第一个数据的位置
            int row = group + 8 * (r % 2);
            int col = 4 * tid + 16 * (r / 2);

            // 转换为 Global Memory 索引
            int global_row = m0 + row;
            int global_col = k0 + col;

            int index = global_row * K + global_col;

            //声明一个 32-bit 变量，用来保存四个 FP8 的原始编码。
            uint32_t packed;

            //计算我们希望读取的第一个 FP8 元素的地址
            const __nv_fp8_e4m3* ptr = A + index;

            asm volatile(
                "ld.global.b32 %0, [%1];"
                : "=r"(packed)
                : "l"(ptr)//"l" 表示64-bit 寄存器输入，因为我的RTX 5070 使用 64-bit 设备指针
                : "memory"//"memory" 是编译器层面的 Memory Clobber，告诉编译器这段汇编存在内存访问，限制周围的内存优化或重排
            );

            //加载已经获得四个 FP8 的原始编码，无需手工 Packing
            a_frag[r] = packed;
        }

        //B
        #pragma unroll
        for(int r = 0; r < 2; r++){
            int row = 4 * tid + 16 * r;
            int col = group;

            // 转换为 Global Memory 索引
            int global_row = k0 + row;
            int global_col = n0 + col;

            int index = global_col * K + global_row;

            uint32_t packed;

            const __nv_fp8_e4m3* ptr = B + index;

            asm volatile(
                "ld.global.b32 %0, [%1];"
                : "=r"(packed)
                : "l"(ptr)
                : "memory"
            );

            //将四个 FP8 打包到一个 32-bit 寄存器
            b_frag[r] = packed;
        }


        // 使用 mma.sync.m16n8k32，更新 acc[0..3]
        //使用next数组存每次mma计算得到的结果
        float next[4];

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"

            // 四个 FP32 输出
            : "=f"(next[0]),
            "=f"(next[1]),
            "=f"(next[2]),
            "=f"(next[3])

            // 填入输入操作数
            //A的4个寄存器
            :"r"(a_frag[0]), 
            "r"(a_frag[1]),
            "r"(a_frag[2]),
            "r"(a_frag[3]),

            //B的2个寄存器
            "r"(b_frag[0]), 
            "r"(b_frag[1]),

            //4个累加
            "f"(acc[0]),
            "f"(acc[1]),
            "f"(acc[2]),
            "f"(acc[3])
        );

        // 更新累加器
        acc[0] = next[0];
        acc[1] = next[1];
        acc[2] = next[2];
        acc[3] = next[3];       
    }


    // 根据 C Fragment 映射，将 acc[0..3] 写回 C
    #pragma unroll
    for(int i = 0; i < 4; i++){
        int row = group + 8 * (i / 2);
        int col = 2 * tid + 1 * (i % 2);

        int global_row = row + m0;
        int global_col = col + n0;

        int index = global_row * N + global_col;

        C[index] = acc[i];
    }
}



// FP8 GEMM V4
// 4-Warp CTA + Shared Memory Data Reuse
// A: M × K, Row-major
// B: K × N, Column-major
// C: M × N, Row-major
//
// CTA Tile: 32 × 16
// Warp Tile: 16 × 8
// K Tile: 32
//
// 要求：M % 32 == 0, N % 16 == 0, K % 32 == 0

__global__ void fp8_gemm_v4(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K)
{
    // Thread / Warp Mapping
    int t = threadIdx.x;

    int warp_id = t / 32;
    int lane = t % 32;

    int group = lane / 4;
    int tid = lane % 4;

    // 2 × 2 Warp Layout
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    // CTA Tile 起点
    int cta_m = blockIdx.y * 32;
    int cta_n = blockIdx.x * 16;

    // 当前 Warp 的输出 Tile 起点
    int m0 = cta_m + warp_m * 16;
    int n0 = cta_n + warp_n * 8;

    //Shared Memory

    // 每个 uint32_t 保存 4 个 FP8
    // A Tile: 32 × 32 FP8 = 32 × 8 Words
    // B Tile: 32 × 16 FP8 = 16 × 8 Words
    //
    // B 使用 Column-major 的物理布局

    __shared__ uint32_t As_words[32][8];
    __shared__ uint32_t Bs_words[16][8];

    // FP32 Accumulator
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    //  K Tile Loop
    for (int k0 = 0; k0 < K; k0 += 32) {
        //  Global → Shared : A
        // A 共 256 Words
        // 128 Threads，每个线程加载 2 Words
        #pragma unroll
        for (int i = 0; i < 2; i++) {

            int word = t + i * 128;

            int row = word / 8;
            int word_k = word % 8;

            int global_row = cta_m + row;
            int global_col = k0 + word_k * 4;

            int index = global_row * K + global_col;

            uint32_t packed;
            const __nv_fp8_e4m3* ptr = A + index;

            asm volatile(
                "ld.global.b32 %0, [%1];"
                : "=r"(packed)
                : "l"(ptr)
                : "memory"
            );

            As_words[row][word_k] = packed;
        }

        // Global → Shared : B
        // B 共 128 Words
        // 每个线程加载 1 Word
        int b_word = t;

        int b_col = b_word / 8;
        int b_word_k = b_word % 8;

        int global_col = cta_n + b_col;
        int global_row = k0 + b_word_k * 4;

        // B 为 Column-major
        int b_index = global_col * K + global_row;

        uint32_t b_packed;
        const __nv_fp8_e4m3* b_ptr = B + b_index;

        asm volatile(
            "ld.global.b32 %0, [%1];"
            : "=r"(b_packed)
            : "l"(b_ptr)
            : "memory"
        );

        Bs_words[b_col][b_word_k] = b_packed;

        // 确保整个 CTA 完成 Global → Shared
        __syncthreads();

        // Shared → Register : A Fragment
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        #pragma unroll
        for (int r = 0; r < 4; r++) {

            // Warp 内原有 MMA Fragment 映射
            int row = group + 8 * (r % 2);
            int col = 4 * tid + 16 * (r / 2);

            // Warp 局部坐标 → CTA Shared 坐标
            int shared_row = warp_m * 16 + row;
            int shared_word_k = col / 4;

            a_frag[r] =
                As_words[shared_row][shared_word_k];
        }

        //Shared → Register : B Fragment
        #pragma unroll
        for (int r = 0; r < 2; r++) {

            int row = 4 * tid + 16 * r;
            int col = group;

            int shared_col = warp_n * 8 + col;
            int shared_word_k = row / 4;

            b_frag[r] =
                Bs_words[shared_col][shared_word_k];
        }

        // FP8 Tensor Core MMA
        float next[4];

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"

            : "=f"(next[0]),
              "=f"(next[1]),
              "=f"(next[2]),
              "=f"(next[3])

            : "r"(a_frag[0]),
              "r"(a_frag[1]),
              "r"(a_frag[2]),
              "r"(a_frag[3]),

              "r"(b_frag[0]),
              "r"(b_frag[1]),

              "f"(acc[0]),
              "f"(acc[1]),
              "f"(acc[2]),
              "f"(acc[3])
        );

        acc[0] = next[0];
        acc[1] = next[1];
        acc[2] = next[2];
        acc[3] = next[3];

        // 确保所有 Warp 已完成当前 Shared Tile 的读取，
        // 再进入下一轮，覆盖同一块 Shared Memory
        __syncthreads();
    }

    // Register → Global : C
    #pragma unroll
    for (int i = 0; i < 4; i++) {

        int row = group + 8 * (i / 2);
        int col = 2 * tid + (i % 2);

        int global_row = m0 + row;
        int global_col = n0 + col;

        int index = global_row * N + global_col;

        C[index] = acc[i];
    }
}



__global__ void fp8_gemm_v5(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K)
{
    //Thread / Warp Mapping
    int t = threadIdx.x;

    int warp_id = t / 32;
    int lane = t % 32;

    int group = lane / 4;
    int tid = lane % 4;

    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    int cta_m = blockIdx.y * 32;
    int cta_n = blockIdx.x * 16;

    int m0 = cta_m + warp_m * 16;
    int n0 = cta_n + warp_n * 8;

    //Shared Memory
    //每个 uint32_t 存储 4 个 FP8
    __shared__ uint32_t As_words[32][8];
    __shared__ uint32_t Bs_words[16][8];

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    // K Tile Loop
    for (int k0 = 0; k0 < K; k0 += 32) {

        //Global -> Shared : A
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int word = t + i * 128;

            int row = word / 8;
            int word_k = word % 8;

            int global_row = cta_m + row;
            int global_col = k0 + word_k * 4;

            int index = global_row * K + global_col;

            uint32_t packed;

            const __nv_fp8_e4m3* ptr = A + index;

            asm volatile(
                "ld.global.b32 %0, [%1];"
                : "=r"(packed)
                : "l"(ptr)
                : "memory"
            );

            // V5: XOR Swizzling
            int physical_word_k = word_k ^ (row & 4);

            As_words[row][physical_word_k] = packed;
        }

        //Global -> Shared : B
        int b_word = t;

        int b_col = b_word / 8;
        int b_word_k = b_word % 8;

        int global_col = cta_n + b_col;
        int global_row = k0 + b_word_k * 4;

        int b_index = global_col * K + global_row;

        uint32_t b_packed;

        const __nv_fp8_e4m3* b_ptr = B + b_index;

        asm volatile(
            "ld.global.b32 %0, [%1];"
            : "=r"(b_packed)
            : "l"(b_ptr)
            : "memory"
        );

        // V5: XOR Swizzling
        int b_physical_word_k = b_word_k ^ (b_col & 4);

        Bs_words[b_col][b_physical_word_k] = b_packed;

        // Global -> Shared 写入完成
        __syncthreads();

        // Shared -> Register : A Fragment
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = group + 8 * (r % 2);
            int col = 4 * tid + 16 * (r / 2);

            int shared_row = warp_m * 16 + row;
            int shared_word_k = col / 4;

            // V5: 使用相同 XOR 映射恢复原始 Fragment
            int physical_word_k =
                shared_word_k ^ (shared_row & 4);

            a_frag[r] =
                As_words[shared_row][physical_word_k];
        }

        //Shared -> Register : B Fragment
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int row = 4 * tid + 16 * r;
            int col = group;

            int shared_col = warp_n * 8 + col;
            int shared_word_k = row / 4;

            // V5: 使用相同 XOR 映射恢复原始 Fragment
            int physical_word_k =
                shared_word_k ^ (shared_col & 4);

            b_frag[r] =
                Bs_words[shared_col][physical_word_k];
        }

        // FP8 Tensor Core MMA
        float next[4];

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"

            : "=f"(next[0]),
              "=f"(next[1]),
              "=f"(next[2]),
              "=f"(next[3])

            : "r"(a_frag[0]),
              "r"(a_frag[1]),
              "r"(a_frag[2]),
              "r"(a_frag[3]),

              "r"(b_frag[0]),
              "r"(b_frag[1]),

              "f"(acc[0]),
              "f"(acc[1]),
              "f"(acc[2]),
              "f"(acc[3])
        );

        acc[0] = next[0];
        acc[1] = next[1];
        acc[2] = next[2];
        acc[3] = next[3];

        // Single Buffer：保护下一轮覆盖
        __syncthreads();
    }

    //Register -> Global : C
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = group + 8 * (i / 2);
        int col = 2 * tid + (i % 2);

        int global_row = m0 + row;
        int global_col = n0 + col;

        int index = global_row * N + global_col;

        C[index] = acc[i];
    }
}

