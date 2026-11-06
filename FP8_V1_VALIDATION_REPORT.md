# FP8 Tensor Core GEMM V1 工程集成与验证报告

验证日期：2026-11-03（Asia/Shanghai）

## 1. 结论

`fp8_gemm_v1()` 已作为独立路径完成工程集成、编译、正确性、指令和基础性能
验证。用户实现的 fragment 加载、FP8 packing、`mma.sync` 和写回逻辑均未修改；
源码只把原来的 `#include <cuda_fp8.h>` 换成了包含同一 CUDA 头文件的声明头。

- 指定的 6 个形状以及额外的零、稀疏、distinct-value 单 tile 案例全部通过。
- 全部输出与基于实际 E4M3 数值的 FP32 reference 逐元素完全一致；全局最大绝对
  误差和最大相对误差均为 0。
- PTX 包含 FP8 `mma.sync.aligned.m16n8k32`。
- SASS 包含 `QMMA.16832.F32.E4M3.E4M3`，确认实际生成 Tensor Core 指令。
- CUDA Event 基础结果约为 0.94--1.10 TFLOPS；该 V1 以正确性为目标，未做优化。

## 2. Git、项目与环境

验证前 Git 分支为 `main`，相对 `origin/main` ahead 3。工作区已有多份未跟踪实验
报告和未跟踪的 `src/fp8_gemm.cu`；验证过程没有删除、覆盖或提交这些内容。

项目原先没有 CMake 配置，README 中记录的构建方式为直接调用 `nvcc`。本次新增
顶层 CMake，并把原有 FP32、FP16 Tensor Core、FP16 随机测试和新 FP8 测试都定义
成独立 target。

| 项目 | 结果 |
|---|---|
| OS/运行方式 | WSL Ubuntu |
| GPU | NVIDIA GeForce RTX 5070，compute capability 12.0 |
| GPU global memory | 12,792,365,056 bytes |
| CUDA Toolkit | 13.2，nvcc 13.2.86 |
| PTX ISA | 9.2 |
| 编译目标 | `sm_120` |
| CMake / Ninja | 3.22.1 / 1.10.1 |
| 分析工具 | `cuobjdump`、`nvdisasm`、Nsight Compute 2026.1.1 |

`nvcc --list-gpu-code` 明确列出 `sm_120`。实际 ptxas 编译成功，因此当前 Toolkit
支持该目标。`nvidia-smi` 在受限会话中无法初始化 NVML，但在沙箱外 CUDA Runtime
成功识别 GPU；环境判断以实际编译和运行结果为准。

## 3. Kernel 审计

核对依据为 NVIDIA PTX ISA 9.2 的 `mma.m16n8k32` fragment 定义。

### A fragment

每个 lane 持有 4 个 `.b32`，每个寄存器按低字节到高字节装 4 个 E4M3 元素。
当前代码的第 `r` 个寄存器使用：

```text
row = group + 8 * (r % 2)
col = 4 * tid + 16 * (r / 2) + byte_in_register
```

这与 PTX 中 a0..a15 的 lane/register 映射一致。

### B fragment

每个 lane 持有 2 个 `.b32`。当前代码的第 `r` 个寄存器使用：

```text
row = 4 * tid + 16 * r + byte_in_register
col = group
```

这与 `.row.col` 指令的 column-major B fragment 映射一致；Host 和 reference 均按
`B[n * K + k]` 存取 K×N column-major B。

### C/D fragment

每个 lane 持有 4 个 FP32 accumulator：

```text
row = group + 8 * (i / 2)
col = 2 * tid + (i % 2)
```

与当前写回映射一致，32 lanes 恰好覆盖 16×8 共 128 个 C 元素。

### Packing 与 inline PTX

- `x0 | x1<<8 | x2<<16 | x3<<24` 使 fragment 的第一个元素位于 `.b32` 低字节，
  顺序正确。
- A/B 使用 32-bit `"r"` constraint，C/D 使用 FP32 `"f"` constraint，类型和
  宽度正确。
- D 使用独立 `"=f"` 输出、C 使用 `"f"` 输入是合法的；生成 PTX 时 ptxas 将同一
  accumulator 的输入/输出安全合并到了同组寄存器。
- `block(32)` 下全 warp 一致执行 `mma.sync`。V1 不支持边界 tile，因此没有增加
  会导致 warp divergence 的边界分支。

审计未发现需要修改核心计算逻辑的问题。

## 4. 新增和修改的文件

| 文件 | 变更 |
|---|---|
| `src/fp8_gemm.cu` | 仅改为包含声明头；Kernel 计算逻辑不变 |
| `src/fp8_gemm.cuh` | 新增 V1 声明和 layout/shape/launch contract |
| `tests/fp8_gemm_v1_test.cu` | 新增 FP8 Host 准备、reference、正确性和 benchmark |
| `CMakeLists.txt` | 新增，包含原有和 FP8 独立 target |
| `README.md` | 增加 FP8 构建、运行和文件入口 |
| `FP8_V1_VALIDATION_REPORT.md` | 本报告 |

没有执行 Git commit。

## 5. 构建

直接 nvcc 命令：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills -Isrc \
  tests/fp8_gemm_v1_test.cu src/fp8_gemm.cu \
  -o build/fp8_gemm_v1_test
```

结果：成功。`fp8_gemm_v1` 使用 39 registers、0-byte stack frame、0 spill、
0 barrier。

CMake 全项目构建：

```bash
cmake -S . -B build/cmake-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-sm120 -j 4
```

结果：`gemm_fp32`、`tensor_core_gemm`、`v3_random_correctness`、
`fp8_gemm_v1_test` 全部构建成功。原文件中已有少量 unused-variable warning，
没有新增编译错误。

## 6. Reference、容差与内存检查

Host 先把输入 float 构造成 `__nv_fp8_e4m3`（CUDA 默认 finite saturation，测试值
均在 [-1, 1] 内），再把实际编码转回 float。Reference 使用：

```text
C[m,n] = FP32-FMA sequence over k of
         float(A_e4m3[m,k]) * float(B_e4m3[n,k])
```

即 reference 与 Kernel 使用完全相同的量化后输入，而不是原始 FP32 输入。

预先设定的判据为：

```text
abs_error <= 1e-4 + 1e-4 * abs(reference)
```

理由：乘数是只有 3 个显式 mantissa bits 的 E4M3，单次乘积可由 FP32 精确表达；
差异只可能来自 FP32 累加结合顺序。对 K≤512、输入幅值≤1 的测试，这一阈值足以
容纳少量 FP32 rounding，又远小于交换 fragment 或错误 K tile 通常产生的误差。
实际所有结果误差为 0，因此通过不依赖该容差的宽松程度。

每次 launch 都执行 `cudaGetLastError()` 和 `cudaDeviceSynchronize()`；C 先填 NaN，
前后各放 64 个 float guard。全部测试 `nonfinite=0` 且 guard 完整，证明所有预期
输出被覆盖且未发现 C 邻接越界写。

尝试运行 Compute Sanitizer memcheck，但当前 WSL/WDDM 环境报告：

```text
Failed to initialize WDDM debugger interface.
Device not supported.
```

因此不能声称 Compute Sanitizer 通过。测试程序仍继续运行并通过，但 sanitizer
给出 2 个工具初始化错误。输入下标也由 fragment 公式审计确认在约束形状内。

## 7. 正确性结果

命令：

```bash
./build/fp8_gemm_v1_test --correctness
```

| Case | M×N×K | 数据 | max abs | max rel | 超容差 | nonfinite | guard | 结果 |
|---|---:|---|---:|---:|---:|---:|---|---|
| zero_single_tile | 16×8×32 | 全零 | 0 | 0 | 0 | 0 | OK | PASS |
| sparse_single_tile | 16×8×32 | 单点稀疏/正负 | 0 | 0 | 0 | 0 | OK | PASS |
| distinct_single_tile | 16×8×32 | 按位置变化 | 0 | 0 | 0 | 0 | OK | PASS |
| mixed_three_k_tiles | 16×8×96 | 正负混合/零 | 0 | 0 | 0 | 0 | OK | PASS |
| random_multi_cta | 32×16×64 | seed 3001 | 0 | 0 | 0 | 0 | OK | PASS |
| random_64 | 64×64×128 | seed 4001 | 0 | 0 | 0 | 0 | OK | PASS |
| random_128 | 128×128×256 | seed 5001 | 0 | 0 | 0 | 0 | OK | PASS |
| random_256 | 256×256×512 | seed 6001 | 0 | 0 | 0 | 0 | OK | PASS |

全局 `CORRECTNESS_RESULT=PASS`。没有失败案例需要进行 lane/register 修复。测试程序
已包含失败定位：会输出最差 C 坐标、预期/实际值以及对应 lane、accumulator 和
K-tile 数量。

为确认没有破坏已有路径，CMake 构建后的 `v3_random_correctness` 也已运行；原有
FP16 V2.1、V3 和 cuBLAS 在 5 个随机案例中全部 PASS，进程返回 0。

## 8. FP8 Tensor Core 指令验证

PTX 生成命令：

```bash
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_probe/fp8_gemm.ptx
```

关键 PTX：

```text
.version 9.2
.target sm_120
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

SASS 检查命令：

```bash
cuobjdump --dump-sass build/fp8_gemm_v1_test
```

关键 SASS：

```text
QMMA.16832.F32.E4M3.E4M3 R4, R8, R20, R4
```

Nsight Compute 在 WSL 中也成功连接并对一个 16×8×32 launch 完成 9-pass basic
采集，确认 Device 0 / CC 12.0、block size 32、39 registers/thread。该 profile 是
单 block 零输入工具验证，报告的 1.73 us 和低 achieved occupancy 不作为性能结果。

## 9. CUDA Event 基础性能

命令：

```bash
./build/fp8_gemm_v1_test --benchmark
```

数据均来自 CUDA Event；Host 输入准备、H2D 和预热不计时。TFLOPS 按
`2*M*N*K/time` 计算。

| 形状 | 预热 | 计时迭代 | 平均延迟 | TFLOPS |
|---:|---:|---:|---:|---:|
| 512³ | 10 | 100 | 0.243956 ms | 1.100343 |
| 1024³ | 10 | 50 | 2.296136 ms | 0.935260 |
| 2048³ | 5 | 20 | 18.061241 ms | 0.951201 |

未运行 cuBLASLt 对照，也没有把 Nsight Compute replay 时间混入上述结果。

## 10. V1 局限与下一阶段瓶颈

当前 V1 的明确限制：

- 仅支持 A row-major、B column-major、C row-major，且都是紧密连续存储。
- M/N/K 必须分别为 16/8/32 的倍数；没有边界 tile。
- 输入必须由调用方提前量化为 E4M3；没有 scale、zero point 或融合 epilogue。
- 一个 CTA 只有一个 warp，只产出 16×8，CTA 粒度很小。
- A/B 从 global memory 直接按 fragment 读取，没有 CTA/warp 间数据复用。
- 没有 arbitrary leading dimension、batch、split-K 或其他通用接口。
- Compute Sanitizer 的 WSL/WDDM memcheck 在当前机器上不可用。

下一阶段最值得关注的瓶颈是 global-memory 数据复用：同一 A tile 会被不同 N tile
重复加载，同一 B tile会被不同 M tile 重复加载；小 CTA tile 还带来较多调度开销。
这与约 1 TFLOPS 的基础结果一致。按本阶段限制，本次没有增加 shared memory、扩大
tiling、`cp.async`、double buffering 或其他性能优化。

## 11. 官方依据

- NVIDIA PTX ISA 9.2：<https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html>
- NVIDIA CUDA Math API FP8：<https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html>
