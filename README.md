# CUDA GEMM Optimization Lab

这是一个 CUDA GEMM 学习与性能优化项目。项目保留了三条独立的优化路线：FP32 CUDA Core、FP16 WMMA Tensor Core，以及使用 E4M3 输入和 FP32 累加的 FP8 Tensor Core。各路线都保留中间版本，用于观察单一工程改动对正确性、指令和性能的影响。

项目不只展示最终 TFLOPS，也保留中间版本，并通过正确性测试、CUDA Event、SASS 和 Nsight Compute 建立从代码改动到性能结果的证据链。

> FP32/FP16 路线的完整实验方法、性能数据和 profiler 分析见 [CUDA GEMM 优化最终分析](docs/FINAL_ANALYSIS.md)，WMMA V4 async pipeline 的专项结果见 [V4 实验总结](docs/WMMA_V4_ASYNC_PIPELINE_ANALYSIS.md)。FP8 路线当前已推进至 V5，最新的正确性、SASS、Nsight Compute 与性能证据见 [FP8 V5 验证与性能报告](FP8_V5_VALIDATION_AND_PERFORMANCE.md)。

## 项目亮点

- 在同一代码库中保留 FP32 CUDA Core、FP16 WMMA 和 FP8 Tensor Core 三条路线，便于逐步对照。
- CUDA Core FP32 路线在 `1024³` 上从 2.084 TFLOPS 提升至 14.755 TFLOPS，达到同轮 `cublasSgemm` 的 74.92%。
- FP16 WMMA V2→V2.1 仅调整 B 的 shared-memory stride，使总 shared-load conflict 降低 87.40%。
- FP16 WMMA V3 使用 128-bit Global→Shared staging；在 `2048³` 上达到 56.017 TFLOPS，为同轮默认 cuBLAS 的 95.46%。
- FP16 WMMA V4.1–V4.3 验证了 `cp.async`、双缓冲和循环尾拆分；V4.3 相对 V4.2 将动态 warp 指令减少 37.31%，但尚未稳定超过 V3。
- FP8 V5 使用 XOR Shared-Memory Swizzling，将 V4 的系统性平均 2-way Shared Load Bank Conflict 降低 99.946%；同轮 `1024³` 中位性能由 29.136 提高至 33.301 TFLOPS。
- 使用实际量化输入的 CPU reference、NaN 预填充、guard、finite 检查和多形状固定种子测试验证结果正确性。

## 优化路线

### CUDA Core FP32 SGEMM

实现位于 [`src/gemm.cu`](src/gemm.cu)：

```text
Naive
  → Shared-memory Tiling
  → 1D Thread Tiling
  → 2D Register Blocking
  → float4 Vectorized Access
  → Vectorized Shared/Global Store
  → Shared Double Buffering
  → Register Prefetch
  → Warp Layout / Shared Padding Experiments
```

### Tensor Core WMMA GEMM

实现位于 [`src/tensor_core_gemm.cu`](src/tensor_core_gemm.cu)：

```text
V0  Global → WMMA Fragment
 → V1    32×32 Shared Tiling
 → V1.1  B Shared Padding
 → V2    64×64 CTA Tile + Fragment Reuse
 → V2.1  B Shared Padding
 → V3    128-bit Global→Shared Staging
 → V4.1  cp.async Single Buffer
 → V4.2  cp.async Double Buffering
 → V4.3  Branch-free Steady Loop + Explicit Epilogue
```

Tensor Core 路线使用 FP16 A/B、FP32 accumulation 和 FP32 C；对照基线为相同数值配置下的默认 `cublasGemmEx`。V3 仍是有稳定性能证据支持的基线；V4.3 是更适合继续研究的双缓冲实现，但现有多批 Event 数据不足以声称它稳定更快。详见 [WMMA V4 Async Pipeline 实验总结](docs/WMMA_V4_ASYNC_PIPELINE_ANALYSIS.md)。

### FP8 Tensor Core GEMM

实现位于 [`src/fp8_gemm.cu`](src/fp8_gemm.cu)，声明位于 [`src/fp8_gemm.cuh`](src/fp8_gemm.cuh)：

```text
V1  1 Warp/CTA，16×8 CTA Tile，逐字节加载并手工 Packing
 → V2  使用 32-bit Global Load 替换 4 次 FP8 byte load 与 Packing
 → V3  4 Warps/CTA，2×2 Warp Layout，32×16 CTA Tile
 → V4  使用 1536 B Shared Memory 复用 32×32 A / 32×16 B Tile
 → V5  对 A/B Shared word 应用 XOR Swizzling，消除系统性 2-way Load Conflict
```

FP8 路线使用 E4M3 A/B、FP32 accumulation 和 FP32 C；A 为 Row-major，B 为 Column-major，C 为 Row-major。输入必须已完成 E4M3 量化，kernel 内不执行 scaling 或 quantization。V1/V2 使用 `block(32)`；V3–V5 使用 `block(128)` 和 2×2 Warp Layout。所有版本仍使用 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`。

每个版本只引入当轮目标改动，并通过相邻版本同输入、同进程交替 Benchmark 对照。阶段报告如下：

- [V1 工程集成与验证](FP8_V1_VALIDATION_REPORT.md)
- [V1 性能瓶颈诊断](FP8_V1_PERFORMANCE_DIAGNOSIS.md)
- [V2 宽加载验证与性能对照](FP8_V2_VALIDATION_AND_PERFORMANCE.md)
- [V3 多 Warp CTA 验证与性能对照](FP8_V3_VALIDATION_AND_PERFORMANCE.md)
- [V4 Shared Memory 复用验证与性能对照](FP8_V4_VALIDATION_AND_PERFORMANCE.md)
- [V5 XOR Swizzling 验证与性能对照](FP8_V5_VALIDATION_AND_PERFORMANCE.md)

## 主要结果

测试环境为 NVIDIA GeForce RTX 5070 12 GB、CUDA 13.2、Nsight Compute 2026.1.1，编译目标为 `sm_120`。

| 路线与形状 | 自写基线 | 最佳自写版本 | 同轮 cuBLAS | 结果 |
|---|---:|---:|---:|---:|
| FP32 CUDA Core，1024³ | Naive：2.084 TFLOPS | V7_2：14.755 TFLOPS | 19.694 TFLOPS | 7.08×；cuBLAS 的 74.92% |
| FP16 WMMA，1024³ | V2.1：18.551 TFLOPS | V3：44.518 TFLOPS | 46.165 TFLOPS | V2.1 的 2.40×；cuBLAS 的 96.43% |
| FP16 WMMA，2048³ | V2.1：25.457 TFLOPS | V3：56.017 TFLOPS | 58.679 TFLOPS | V2.1 的 2.20×；cuBLAS 的 95.46% |
| FP8 Tensor Core，1024³ | V4：29.136 TFLOPS | V5：33.301 TFLOPS | 未测试 | 同轮中位数 1.143× |
| FP8 Tensor Core，2048³ | V4：31.330 TFLOPS | V5：34.651 TFLOPS | 未测试 | 同轮中位数 1.106× |

FP32、FP16 WMMA 与 FP8 路线的数据类型、执行单元、布局和测试协议不同，因此不直接横向比较各组 TFLOPS。FP8 表中的 V4/V5 是同一进程、相同输入、交替顺序、7 轮测试的各自中位数；没有进行 cuBLASLt FP8 对照。以上结果只代表当前软硬件环境和已测试形状；详细统计口径分别见 [FP32/FP16 最终分析](docs/FINAL_ANALYSIS.md) 和 [FP8 V5 报告](FP8_V5_VALIDATION_AND_PERFORMANCE.md)。

## 正确性验证

- FP32 主测试相对 CPU reference 的最大绝对误差低于程序使用的 `1e-5` 阈值，并覆盖非整 tile 形状。
- Tensor Core 使用 FP16 量化后的输入计算 FP32 CPU reference，V2.1、V3 与 V4.1–V4.3 的判定条件为：

  ```text
  abs_error <= 1e-2 + 1e-2 * abs(reference)
  ```

- V2.1、V3 和 cuBLAS 通过 5 种矩阵形状、5 个固定随机种子及多类输入分布测试；V4.1–V4.3 另覆盖 `K=16/32/48` 等 pipeline 边界和多个规则形状。
- 每次运行前将输出矩阵填充为 NaN，并检查 reference、输出和误差均为 finite，以发现漏写 tile。
- FP8 Reference 使用实际量化后的 E4M3 A/B 重新扩展为 FP32，再按 A Row-major、B Column-major 约定执行 FP32 FMA；V5 的判定条件为：

  ```text
  abs_error <= 1e-4 + 1e-4 * abs(reference)
  ```

- FP8 V5 覆盖零、稀疏、正负混合、非均匀、固定随机、多 CTA 和最多 10 个 K Tile 等 12 组测试；V5/Reference 与 V5/V4 的最大绝对误差和最大相对误差均为 0，所有 guard 完整。

## 构建与运行

```bash
mkdir -p build

# CUDA Core FP32
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  src/gemm.cu -o build/gemm_fp32 -lcublas
./build/gemm_fp32

# Tensor Core + cuBLAS benchmark
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu \
  -o build/tensor_core_gemm -lcublas
./build/tensor_core_gemm 1024 1024 1024
./build/tensor_core_gemm 2048 2048 2048

# Tensor Core random correctness test
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  tests/v3_random_correctness.cu \
  -o build/v3_random_correctness -lcublas
./build/v3_random_correctness

# 推荐：使用 CMake/Ninja 构建全部路线
cmake -S . -B build/cmake-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-sm120 -j 4

# FP8 V1 冻结基线
./build/cmake-sm120/fp8_gemm_v1_test --correctness
./build/cmake-sm120/fp8_gemm_v1_test --benchmark

# FP8 最新相邻版本 V4/V5 验证与交替 Benchmark
./build/cmake-sm120/fp8_gemm_v5_compare --correctness
./build/cmake-sm120/fp8_gemm_v5_compare --benchmark
```

也可以只构建 FP8 目标：

```bash
cmake --build build/cmake-sm120 --target \
  fp8_gemm_v1_test fp8_gemm_v2_compare fp8_gemm_v3_compare \
  fp8_gemm_v4_compare fp8_gemm_v5_compare -j 4
```

## 仓库结构

- [`src/gemm.cu`](src/gemm.cu)：CUDA Core FP32 SGEMM 各优化版本及 cuBLAS 对照。
- [`src/tensor_core_gemm.cu`](src/tensor_core_gemm.cu)：WMMA V0–V4.3、正确性检查和 cuBLAS benchmark。
- [`src/fp8_gemm.cu`](src/fp8_gemm.cu)：FP8 E4M3 Tensor Core GEMM V1–V5 kernels。
- [`src/fp8_gemm.cuh`](src/fp8_gemm.cuh)：FP8 V1–V5 kernel 声明。
- [`tests/fp8_gemm_v1_test.cu`](tests/fp8_gemm_v1_test.cu)：FP8 V1 独立正确性与 CUDA Event 冻结基线。
- [`tests/fp8_gemm_v2_compare.cu`](tests/fp8_gemm_v2_compare.cu)：FP8 V1/V2 同输入正确性、交替 Benchmark 与 NCU workload。
- [`tests/fp8_gemm_v3_compare.cu`](tests/fp8_gemm_v3_compare.cu)：FP8 V2/V3 对照与 V1 计时审计。
- [`tests/fp8_gemm_v4_compare.cu`](tests/fp8_gemm_v4_compare.cu)：FP8 V3/V4 Shared Memory 对照。
- [`tests/fp8_gemm_v5_compare.cu`](tests/fp8_gemm_v5_compare.cu)：FP8 V4/V5 XOR Swizzling 对照。
- [`tests/v3_random_correctness.cu`](tests/v3_random_correctness.cu)：V2.1、V3 和 cuBLAS 的多形状随机正确性测试。
- [`docs/FINAL_ANALYSIS.md`](docs/FINAL_ANALYSIS.md)：完整性能结果、实验方法、Profiler 证据和限制说明。
- [`docs/WMMA_V4_ASYNC_PIPELINE_ANALYSIS.md`](docs/WMMA_V4_ASYNC_PIPELINE_ANALYSIS.md)：WMMA V4.1–V4.3 async-copy、双缓冲与动态指令分析。
- [`docs/archive/wmma/`](docs/archive/wmma/)：WMMA 各阶段原始实验记录。
- [`FP8_V5_VALIDATION_AND_PERFORMANCE.md`](FP8_V5_VALIDATION_AND_PERFORMANCE.md)：最新 FP8 正确性、PTX/SASS、NCU 和性能报告。

## 已知限制

- FP16 WMMA V3–V4.3 要求 M/N 为 64 的倍数、K 为 16 的倍数，不满足条件时不会执行通用边界路径。
- FP8 V1/V2 要求 M/N/K 分别为 16/8/32 的整数倍；V3–V5 要求分别为 32/16/32 的整数倍，均无通用边界路径。
- FP8 固定使用 A Row-major、B Column-major、C Row-major，并要求 Host 端提前完成 E4M3 量化和 scaling 策略；kernel 不接受 arbitrary leading dimension。
- 当前未实现 batched GEMM、split-K 或融合 epilogue。
- FP16 WMMA 的固定 64×64 CTA tile 只完成了方阵性能验证，不能据此声称任意形状都接近 cuBLAS。
- FP32/FP16 的 cuBLAS 对照使用默认接口，没有搜索 cuBLASLt heuristic；FP8 路线尚未进行 cuBLASLt 对照。

## 贡献说明

GEMM kernel 的核心学习与实现由项目作者完成；Host correctness/benchmark 集成、Nsight Compute 数据整理和报告撰写有 Codex 辅助。项目没有虚构生产部署、团队协作或上游提交经历。
