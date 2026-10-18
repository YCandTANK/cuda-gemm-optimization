# CUDA GEMM Optimization Lab

这是一个 CUDA GEMM 学习与性能优化项目。项目保留了两条独立的优化路线：一条从朴素 FP32 CUDA Core kernel 逐步演进到分块、寄存器复用和向量化；另一条使用 WMMA/Tensor Core，围绕 shared-memory layout、fragment reuse 和向量化数据搬运进行优化。

项目不只展示最终 TFLOPS，也保留中间版本，并通过正确性测试、CUDA Event、SASS 和 Nsight Compute 建立从代码改动到性能结果的证据链。

> 完整的实验方法、性能数据和 profiler 分析见 [CUDA GEMM 优化最终分析](docs/FINAL_ANALYSIS.md)。

## 项目亮点

- 在同一代码库中保留 CUDA Core 和 Tensor Core 两条优化路线，便于逐步对照。
- CUDA Core FP32 路线在 `1024³` 上从 2.084 TFLOPS 提升至 14.755 TFLOPS，达到同轮 `cublasSgemm` 的 74.92%。
- Tensor Core V2→V2.1 仅调整 B 的 shared-memory stride，使总 shared-load conflict 降低 87.40%。
- Tensor Core V3 使用 128-bit Global→Shared staging；在 `2048³` 上达到 56.017 TFLOPS，为同轮默认 cuBLAS 的 95.46%。
- 使用 CPU reference、NaN 预填充、finite 检查和多形状随机测试验证结果正确性。

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
```

Tensor Core 路线使用 FP16 A/B、FP32 accumulation 和 FP32 C；对照基线为相同数值配置下的默认 `cublasGemmEx`。

## 主要结果

测试环境为 NVIDIA GeForce RTX 5070 12 GB、CUDA 13.2、Nsight Compute 2026.1.1，编译目标为 `sm_120`。

| 路线与形状 | 自写基线 | 最佳自写版本 | 同轮 cuBLAS | 结果 |
|---|---:|---:|---:|---:|
| FP32 CUDA Core，1024³ | Naive：2.084 TFLOPS | V7_2：14.755 TFLOPS | 19.694 TFLOPS | 7.08×；cuBLAS 的 74.92% |
| Tensor Core，1024³ | V2.1：18.551 TFLOPS | V3：44.518 TFLOPS | 46.165 TFLOPS | V2.1 的 2.40×；cuBLAS 的 96.43% |
| Tensor Core，2048³ | V2.1：25.457 TFLOPS | V3：56.017 TFLOPS | 58.679 TFLOPS | V2.1 的 2.20×；cuBLAS 的 95.46% |

FP32 与 Tensor Core 路线的数据类型、执行单元和测试协议不同，因此不直接横向比较两组 TFLOPS。以上结果只代表当前软硬件环境和已测试形状；完整统计口径及 Nsight Compute 指标见 [最终分析](docs/FINAL_ANALYSIS.md)。

## 正确性验证

- FP32 主测试相对 CPU reference 的最大绝对误差低于程序使用的 `1e-5` 阈值，并覆盖非整 tile 形状。
- Tensor Core 使用 FP16 量化后的输入计算 FP32 CPU reference，判定条件为：

  ```text
  abs_error <= 1e-2 + 1e-2 * abs(reference)
  ```

- V2.1、V3 和 cuBLAS 通过 5 种矩阵形状、5 个固定随机种子及多类输入分布测试。
- 每次运行前将输出矩阵填充为 NaN，并检查 reference、输出和误差均为 finite，以发现漏写 tile。

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
```

## 仓库结构

- [`src/gemm.cu`](src/gemm.cu)：CUDA Core FP32 SGEMM 各优化版本及 cuBLAS 对照。
- [`src/tensor_core_gemm.cu`](src/tensor_core_gemm.cu)：WMMA V0–V3、正确性检查和 cuBLAS benchmark。
- [`tests/v3_random_correctness.cu`](tests/v3_random_correctness.cu)：V2.1、V3 和 cuBLAS 的多形状随机正确性测试。
- [`docs/FINAL_ANALYSIS.md`](docs/FINAL_ANALYSIS.md)：完整性能结果、实验方法、Profiler 证据和限制说明。

## 已知限制

- V3 要求 M/N 为 64 的倍数、K 为 16 的倍数，不满足条件时不会执行通用边界路径。
- 当前只支持连续 Row-major 普通 GEMM，未实现 arbitrary leading dimension、batched GEMM、split-K 或融合 epilogue。
- 固定 64×64 CTA tile 只完成了方阵性能验证，不能据此声称任意形状都接近 cuBLAS。
- cuBLAS 对照使用默认 `cublasGemmEx`，没有搜索 cuBLASLt heuristic。

## 贡献说明

GEMM kernel 的核心学习与实现由项目作者完成；Host correctness/benchmark 集成、Nsight Compute 数据整理和报告撰写有 Codex 辅助。项目没有虚构生产部署、团队协作或上游提交经历。
