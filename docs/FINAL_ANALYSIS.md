# CUDA GEMM 优化最终分析

本文总结仓库中两条相互独立的 GEMM 优化路线：

1. FP32 输入、FP32 累加和输出的 CUDA Core SGEMM；
2. FP16 输入、FP32 累加和输出的 WMMA Tensor Core GEMM。

两条路线的数据类型、执行单元和测试协议不同，因此本文只在各自路线内部比较版本，不直接用两组 TFLOPS 判断哪种实现“更优”。文中的性能结论仅适用于下述软硬件环境和已测试矩阵形状。

## 测试环境与证据口径

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 12 GB，48 SM |
| Compute Capability | 12.0 |
| Driver | 596.49 |
| CUDA Toolkit / nvcc | 13.2 / 13.2.86 |
| CUDA runtime / driver API | 13020 / 13020 |
| cuBLAS runtime | 13.4.1 |
| Nsight Compute | 2026.1.1.0 |
| 编译目标 | `sm_120` |

本文区分四类信息：

- **代码事实**：可以直接从 [`src/gemm.cu`](../src/gemm.cu) 和 [`src/tensor_core_gemm.cu`](../src/tensor_core_gemm.cu) 核对的实现与资源配置；
- **普通实测**：CUDA Event 记录的 kernel 延迟与据此计算的吞吐；
- **Profiler 实测**：Nsight Compute 的指令、wavefront、stall 和 occupancy 指标；
- **解释**：根据代码和指标提出的原因分析，不把未测量的 cuBLAS 内部实现当作事实。

---

## 第一部分：CUDA Core FP32 SGEMM

### 1. 优化路径

CUDA Core 路线在同一文件中保留了从直接实现到多级分块的多个版本，便于逐步对照。

| 输出名称 | Kernel | 主要变化 |
|---|---|---|
| Naive | `gemmNaive` | 每线程计算一个 C 元素，A/B 直接从 global memory 读取 |
| Tiled | `gemmTiled` | 16×16 shared-memory tiling |
| Thread Tiled | `gemmThreadTiled` | 一维 register blocking，每线程计算多个相邻列 |
| 2D Thread Tiled | `gemm2DThreadTiled` | 64×64×16 block tile，4×4 thread micro-tile |
| Vectorized | `gemm2DThreadTiledVectorized` | 使用 `float4` 搬运 A/B |
| V5 | `gemm2DThreadTiledVectorizedSharedStore` | 调整 Global→Shared 的向量化 store |
| V6 | `gemm2DThreadTiledVectorizedGlobalStore` | 对 C 使用 `float4` global store |
| V7 | `gemm2DThreadTiledDoubleBuffered` | shared-memory double buffering |
| V7_2 | `gemm2DThreadTiledDoubleBufferedRegisterPrefetch` | 在 V7 上加入 register prefetch |
| V8_1 | `gemmWarpTiledDoubleBufferedV8_1` | 显式 warp tile 布局实验 |
| V8_2 | `gemmWarpTiledDoubleBufferedV8_2` | 在 V8_1 的 A shared layout 上加入 padding |

这条路径的核心是逐级增加数据复用：block tile 复用 global-memory 数据，thread micro-tile 在寄存器中复用 shared-memory 数据，向量化减少搬运指令，double buffering 和 register prefetch 尝试重叠访存与计算。

### 2. 性能结果

测试固定为 `M=N=K=1024`。每个实现预热 10 次，再在 CUDA Event 区间内连续执行 100 次；表中延迟是单次 launch 的平均值。输入初始化、Host/Device 传输和 CPU reference 不计时。

| 版本 | 平均延迟 | 吞吐 | 相对前一版本 |
|---|---:|---:|---:|
| Naive | 1.030340 ms | 2.084 TFLOPS | — |
| Tiled | 0.778897 ms | 2.757 TFLOPS | 1.323× |
| Thread Tiled | 0.670893 ms | 3.201 TFLOPS | 1.161× |
| 2D Thread Tiled | 0.275297 ms | 7.801 TFLOPS | 2.437× |
| Vectorized | 0.166094 ms | 12.929 TFLOPS | 1.657× |
| V5 | 0.157825 ms | 13.607 TFLOPS | 1.052× |
| V6 | 0.157570 ms | 13.629 TFLOPS | 1.002× |
| V7 | 0.149161 ms | 14.397 TFLOPS | 1.056× |
| V7_2 | 0.145540 ms | **14.755 TFLOPS** | 1.025× |
| V8_1 | 0.173926 ms | 12.347 TFLOPS | 0.858×（相对 V7） |
| V8_2 | 0.152652 ms | 14.068 TFLOPS | 1.139×（相对 V8_1） |
| FP32 `cublasSgemm` | 0.109040 ms | 19.694 TFLOPS | — |

在这组测试中，V7_2 是最佳自写版本：相对 Naive 提升约 7.08×，达到同程序 `cublasSgemm` 吞吐的 74.92%。V8_2 修复了 V8_1 的一部分布局代价，但仍慢于 V7_2，说明更复杂的 warp 布局没有在该形状上自动转化为更高性能。

### 3. 性能解释

从结果看，最大两次增益来自二维 register blocking 和 `float4` 向量化。它们分别提高了线程内数据复用和内存事务/指令效率。V5 之后每一步的收益明显缩小，说明瓶颈已经从最初的重复 global load 转移到 shared-memory、指令发射和资源占用等路径。

局部 Nsight Compute 数据与这一变化方向一致：

| 版本 | L1/TEX throughput | Compute throughput | 补充观察 |
|---|---:|---:|---|
| V6 | 74.29% | 49.84% | 向量化后的基础版本 |
| V7 | 88.73% | 54.91% | 理论 occupancy 因资源占用降至 50% |
| V7_2 | 96.28% | 69.94% | MIO throttle 为主要 stall 之一，约 5.3 cycles/issued instruction |

这些 profiler 数据来自 replay，会改变 kernel 的运行条件，所以只用于定位方向，不与普通 CUDA Event 延迟混合排名。它们也不能单独证明某个指标变化就是性能变化的唯一原因。

### 4. 正确性与适用范围

- 1024³ 主测试中，各版本相对 FP32 CPU reference 的最大绝对误差为 `4.76837e-6`，低于程序使用的 `1e-5` 阈值。
- V5、V6、V7 和 V7_2 通过了 `130×129×137` 非整 tile 测试。
- V7、V7_2、V8_1 和 V8_2 还覆盖了 `64×64×16`、`67×69×7`、`67×69×17` 和 `130×129×137` 的控制流测试。

当前 FP32 性能表只来自一个固定方阵和一轮 100 次平均测试，没有 Tensor Core 最终测试所采用的多进程、交替顺序和中位数统计。因此 74.92% 是该次 1024³ 快照，不应外推到其他形状、GPU 或系统状态。

---

## 第二部分：Tensor Core WMMA GEMM

### 1. 数值配置与实现演进

Tensor Core 路线使用 FP16 A/B、FP32 accumulator 和 FP32 C。默认 cuBLAS 基线使用 `cublasGemmEx`、`CUBLAS_COMPUTE_32F` 和 `CUBLAS_GEMM_DEFAULT`；两者没有通过降低累加或输出精度换取性能。

| 版本 | 主要设计 | 直接观测到的问题或收益 |
|---|---|---|
| V0 | 每个 warp 计算 16×16 C tile，global memory 直接加载 WMMA fragment | global-load sector 平均只使用 16/32 B，且多个 warp 重复读取输入 |
| V1 | 32×32 block shared tiling | global-load sectors 降为 V0 的 1/4，但 B fragment load 出现 16 actual / 4 ideal wavefronts |
| V1.1 | 只把 B shared stride 从 32 half 改为 40 | B load 变为 4/4，总 shared-load conflict 降低 73.97% |
| V2 | 64×64 block、32×32 warp tile；A/B fragment 各复用两次 | staging 与 fragment request 减半，但 B stride 64 产生 32/4 wavefronts；寄存器升至 56/thread |
| V2.1 | 只把 V2 的 B stride 从 64 half 改为 72 | 两个 B load 均变为 4/4，总 shared-load conflict 降低 87.40% |
| V3 | 保持 V2.1 的 tile、fragment reuse 和 B padding，改用 128-bit Global→Shared staging | 编译为 `LDG.E.128`/`STS.128`，显著减少搬运和地址指令 |

各阶段的直接对照均在相应实验中重新测量，而不是用不同日期的绝对时间相除。1024³ / 2048³ 的相对结果为：

| 对照 | 1024³ | 2048³ | 结论范围 |
|---|---:|---:|---|
| V1 相对 V0 | +5.36% | +10.90% | shared tiling 减少重复 global load，但引入 shared/LDSM 代价 |
| V1.1 相对 V1 | +9.92% | +11.18% | 单变量 B padding 实验 |
| V2 相对 V1.1 | -2.13% | +18.16% | reuse 有效，但小尺寸受 bank conflict、寄存器和较小 grid 影响 |
| V2.1 相对 V2 | +26.28% | +27.02% | 单变量 B padding 实验 |
| V3 相对 V2.1 | +139.98% | +120.04% | 最终双进程、同轮交替测试 |

### 2. Bank conflict 控制实验

V1→V1.1 和 V2→V2.1 都只改变 B 的 shared-memory 物理 stride，逻辑 tile、MMA 次数和输出映射不变，因此可以较清楚地观察 layout 的影响。

以 V2→V2.1 为例，B 的行跨度从 64 half（128 B）改为 72 half（144 B）。前者使相邻行从相同 bank 相位开始，后者让行起点发生偏移。Nsight Compute 对两条 B `LDSM.16.MT88.4` 的观测从每请求 32 actual / 4 ideal wavefronts 变为 4/4；总 shared-load conflicts 从 4,208,358 降至 530,059，下降 87.40%。同一实验中，1024³ 和 2048³ 的普通 Event 性能分别提高 26.28% 和 27.02%。

这组数据支持“B layout 导致了可测量的 shared-load serialization，并影响端到端性能”。它不表示所有 shared conflict 已消失：A0/A1 的 fragment load 仍为 8 actual / 4 ideal。

### 3. V3 最终性能

最终验证对 1024³ 和 2048³ 各执行两个独立普通进程。每个进程包含五轮测量；每轮对 V2.1、V3 和 cuBLAS 轮转测试顺序，每个实现先预热 10 次，再在 Event 区间内连续执行 100 次。三者复用同一输入、device allocations、默认 stream 和 FP16/FP32 数值配置。表中为每个实现的 10 个样本聚合结果。

| Shape / 实现 | Median 延迟 | Min–Max | CV | Median 吞吐 |
|---|---:|---:|---:|---:|
| 1024³ V2.1 | 0.115761 ms | 0.102810–0.121852 ms | 5.78% | 18.551 TFLOPS |
| 1024³ V3 | 0.048239 ms | 0.045490–0.063731 ms | 13.27% | **44.518 TFLOPS** |
| 1024³ cuBLAS | 0.046518 ms | 0.040822–0.056901 ms | 13.20% | 46.165 TFLOPS |
| 2048³ V2.1 | 0.674858 ms | 0.666701–0.692816 ms | 1.06% | 25.457 TFLOPS |
| 2048³ V3 | 0.306693 ms | 0.298850–0.316034 ms | 1.96% | **56.017 TFLOPS** |
| 2048³ cuBLAS | 0.292776 ms | 0.281222–0.298137 ms | 1.81% | 58.679 TFLOPS |

V3 在 1024³ 和 2048³ 上分别达到同轮默认 cuBLAS 吞吐的 96.43% 和 95.46%。当前数据没有显示 V3 超过 cuBLAS。1024³ 的 V3 与 cuBLAS CV 都约为 13%，短任务对 GPU 升频和系统状态较敏感；2048³ 的 CV 低于 2%，是更稳定的复现点。

### 4. V3 的 profiler 证据

2048³ 定向 Nsight Compute 结果显示：

| 指标 | V2.1 | V3 | 变化 |
|---|---:|---:|---:|
| Global load instructions/requests | 8,388,608 | 1,048,576 | 降低 8× |
| Global load sectors | 16,777,216 | 16,777,216 | 不变 |
| Shared store instructions | 8,388,608 | 1,048,576 | 降低 8× |
| LDSM instructions | 2,097,152 | 2,097,152 | 不变 |
| Total executed instructions | 158,863,360 | 14,270,464 | -91.02% |
| Tensor/HMMA pipe active | 31.22% | 87.53% | +56.31 pp |
| Registers/thread | 56 | 78 | +22 |
| Theoretical occupancy | 75.00% | 50.00% | -25.00 pp |
| Achieved occupancy | 60.91% | 45.93% | -14.98 pp |
| Local load/store | 0 / 0 | 0 / 0 | 无 spill traffic |

相同 global sectors 表明 V3 没有减少必须读取的数据量；主要变化是每条 128-bit 指令承担了原来多条标量 half 搬运的工作。总指令减少且 Tensor pipe active 提高，与普通 Event 中观察到的加速方向一致。

V3 仍有明确代价：78 registers/thread 将理论 occupancy 限制为 50%；A0/A1 各有 8/4 LDSM wavefront；Long Scoreboard 为 18.645 cycles/issued instruction。由于 V3 删除了大量标量指令，`cycles/issued instruction` 的分母也发生了变化，不能仅凭 stall 数值上升断言性能退化。

### 5. 正确性验证

程序先将 FP32 输入量化为 FP16，再把量化值转回 FP32 计算 CPU reference。Tensor kernel 和 cuBLAS 均与这套 reference 比较，判定条件为：

```text
abs_error <= 1e-2 + 1e-2 * abs(reference)
```

每次 kernel 运行前将 C 填充为 NaN，并检查 reference、输出和误差均为 finite，以发现漏写 tile。独立测试覆盖五个形状、五个固定随机种子和五类输入分布：

- `128×192×64`：正负均匀分布，并注入 0/近零；
- `256×256×256`：正数均匀分布，并注入 0；
- `512×512×512`：负数均匀分布，并注入 0；
- `512×1024×256`：接近零的正负均匀分布；
- `1024×512×1024`：截断正态分布，并注入 0/近零。

V2.1、V3 和 cuBLAS 均通过。该次测试中三者输出恰好逐元素一致，但浮点 GEMM 的不同实现不保证 bitwise 等价，因此正确性标准仍是对量化输入 reference 的混合容差，而不是互相逐位相等。

### 6. 当前限制

- V3 要求 M/N 为 64 的倍数、K 为 16 的倍数；不满足条件时 runner 会跳过 V3，而不是执行通用边界路径。
- 当前只支持连续 Row-major 普通 GEMM，没有 arbitrary leading dimension、batched、split-K 或融合 epilogue。
- 固定 64×64 CTA tile 只完成了方阵性能验证；随机测试中的非方阵用于正确性验证，不能据此声称非方阵性能接近 cuBLAS。
- 尚未通过单变量实验确认 A shared padding、`cp.async`、double buffering 或降低寄存器能继续改善 V3。
- cuBLAS 对照是默认 `cublasGemmEx`，没有搜索 cuBLASLt heuristic，也没有 profile cuBLAS 内部 kernel；因此不对其具体 tiling、流水或数据搬运策略作实测声明。

---

## 构建与复现

### CUDA Core

```bash
mkdir -p build
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  src/gemm.cu -o build/gemm_fp32 -lcublas

./build/gemm_fp32
```

FP32 runner 当前将 1024³ 写在 `main()` 中；改变性能尺寸需要修改源码并重新编译。

### Tensor Core

```bash
mkdir -p build
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu \
  -o build/tensor_core_gemm -lcublas

nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  tests/v3_random_correctness.cu \
  -o build/v3_random_correctness -lcublas

./build/v3_random_correctness
./build/tensor_core_gemm 1024 1024 1024
./build/tensor_core_gemm 2048 2048 2048
```

为了复现十样本统计，每个 Tensor Core 性能命令需运行两个独立进程，再聚合两次输出中的五轮延迟。普通 CUDA Event 时间用于性能比较；Nsight Compute replay duration 只用于 profiler 内部分析。

## 总结

CUDA Core 路线的主要收益来自二维 register blocking 与向量化；在 1024³ 快照中，最佳 V7_2 为 14.755 TFLOPS，是 Naive 的 7.08×、同轮 `cublasSgemm` 的 74.92%。后续 double buffering 和 prefetch 只有小幅收益，V8 warp-layout 实验还出现过回退。

Tensor Core 路线先通过 shared tiling 和 fragment reuse减少重复工作，再用两个单变量 padding 实验定位并降低 B 侧 LDSM bank conflict，最后用 V3 的 128-bit staging 大幅减少搬运指令。V3 在更严格的双进程测试中达到 44.518 TFLOPS（1024³）和 56.017 TFLOPS（2048³），分别为同轮默认 cuBLAS 的 96.43% 和 95.46%。这些结果证明了当前实现对已测规则形状的有效性，但不代表对任意矩阵形状、GPU 或 GEMM 功能组合具有同等表现。
