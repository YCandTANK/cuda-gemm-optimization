# FP8 Tensor Core GEMM V5 验证与性能报告

日期：2026-11-03  
平台：WSL Ubuntu / NVIDIA GeForce RTX 5070 / Compute Capability 12.0（SM120）

## 1. 结论摘要

本轮已完成 `fp8_gemm_v5()` 的工程集成、正确性回归、PTX/SASS 检查、CUDA Event 性能对照和 1024³ Nsight Compute 对照。没有修改 V1/V2/V3/V4 的核心计算实现，也没有继续进行 Padding、`cp.async`、Double Buffering、流水线或更大 CTA Tile 等优化。

实测结论：

- V5 的 12/12 组正确性测试全部通过；V5 相对 FP32 Reference、V4 的最大绝对误差和最大相对误差均为 0，未出现 NaN/Inf、输出遗漏或 guard 区破坏。
- XOR Swizzle 将 V4 的 Shared Load Bank Conflict 从 1,575,683 降到 854，下降 99.946%；NCU 的理论 Excessive Shared Wavefronts 从 1,572,864 降到 0。Shared Store Bank Conflict 在 V4/V5 中均为 0。
- 1024³ CUDA Event 中位性能由 29.136 TFLOPS 提高到 33.301 TFLOPS，V5 加速 1.143x；2048³ 由 31.330 提高到 34.651 TFLOPS，加速 1.106x。
- 512³ kernel 仅约 0.013 ms，轮次波动显著；中位数显示 V5 为 0.984x，不能认为在该尺寸上有稳定提升。
- V5 的代价是 Registers/Thread 从 44 增到 48，总动态指令从 6,078,464 增到 6,356,992（+4.58%）。理论 Occupancy 仍为 83.33%，未发生 spill。
- V5 保留 `QMMA.16832.F32.E4M3.E4M3`、32-bit Global Load、Shared Store/Load 和两个 CTA Barrier；本轮优化目标与实际性能变化一致。

## 2. 工程集成与文件状态

本轮新增/修改：

- `src/fp8_gemm.cuh`：新增 `fp8_gemm_v5()` 声明。
- `CMakeLists.txt`：新增独立目标 `fp8_gemm_v5_compare`。
- `tests/fp8_gemm_v5_compare.cu`：新增 V4/V5 正确性、CUDA Event Benchmark 和 NCU profile 入口。
- `FP8_V5_VALIDATION_AND_PERFORMANCE.md`：本报告。

用户已写好的 V5 位于 `src/fp8_gemm.cu:643`。本轮没有改写 V5 核心，也没有改动 V1/V2/V3/V4 核心。与 `fp8-v4` tag 比较时，V4 区域存在本轮开始前已有的注释文字变化（编号/空格），执行语句不变；V5 是追加的新 kernel。

测试启动配置与 V4 完全相同：

```text
block = (128, 1, 1)
grid  = (N / 16, M / 32, 1)
CTA output tile = 32 x 16
K tile = 32
static shared memory = 1536 bytes
```

本轮未创建 Git Commit 或 Tag。工作树原本已有 `README.md` 修改、多份历史报告及 V3/V4 测试文件未跟踪；这些文件没有被清理、覆盖或提交。

## 3. 构建环境与命令

环境：

```text
GPU: NVIDIA GeForce RTX 5070
Compute Capability: 12.0
VRAM reported by CUDA Runtime: 12,792,365,056 bytes
NVCC: CUDA 13.2, V13.2.86
Nsight Compute CLI: 2026.1.1.0
Target architecture: sm_120
```

构建命令：

```bash
cmake -S . -B build/cmake-sm120 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-sm120 -j --target \
  fp8_gemm_v1_test fp8_gemm_v2_compare fp8_gemm_v3_compare \
  fp8_gemm_v4_compare fp8_gemm_v5_compare
```

所有目标构建成功。`ptxas` 资源结果如下：

| Kernel | Registers/Thread | Static Shared | Barrier slots | Spill |
|---|---:|---:|---:|---:|
| V1 | 39 | 0 B | 0 | 0 B |
| V2 | 40 | 0 B | 0 | 0 B |
| V3 | 48 | 0 B | 0 | 0 B |
| V4 | 44 | 1536 B | 1 | 0 B |
| V5 | 48 | 1536 B | 1 | 0 B |

V5 相对 V4 增加 4 个寄存器，但理论活动 block/warp 数未改变。

## 4. XOR Swizzling 索引与并发安全检查

V5 对 A/B 使用相同形式：

```text
physical_word_k = word_k ^ (major_index & 4)
```

其中 `word_k` 范围为 `[0, 7]`，`major_index & 4` 只能是 0 或 4。因此该映射只翻转 `word_k` 的 bit 2：

```text
major bit2 = 0: 0 1 2 3 4 5 6 7 -> 0 1 2 3 4 5 6 7
major bit2 = 1: 0 1 2 3 4 5 6 7 -> 4 5 6 7 0 1 2 3
```

静态证明与检查结果：

- 映射闭合在 `[0, 7]`，因此 `As_words[32][8]` 和 `Bs_words[16][8]` 不会因 Swizzle 越界。
- 对固定 major，该映射是双射，且 XOR 自反：`f(f(word_k)) = word_k`；没有遗漏或两个逻辑 word 写入同一物理 word。
- A Store 使用 `row`，A Load 使用同一逻辑坐标的 `shared_row`；B Store 使用 `b_col`，B Load 使用同一逻辑坐标的 `shared_col`。Store/Load 的 major 定义一致。
- V4 原有 cooperative write 分工不变：A 的 256 个 word 每个恰好写一次，B 的 128 个 word 每个恰好写一次；Swizzle 只是每行/列内部的排列，不引入写竞争。
- 第一次 `__syncthreads()` 保证本 K Tile 的 Shared 写入完成后才读取；第二次保证所有 Warp 读取完成后才允许下一 K Tile 覆盖 single buffer。两处 barrier 均为无条件 CTA-wide 路径。
- 32-bit Global Load 的逻辑地址仍以 4 个 FP8 元素为一组；Swizzle 只作用于 Shared word 下标，不改变 Global pointer 对齐。

## 5. 正确性验证

运行命令：

```bash
./build/cmake-sm120/fp8_gemm_v4_compare --correctness
./build/cmake-sm120/fp8_gemm_v5_compare --correctness
```

Reference 使用实际 E4M3 量化并重新扩展到 FP32 的 A/B；A 为 Row-major，B 按 Column-major 物理存储；FP32 Reference 按 K 顺序使用 `std::fma` 累加。V4 与 V5 使用完全相同的量化输入。

判定条件：

```text
abs_error <= 1e-4 + 1e-4 * abs(reference)
```

该容差用于避免不同浮点实现顺序的误判，但本次所有结果实际为 bitwise-equivalent 数值（报告误差均为 0），不是依赖宽松容差通过。输出在运行前填充 NaN，并在 C 前后各设置 64 个 guard 元素，以检查未覆盖和越界写。

| Case | Shape MxNxK | Pattern / Seed | V5 vs Ref max abs / rel | V5 vs V4 max abs / rel | Result |
|---|---:|---|---:|---:|---|
| zero_single_cta | 32x16x32 | zero / 8101 | 0 / 0 | 0 / 0 | PASS |
| sparse_single_cta | 32x16x32 | sparse / 8102 | 0 / 0 | 0 / 0 | PASS |
| distinct_single_cta | 32x16x32 | nonuniform / 8103 | 0 / 0 | 0 / 0 | PASS |
| mixed_three_k_tiles | 32x16x96 | signed mixed / 8201 | 0 / 0 | 0 / 0 | PASS |
| random_multi_cta_a | 64x32x64 | random / 8301 | 0 / 0 | 0 / 0 | PASS |
| rectangular_nonuniform | 96x48x160 | nonuniform / 8401 | 0 / 0 | 0 / 0 | PASS |
| random_multi_cta_b | 64x64x128 | random / 8501 | 0 / 0 | 0 / 0 | PASS |
| random_128 | 128x128x256 | random / 8601 | 0 / 0 | 0 / 0 | PASS |
| random_256 | 256x256x512 | random / 8701 | 0 / 0 | 0 / 0 | PASS |
| random_ten_k_tiles | 32x16x320 | random / 8801 | 0 / 0 | 0 / 0 | PASS |
| random_rectangular_extra | 160x80x224 | random / 8802 | 0 / 0 | 0 / 0 | PASS |
| nonuniform_multi_cta_extra | 192x96x96 | nonuniform / 8803 | 0 / 0 | 0 / 0 | PASS |

汇总：12/12 PASS；V4/V5 的 `over_tol=0`、`nonfinite=0`，所有 guard 均为 `OK`。另外单独复跑 V4 原有 9 组回归也全部通过。

## 6. PTX 与 SASS 检查

检查命令：

```bash
nvcc -std=c++17 -O3 -arch=sm_120 -I src -ptx \
  src/fp8_gemm.cu -o build/fp8_v5_probe/fp8_gemm.ptx
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v5_compare
```

V5 PTX 中确认：

```text
xor.b32
ld.global.b32
st.shared.b32
ld.shared.b32
bar.sync 0
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

V5 SASS 中确认：

```text
LDG.E
STS
LDS
BAR.SYNC.DEFER_BLOCKING
QMMA.16832.F32.E4M3.E4M3
LOP3.LUT                 # XOR/掩码及地址逻辑的一部分
```

因此 V5 仍然是真实 FP8 Tensor Core 路径，且宽 Global Load、Shared Store/Load、Barrier 均没有被错误消除。PTX 的 XOR 在 SASS 中主要被折叠为 `LOP3.LUT` 和地址生成逻辑，而不是机械地保持为单独命名的 XOR 指令。

1024³ 时共有 2048 CTA、每 CTA 32 个 K Tile，即 65,536 个 CTA-K-Tile。NCU 动态计数与源码每 K Tile 的预期关系一致：

| Warp-level dynamic operation | Total | Per CTA-K-Tile | 说明 |
|---|---:|---:|---|
| Global Load | 786,432 | 12 | A: 8，B: 4 |
| Shared Store | 786,432 | 12 | A: 8，B: 4 |
| Shared Load | 1,572,864 | 24 | 每 Warp 4 次 A + 2 次 B |
| CTA barrier arrival | 524,288（结构推导） | 8 | 2 barriers x 4 warps |
| FP8 QMMA | 262,144（结构推导） | 4 | 每 Warp 每 K Tile 1 次 |

V4/V5 的 Global/Shared 指令数量完全相同。NCU 总动态指令由 6,078,464 增至 6,356,992，净增加 278,528，即平均每 CTA-K-Tile 增加 4.25 条 warp-level 指令。这个净值包含编译器对 Swizzle、循环展开和地址生成的整体重新调度，不能简单等同于源码 XOR 的条数。

## 7. CUDA Event Benchmark

运行命令：

```bash
./build/cmake-sm120/fp8_gemm_v5_compare --benchmark
```

协议：同一进程、同一输入/分配、V4/V5 交替顺序；每个版本在每轮分别预热并用 CUDA Event 计时。512³/1024³ 为 10 次预热，分别计时 100/50 次；2048³ 为 5 次预热、计时 20 次。TFLOPS 按 `2*M*N*K/time` 计算。

### 512³

| Round | Order | V4 ms | V4 TFLOPS | V5 ms | V5 TFLOPS | V5/V4 speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V4,V5 | 0.014464 | 18.559 | 0.014641 | 18.335 | 0.988 |
| 2 | V5,V4 | 0.012595 | 21.314 | 0.016481 | 16.288 | 0.764 |
| 3 | V4,V5 | 0.013362 | 20.089 | 0.013342 | 20.119 | 1.002 |
| 4 | V5,V4 | 0.013762 | 19.505 | 0.013906 | 19.304 | 0.990 |
| 5 | V4,V5 | 0.012553 | 21.384 | 0.013581 | 19.765 | 0.924 |
| 6 | V5,V4 | 0.012568 | 21.358 | 0.012484 | 21.502 | 1.007 |
| 7 | V4,V5 | 0.015416 | 17.412 | 0.012611 | 21.285 | 1.222 |
| Median | - | **0.013362** | **20.089** | **0.013581** | **19.765** | **0.984** |

该尺寸单次 kernel 极短，轮次 speedup 为 0.764x～1.222x，噪声远大于预期优化量；结论是“未观察到稳定提升”，而不是 V5 确定退化。

### 1024³

| Round | Order | V4 ms | V4 TFLOPS | V5 ms | V5 TFLOPS | V5/V4 speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V4,V5 | 0.078623 | 27.314 | 0.067656 | 31.741 | 1.162 |
| 2 | V5,V4 | 0.073704 | 29.136 | 0.067576 | 31.779 | 1.091 |
| 3 | V4,V5 | 0.073192 | 29.340 | 0.064122 | 33.490 | 1.141 |
| 4 | V5,V4 | 0.076528 | 28.061 | 0.064058 | 33.524 | 1.195 |
| 5 | V4,V5 | 0.069731 | 30.797 | 0.064077 | 33.514 | 1.088 |
| 6 | V5,V4 | 0.077128 | 27.843 | 0.073264 | 29.312 | 1.053 |
| 7 | V4,V5 | 0.069811 | 30.761 | 0.064486 | 33.301 | 1.083 |
| Median | - | **0.073704** | **29.136** | **0.064486** | **33.301** | **1.143** |

7/7 轮 V5 均快于 V4，提升具有方向一致性。

### 2048³

| Round | Order | V4 ms | V4 TFLOPS | V5 ms | V5 TFLOPS | V5/V4 speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V4,V5 | 0.518310 | 33.146 | 0.495790 | 34.651 | 1.045 |
| 2 | V5,V4 | 0.548613 | 31.315 | 0.479685 | 35.815 | 1.144 |
| 3 | V4,V5 | 0.550134 | 31.228 | 0.498790 | 34.443 | 1.103 |
| 4 | V5,V4 | 0.548350 | 31.330 | 0.480426 | 35.760 | 1.141 |
| 5 | V4,V5 | 0.537880 | 31.940 | 0.507323 | 33.864 | 1.060 |
| 6 | V5,V4 | 0.558373 | 30.768 | 0.496914 | 34.573 | 1.124 |
| 7 | V4,V5 | 0.537043 | 31.990 | 0.479794 | 35.807 | 1.119 |
| Median | - | **0.548350** | **31.330** | **0.495790** | **34.651** | **1.106** |

7/7 轮 V5 均快于 V4。

## 8. Nsight Compute 1024³ 对照

采集命令的核心参数：

```bash
ncu \
  --section SpeedOfLight \
  --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --section WarpStateStats \
  --section SchedulerStats \
  --section InstructionStats \
  --section SourceCounters \
  --section LaunchStats \
  --section Occupancy \
  --kernel-name-base function --kernel-name fp8_gemm_v4 \
  --launch-skip 5 --launch-count 1 --kill yes \
  --cache-control none --force-overwrite \
  -o build/ncu/fp8_v4_1024_v5_compare_20261103 \
  ./build/cmake-sm120/fp8_gemm_v5_compare --profile-v4
```

V5 仅将 kernel name/output/mode 换成 `fp8_gemm_v5`、`fp8_v5_1024_compare_20261103`、`--profile-v5`。两份报告均完成 35 passes：

```text
build/ncu/fp8_v4_1024_v5_compare_20261103.ncu-rep
build/ncu/fp8_v5_1024_compare_20261103.ncu-rep
```

### 8.1 Shared Memory 冲突

| Metric | V4 | V5 | 变化 |
|---|---:|---:|---:|
| Shared Load Requests / instructions | 1,572,864 | 1,572,864 | 相同 |
| Shared Load Wavefronts | 3,148,708 | 1,573,700 | -50.02% |
| Avg. wavefronts per Shared Load | 2.0019 | 1.0005 | 接近 2-way -> 1-way |
| Shared Load Bank Conflicts | 1,575,683 | 854 | **-99.946%** |
| Theoretical Excessive Shared Wavefronts | 1,572,864 | 0 | **完全消除** |
| Shared Store Requests / wavefronts | 786,432 / 786,432 | 786,432 / 786,432 | 相同 |
| Shared Store Bank Conflicts | 0 | 0 | 相同 |

V5 的 load conflict 计数不是数学上的绝对 0，但 854 仅占 1,572,864 次 load 的约 0.054%；同时 NCU 的理论 Excessive Wavefronts 已为 0。因此可以确认 V4 的系统性平均 2-way Shared Load Bank Conflict 已消除，剩余值属于极小的运行/计数残差，不再是结构性 2-way 模式。

### 8.2 Warp stall、Tensor Pipe 与 Occupancy

以下 stall 值为 NCU 的 average warps stalled per issued instruction（cycle/issue 口径），不是 wall-clock 时间：

| Metric | V4 | V5 | 相对变化 |
|---|---:|---:|---:|
| MIO Throttle | 22.107 | 16.901 | -23.5% |
| Barrier Stall | 9.055 | 7.214 | -20.3% |
| Long Scoreboard | 15.537 | 16.321 | +5.0% |
| LG Throttle | 0.298 | 0.221 | -26.0% |
| Tensor Pipe Active（active cycles） | 23.66% | 26.01% | +2.35 pp |
| Tensor Pipe Active（elapsed cycles） | 22.47% | 24.58% | +2.11 pp |
| Active Warps / Scheduler | 9.117 | 9.113 | 基本相同 |
| Eligible Warps / Scheduler | 0.549 | 0.589 | +7.2% |
| Theoretical Occupancy | 83.33% | 83.33% | 相同 |
| Achieved Occupancy | 76.09% | 76.34% | 基本相同 |
| Registers/Thread | 44 | 48 | +4 |
| Spill | 0 | 0 | 相同 |

冲突消除后，Shared/L1TEX 每次 LDS 所需 wavefront 减半，MIO queue 压力和 barrier 等待明显下降，可发射 Warp 增多，Tensor Pipe 利用率提高。这一证据链与 CUDA Event 的 1024³/2048³ 提升一致。

Long Scoreboard 的 ratio 略升并不表示绝对等待时间一定增加：V5 总执行时间和每条指令间隔均下降，同时动态指令数增加，该 ratio 的分母和调度状态发生变化。它说明冲突消除后，Global/L1TEX dependency 在剩余停顿中的相对重要性上升，已成为更显眼的下一层瓶颈。

### 8.3 Global Load 与 SOL

| Metric | V4 | V5 |
|---|---:|---:|
| Global Load Requests | 786,432 | 786,432 |
| Global Load Sectors | 3,145,728 | 3,145,728 |
| Global Store instructions | 32,768 | 32,768 |
| L1/TEX Throughput | 84.13% | 76.88% |
| L2 Throughput | 68.03% | 73.69% |
| DRAM Throughput | 4.60% | 7.98% |
| Compute Throughput | 66.65% | 72.66% |
| Total dynamic instructions | 6,078,464 | 6,356,992 |
| NCU profiled duration | 80.608 us | 74.528 us |

Global Load 请求与 sector 数完全相同，证明 V5 的收益不是减少 Global Memory traffic；L1/TEX throughput 百分比下降主要反映 Shared replay/wavefront 压力下降。DRAM 利用率始终很低，不能把 V4/V5 的性能解释为 DRAM 带宽瓶颈。

NCU 明确警告本次使用 `--cache-control none`，GPU cache 未受控；多 pass 报告中的 L2 hit rate 甚至出现略大于 100% 的不可解释派生值。因此本报告不使用 L2 hit rate、DRAM 百分比的细小差异做因果结论，也不把 NCU replay duration 当作 CUDA Event Benchmark 时间。

## 9. 理论预期与实测结果的区分

理论预期：

- XOR bit2 应把 V4 同 bank 的两组地址打散，使 Shared Load 从平均 2-way 变为无冲突。
- Global load/store 数、Shared load/store 指令数、QMMA 数和 barrier 数应保持不变。
- Swizzle 会增加整数/XOR/地址生成工作，可能增加寄存器压力；如果这些成本大于 replay 减少，性能可能不升。

实测结果：

- Shared Load 的系统性 2-way conflict 确实消失；store 原本无冲突且保持无冲突。
- Global/Shared/QMMA/Barrier 路径均保留，Global traffic 数量不变。
- V5 增加 4 registers/thread 和 4.58% 总动态指令，但没有降低理论 Occupancy，也没有 spill。
- replay 减少的收益大于新增索引成本：1024³、2048³ 分别获得 1.143x、1.106x CUDA Event 中位加速。
- 512³ 太短且波动过大，未得到稳定收益；不能把大尺寸结论外推到该尺寸。

## 10. 当前 V5 局限与后续边界

- 仍要求 M/N/K 分别是 32/16/32 的整数倍，不支持边界 tile。
- A 为 Row-major，B 为 Column-major，输入必须已量化为 E4M3；kernel 不负责 scaling/quantization。
- CTA 仍是 single-buffer shared staging，每个 K Tile 有两次 CTA barrier。
- V5 已解决 V4 最明显的 Shared replay 问题，但 MIO Throttle 和 Long Scoreboard 仍高，Tensor Pipe Active 仍只有约 26%。
- Global load sector/request 比没有变化，C Global Store 仍存在 NCU 指出的 sector 利用不足；这些均不属于本轮允许修改的范围。
- 本轮只完成诊断和验证，没有加入 Padding、`cp.async`、Double Buffering、多级流水线或更大 CTA Tile。

最终判定：`fp8_gemm_v5()` 工程集成和正确性验证通过；SASS/NCU 证据确认 XOR Swizzling 达到了消除 V4 平均 2-way Shared Load Bank Conflict 的目标，并在代表性 1024³、2048³ 负载上带来可重复的性能提升。
