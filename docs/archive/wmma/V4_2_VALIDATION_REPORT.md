# Tensor Core WMMA GEMM V4_2 Double Buffering 验证报告

日期：2026-11-02（Asia/Shanghai）

## 1. 结论摘要

本轮开始时，工作树中已经存在用户手写的 `wmmaGemmV4_2()`；没有重新设计或生成另一套 kernel。静态审查未发现需要修改的正确性 Bug，因此 V4_2 kernel body 保持原样。本轮只完成 host 侧工程集成、测试、PTX/SASS、Benchmark 和 NCU 分析；冻结的 V3、V4_1 kernel 均未修改。

结论如下：

- V4_2 的 prologue、`current/next` 轮换、wait/barrier 和最后一 tile 处理正确；K=16、32、48 及更多 tile 均通过。
- `sm_120` 编译成功。V4_2 为 78 registers/thread、8704 B kernel static shared memory，无 register spill。
- 所有要求尺寸均通过 FP16 量化输入 CPU reference；V4_2 与 V3、V4_1 的输出逐元素完全一致。
- PTX 确实包含 `cp.async`；SASS 确实包含 `LDGSTS`、`LDGDEPBAR` 和 `DEPBAR`。稳态 SASS 在发出下一 tile 的异步复制后执行当前 tile 的 LDSM/HMMA，随后才 wait/barrier，静态顺序提供了真实的重叠机会。
- 2048³ 正式 CUDA Event 中，V4_2 比 V4_1 快 1.0383x，但比 V3 慢约 0.45%，达到同轮 cuBLAS 的 94.16%。因此 V4_2 修复了 V4_1 大部分性能退化，但没有在该次正式测量中稳定超过 V3。
- 1024³ 两次独立测量中，V4_2 均慢于 V3；相对 V4_1 一次快、一次慢，短任务波动明显，不能据单轮极值宣布加速。
- NCU 中，V4_2 相比 V4_1 将动态 CTA barrier 指令减半，Long Scoreboard、Barrier Stall 明显下降，Tensor/HMMA active 回升；但动态总指令显著增加、寄存器和 occupancy 回到 V3 水平，而且 V4_1 的 B-side global sector 增量仍然存在。
- 当前证据不支持用 V4_2 替换冻结的 V3 作为稳定更快版本。下一步应优先调查动态控制/地址计算指令膨胀和 B-side sector 行为，而不是自动增加第三套 buffer 或改变 tile/layout。

## 2. 修改范围

修改/新增文件：

- `src/tensor_core_gemm.cu`
  - 确认已有 `<cuda_pipeline.h>` 和 `wmmaGemmV4_2()`。
  - `KernelVersion` 增加 `V4_2`。
  - `kernelName()` 增加 V4_2 名称。
  - `launchKernel()` 增加 V4_2 分支，沿用 V3 的 grid/block。
  - `runCase()` 增加 V4_2 correctness、V3/V4_2 与 V4_1/V4_2 精确输出对照、V4_2/cuBLAS 对照。
  - 将 V4_2 加入现有 rotating-order interleaved CUDA Event benchmark，输出 V4_2/V3、V4_2/V4_1 和 V4_2/cuBLAS 比例。
- `V4_2_VALIDATION_REPORT.md`
  - 本报告。

没有 Git commit/push，没有扩大尺寸支持范围，也没有加入第三套 buffer、修改 tile、padding、WMMA 计算或 C 写回。

## 3. V4_2 源码与同步正确性审查

以下是由源码直接确认的事实。

### 3.1 线程映射与布局

V4_2 继续使用 V3/V4_1 的 64×64×16 CTA tile、32×32 warp tile、128 threads/CTA、A row stride 16 和 B shared stride 72。A/B 的 cooperative load 索引公式未变；唯一算法级变化是 shared 数组增加首维 `[2]`，并在稳态提前复制下一 tile。

所有 WMMA shared 指针都包含 `current`：

```cpp
&As[current][...][0]
&Bs[current][0][...]
```

异步写入都包含 `next`，因此计算和预取访问不同 plane。

### 3.2 16-byte 对齐

仍只支持 M/N 为 64 的倍数、K 为 16 的倍数。在此条件下：

- `cudaMalloc` 的 A/B 基址满足 16-byte 对齐；A/B global row stride 和每线程列偏移均为 16 B 的倍数。
- `As`、`Bs` 显式 32-byte 对齐。
- 一个 A plane 为 `64*16*2 = 2048 B`，一个 B plane 为 `16*72*2 = 2304 B`；二者均为 16 B 的倍数。
- A shared row stride 为 32 B；B shared row stride 为 144 B，也为 16 B 的整数倍。

因此两个 buffer plane 内每次 16 B async copy 的 source/destination 均满足对齐要求。

### 3.3 时序与生命周期

Prologue 向 `current=0` 发出 tile 0 的 A/B copy，commit 后 `wait_prior(0)`，再执行 CTA barrier，故首次计算前 tile 0 已对所有 warp 可见。

稳态每轮顺序为：

```text
若有下一 tile：copy(tile+1 -> next) + commit
compute(tile <- current)
若有下一 tile：wait_prior(0) + CTA barrier
swap(current, next)
```

安全性分情况如下：

- K=16：只有 prologue 和一次计算；不预取、不越界，循环后的无意义 swap 不再被读取。
- K=32：tile 0 计算期间预取 tile 1；wait/barrier 后交换，tile 1 只计算、不再预取。
- K≥48：在复用旧 plane 前，上一轮计算已经完成，且所有 warp 都通过该轮末尾 CTA barrier，因此不会覆盖仍被其他 warp 读取的 buffer。
- 最后一 tile 不进入预取分支，不会访问 `K` 之外；最后结果也不需要额外 wait/barrier。

`__pipeline_wait_prior(0)` 保证每线程已提交 copy 完成；随后的 `__syncthreads()` 将这个条件提升为 CTA-wide 的可见性与执行 rendezvous。没有发现提前读取未完成 copy，或提前覆盖仍在读取 plane 的路径。

本轮没有对 V4_2 kernel body 做修复。

## 4. 构建、资源与正确性

### 4.1 构建环境与命令

- GPU：NVIDIA GeForce RTX 5070，CC 12.0
- CUDA compiler/runtime：13.2；Driver API：13050；cuBLAS runtime：130401
- 构建命令：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu \
  -o build/tensor_core_gemm_v4_2 -lcublas
```

编译成功。唯一 warning 是既有 V0 的未使用 `lane_id`，与 V4_2 无关。

### 4.2 编译资源

| Kernel | Registers/thread | Kernel static shared | Stack | Spill store/load |
|---|---:|---:|---:|---:|
| V3 | 78 | 4352 B | 0 B | 0 / 0 B |
| V4_1 | 72 | 4352 B | 0 B | 0 / 0 B |
| V4_2 | 78 | 8704 B | 0 B | 0 / 0 B |

NCU 报告 V4_2 static shared 为 8.70 KB、driver shared 为 1.02 KB。`cuobjdump --dump-resource-usage` 的 9728 B 是二者合计；kernel 自身的双缓冲容量与预期 8704 B 完全一致。

### 4.3 正确性矩阵

判定标准保持原工程不变：

```text
abs_error <= 1e-2 + 1e-2 * abs(FP16-quantized-input CPU reference)
```

| Shape | V4_2 quantized-ref max abs | max rel | V4_2 | V3/V4_1 vs V4_2 |
|---|---:|---:|---|---|
| 64×64×16 | 4.768372e-07 | 3.003707e-06 | PASS | max abs=0、max rel=0、exact PASS |
| 64×64×32 | 5.960464e-07 | 1.157327e-05 | PASS | max abs=0、max rel=0、exact PASS |
| 64×64×48 | 5.364418e-07 | 2.008032e-03 | PASS | max abs=0、max rel=0、exact PASS |
| 128×192×64 | 8.344650e-07 | 8.325396e-07 | PASS | max abs=0、max rel=0、exact PASS |
| 1024³ | 6.079674e-06 | 1.020690e-05 | PASS | max abs=0、max rel=0、exact PASS |
| 2048³ | 1.084805e-05 | 3.015384e-04 | PASS | max abs=0、max rel=0、exact PASS |

64×64×48 另外以 5 个独立进程重复运行，5/5 均通过 reference 和精确输出对照。每次正式检查都执行 NaN fill、launch、`cudaGetLastError()`、`cudaDeviceSynchronize()` 和 D2H copy；所有正式运行 exit code 为 0，未观察到 CUDA Runtime error。

64×96×32 不满足 V3/V4.1/V4.2 的 M/N 64-multiple 条件，按原支持范围 skip，没有实现边界处理。

### 4.4 Compute Sanitizer 限制

对 K=48 的 V4_2 分别尝试了 `memcheck` 和 `racecheck`，并使用 V4_2 kernel filter。两者都在开始硬件插桩前失败：

```text
Failed to initialize WDDM debugger interface.
Please run EnableDebuggerInterface.bat as an administrator
Device not supported
```

`racecheck` 的汇总数字来自这两个初始化错误，没有给出任何具体 shared address、线程对或 RAW/WAR/WAW 诊断。因此本轮只能报告 sanitizer 未完成，不能据此宣称 memcheck/racecheck clean，也不能据此否定静态同步审查。

## 5. PTX / SASS 证据

PTX 构建：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=compute_120 -ptx \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_v4_2.ptx
```

V4_2 PTX 实际出现：

```text
cp.async.cg.shared.global [...], [...], 16, 16;
cp.async.commit_group;
cp.async.wait_group 0;
```

Prologue 的 V4_2 SASS 为：

```text
/*0380*/ LDGSTS.E.BYPASS.128 [R61], desc[UR10][R4.64]
/*03a0*/ LDGSTS.E.BYPASS.128 [R57], desc[UR10][R8.64]
/*03b0*/ LDGDEPBAR
/*03e0*/ DEPBAR.LE SB0, 0x0
/*03f0*/ BAR.SYNC.DEFER_BLOCKING 0x0
```

一个稳态片段为：

```text
/*08a0*/ LDGSTS.E.BYPASS.128 [R61+0x800], desc[UR10][R28.64]
/*08b0*/ LDGSTS.E.BYPASS.128 [R57+0x900], desc[UR10][R30.64]
/*08c0*/ LDGDEPBAR
/*0910*/ LDSM.16.M88.4 ...
/*0950*/ LDSM.16.MT88.4 ...
/*09f0*/ HMMA.16816.F32 ...
          ...其余当前 tile HMMA...
/*0af0*/ DEPBAR.LE SB0, 0x0
/*0b00*/ BAR.SYNC.DEFER_BLOCKING 0x0
```

这证明编译后的软件顺序在异步复制发出后安排了独立的当前 tile WMMA 工作，并非 commit 后立即 wait。它证明“存在重叠机会”，但不能单独证明硬件实际隐藏了多少延迟；后者需结合 NCU 与 Event 结果判断。

V3 的 staging 路径仍为独立的：

```text
LDG.E.128 ...
LDG.E.128 ...
STS.128 ...
STS.128 ...
```

V4_2 中这组显式 `LDG.E.128 -> STS.128` 被融合为 `LDGSTS.E.BYPASS.128`。PTX `commit_group`/`wait_group` 在 SASS 中表现为 `LDGDEPBAR`/`DEPBAR.LE`，而不是保留同名文本指令。

## 6. CUDA Event Benchmark

协议：默认 stream、相同 FP16 A/B 和 FP32 C allocation、每轮 10 次 warmup、100 次 kernel 的平均延迟作为原始轮次、5 个 rotating-order interleaved rounds，最终取中位数。原框架中的 V2.1 仍保留在交错序列中，但以下只列本任务的 V3、V4_1、V4_2、cuBLAS。所有 cuBLAS 数值均来自本轮同进程同输入实测，没有混用历史数据。

### 6.1 1024³：独立测量 1

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.045280, 0.045351, 0.045247, 0.045126, 0.045306 | 0.045280 | 47.426761 |
| V4_1 | 0.050800, 0.056528, 0.047615, 0.047487, 0.051483 | 0.050800 | 42.273301 |
| V4_2 | 0.046357, 0.048878, 0.055185, 0.046303, 0.046762 | 0.046762 | 45.924086 |
| cuBLAS | 0.040368, 0.040430, 0.040508, 0.048825, 0.040737 | 0.040508 | 53.014442 |

- V4_2/V3：0.9683x
- V4_2/V4_1：1.0864x
- V4_2/cuBLAS：0.8663

### 6.2 1024³：独立测量 2

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.045505, 0.045283, 0.053524, 0.045407, 0.047232 | 0.045505 | 47.192297 |
| V4_1 | 0.047714, 0.048981, 0.047405, 0.056537, 0.047582 | 0.047714 | 45.007487 |
| V4_2 | 0.048166, 0.046256, 0.050629, 0.048484, 0.053814 | 0.048484 | 44.292481 |
| cuBLAS | 0.051339, 0.040115, 0.040504, 0.040363, 0.040407 | 0.040407 | 53.146697 |

- V4_2/V3：0.9386x
- V4_2/V4_1：0.9841x
- V4_2/cuBLAS：0.8334

两个独立进程都出现单轮高延迟，且 V4_2 对 V4_1 的方向不一致。测量后的空闲快照为 42°C、SM 480 MHz、显存 14001 MHz、27.63/250 W、GPU utilization 3%；该快照不是计时期间遥测，不能用于解释单轮抖动。可靠结论仅是：1024³ 没有显示 V4_2 稳定超过 V3。

### 6.3 2048³：正式测量

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.301420, 0.303914, 0.300278, 0.304521, 0.304079 | 0.303914 | 56.528795 |
| V4_1 | 0.317343, 0.316991, 0.318605, 0.313757, 0.314664 | 0.316991 | 54.196704 |
| V4_2 | 0.305815, 0.305617, 0.291834, 0.305285, 0.304533 | 0.305285 | 56.274892 |
| cuBLAS | 0.290818, 0.292756, 0.279330, 0.284974, 0.287444 | 0.287444 | 59.767678 |

- V4_2/V3：0.9955x，即 V4_2 median latency 高约 0.45%。
- V4_2/V4_1：1.0383x。
- V4_2/cuBLAS：0.9416。

V4_2 和 cuBLAS 各有一轮偏低值；结论采用中位数而非该极值。NCU 进程中的附带 benchmark 数值不用于本节统计。

## 7. Nsight Compute：2048³

V3、V4_1 和 V4_2 使用完全相同的 9 个 section、2048³ 输入、`--launch-count 1` 和 35-pass 配置。V3/V4_1 报告来自同日 V4_1 验证，V4_2 为本轮新采：

```bash
ncu \
  --section SpeedOfLight \
  --section ComputeWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --section InstructionStats \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis_Tables \
  --section SourceCounters \
  --kernel-name-base demangled \
  --kernel-name 'regex:<target-kernel>' \
  --launch-count 1 ...
```

- `build/ncu/v4_1_v3_2048.ncu-rep`
- `build/ncu/v4_1_v4_1_2048.ncu-rep`
- `build/ncu/v4_2_v4_2_2048.ncu-rep`

NCU replay duration 只用于 profiler 内的方向比较，不与 CUDA Event median 混用；stall 数值是 cycles/issued，不是耗时百分比。

### 7.1 核心指标

| Metric | V3 | V4_1 | V4_2 |
|---|---:|---:|---:|
| NCU replay duration | 325.60 us | 340.13 us | 328.42 us |
| Registers/thread | 78 | 72 | 78 |
| Static shared/block | 4.35 KB | 4.35 KB | 8.70 KB |
| Theoretical occupancy | 50.00% | 58.33% | 50.00% |
| Achieved occupancy | 46.00% | 52.47% | 46.29% |
| Local/shared spill requests | 0 / 0 B | 0 / 0 B | 0 / 0 B |
| Executed warp instructions | 14,270,464 | 15,839,232 | 24,944,640 |
| Active warps/scheduler | 5.53 | 6.38 | 5.58 |
| Eligible warps/scheduler | 0.15 | 0.15 | 0.23 |
| Issued warp/scheduler | 0.09 | 0.10 | 0.17 |
| Warp cycles/issued | 58.43 | 63.14 | 33.71 |
| Long Scoreboard | 18.847 | 29.080 | 10.138 |
| Short Scoreboard | 3.637 | 3.979 | 2.132 |
| Barrier Stall | 7.399 | 8.666 | 4.133 |
| Math Pipe Throttle | 19.221 | 14.379 | 10.237 |
| Tensor Pipe Active | 87.316% | 82.768% | 88.065% |
| HMMA subpipe active | 87.320% | 82.765% | 88.032% |
| DRAM throughput | 14.93% | 15.28% | 15.06% |
| L1/TEX throughput | 75.00% | 61.64% | 64.24% |
| L2 throughput | 62.75% | 79.42% | 82.36% |

V4_2 的动态总指令比 V4_1 多约 57.5%，比 V3 多约 74.8%。该增量与 V4_2 新增的 buffer 索引、循环控制和地址选择代码同时出现，但总数不能精确归因到某一种源码操作；可以确认的是，不能只因 stall 降低就忽略这部分动态开销。

### 7.2 动态 staging、barrier 与 Tensor 指令

| SASS 类别 | V3 | V4_1 | V4_2 |
|---|---:|---:|---:|
| Normal `LDG.E.128` | 1,048,576 | 0 | 0 |
| Normal `STS.128` | 1,048,576 | 0 | 0 |
| `LDGSTS` | 0 | 1,048,576 | 1,048,576 |
| `LDGDEPBAR` | 0 | 524,288 | 524,288 |
| `DEPBAR` | 0 | 524,288 | 524,288 |
| CTA barrier warp instructions | 1,048,576 | 1,048,576 | 524,288 |
| LDSM | 2,097,152 | 2,097,152 | 2,097,152 |
| HMMA | 4,194,304 | 4,194,304 | 4,194,304 |

双缓冲没有改变 copy、LDSM 或 HMMA 工作量，但把每 tile 两次 barrier 的 V4_1 结构降为稳态每 tile 一次，动态 barrier 正好减半。与 stall 和 Tensor active 的改善相互印证：V4_2 确实比 V4_1 更有效地安排了等待，但这些指标仍不能量化每次 copy 与 HMMA 的精确硬件重叠比例。

### 7.3 Global sectors 与 B-side 增量

| Metric | V3 | V4_1 | V4_2 |
|---|---:|---:|---:|
| Global load requests | 1,048,576 | 1,048,576 | 1,048,576 |
| Actual global load sectors | 16,777,216 | 17,825,792 | 17,825,792 |
| Source theoretical sectors | 16,777,216 | 20,971,520 | 20,971,520 |
| Source ideal sectors | 16,777,216 | 16,777,216 | 16,777,216 |
| Source excessive sectors | 0 | 4,194,304 | 4,194,304 |

对 V4_1/V4_2 的 source-correlated `LDGSTS` 分拆结果完全一致：

| Operand | L1 tag requests | Theoretical sectors | Ideal | Excessive |
|---|---:|---:|---:|---:|
| A | 8,388,608 | 8,388,608 | 8,388,608 | 0 |
| B | 2,097,152 | 12,582,912 | 8,388,608 | 4,194,304 |

因此 V4_1 中观察到的 B-side sector 增量在 V4_2 中仍然存在，双缓冲没有改善该访问属性。这里同时列出 actual hardware sector metric 和 source theoretical attribution，避免把两个不同口径混为同一个数值。

### 7.4 LDSM wavefront / bank conflict

三个版本的 WMMA shared layout 未变，source-correlated LDSM 结果也相同：

| Operand | LDSM instructions | Actual wavefronts | Ideal | Excessive |
|---|---:|---:|---:|---:|
| A | 1,048,576 | 8,388,608 | 4,194,304 | 4,194,304 |
| B | 1,048,576 | 4,194,304 | 4,194,304 | 0 |

V4_2 没有引入新的 LDSM bank-conflict 变化，也没有解决既有 A-side excessive wavefronts；这符合“不得修改 Shared Layout”的任务约束。

## 8. 证据解释与后续建议

### 8.1 已证实

- V4_2 的源码和实测输出均支持其双缓冲同步正确性。
- 编译器生成了真正的 async global-to-shared 路径，并在 wait 前放置当前 tile 的 Tensor 工作。
- 相比 V4_1，V4_2 降低 scoreboard/barrier 等待、恢复 Tensor pipe 利用率，并改善 2048³ Event/NCU 性能。
- 相比 V3，V4_2 在本轮两个目标尺寸上都没有形成稳定加速。

### 8.2 有证据支持的瓶颈

- V4_2 动态 warp 指令数显著膨胀，抵消了一部分等待下降收益。
- 寄存器回到 78/thread，使理论 occupancy 从 V4_1 的 58.33% 回到 50%；但 occupancy 不是独立的性能结论。
- B-side `LDGSTS` sector overhead 与 V4_1 相同。
- A-side LDSM excessive wavefronts 仍在，且受本轮冻结 shared layout 的约束无法处理。
- Tensor pipe 已约 88% active，说明单纯增加更多 buffer 未必有足够回报。

### 8.3 尚属假设

- V4_2 可能确实隐藏了部分 global-to-shared 等待；SASS 顺序和 stall 下降支持这个方向，但没有直接的“重叠周期数”计数器，因此不能把隐藏比例写成既定事实。
- 1024³ 的波动可能受短 kernel 调度、频率状态或系统负载影响；缺少计时期间连续遥测，不能指定唯一根因。

### 8.4 是否继续优化

V4_2 已达到“功能正确、真实 async、真实双缓冲”的实验目标，但目前不适合作为优于 V3 的最终版本。下一步候选应保持为独立实验：先精简 buffer 选择与循环地址计算、检查为何 V4_2 动态指令达到 24.9M，并研究 B-side async sector attribution。只有这些开销下降且 Event 中位数稳定超过 V3 后，才值得评估进一步流水化；当前证据不支持直接上 triple buffering、改变 tile/padding 或重写 WMMA。
