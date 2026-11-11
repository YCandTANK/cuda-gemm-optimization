# Tensor Core WMMA GEMM V4_1 集成、验证与性能分析

日期：2026-11-02（Asia/Shanghai）

## 1. 结论摘要

V4_1 在本轮开始时已经存在于工作树的 `src/tensor_core_gemm.cu` 中，`<cuda_pipeline.h>` 也已经加入；它并非本轮凭空补写。已有实现尚未接入 `KernelVersion`、launch、correctness 和 benchmark。本轮保留冻结的 V3 kernel，不修改线程映射、WMMA 计算、Shared Layout 或 C 写回，仅完成 V4_1 host 集成、V3/V4_1 输出对照、编译、正确性、PTX/SASS、CUDA Event benchmark 和 NCU 对照。

最终结果：

- `sm_120` 编译成功；V4_1 为 72 registers/thread，V3 为 78，二者均无 spill。
- V4_1 实际生成 PTX `cp.async`，SASS 实际生成 `LDGSTS.E.BYPASS.128`，不是仅凭源码 intrinsic 推断。
- 1024³、2048³ 和 128×192×64 均通过 CPU reference；V3/V4_1 输出最大绝对和相对差异均为 0。
- 1024³ 本轮中位数：V4_1 比 V3 快 1.1724x，但 V3 和 cuBLAS 原始轮次波动较大，不宜仅据该尺寸断言稳定收益。
- 2048³ 本轮中位数：V4_1 为 V3 的 0.9513x，即延迟高约 5.12%；V4_1 达到同轮 cuBLAS 的 90.44%。
- 2048³ NCU 中，V4_1 的 occupancy 提高，但动态总指令、Long Scoreboard 和 Barrier stall 增加，Tensor Pipe Active 降低；单 buffer 立即 wait 没有形成跨 K tile 的 load/compute 重叠。
- V4_1 已证明异步复制路径正确且工具链确实生成目标指令，适合把 V4_2 Double Buffering 作为下一项独立实验；但 V4_1 本身不能替代冻结的 V3 作为所有尺寸上的更快版本。

## 2. 修改范围

修改/新增文件：

- `src/tensor_core_gemm.cu`
  - 复核已有 `<cuda_pipeline.h>` 和 `wmmaGemmV4_1()`。
  - `KernelVersion` 增加 `V4_1`。
  - `kernelName()` 增加 V4_1 名称。
  - `launchKernel()` 使用与 V3 相同的 64×64×16 CTA tile、128 threads/CTA 和 grid 计算启动 V4_1。
  - `runCase()` 增加 V4_1 correctness、V3/V4_1 精确输出对照、V4_1/cuBLAS 对照。
  - 把 V4_1 加入原有 rotating-order interleaved CUDA Event benchmark，并输出 V4_1/V3 和 V4_1/cuBLAS 比例。
- `V4_1_VALIDATION_REPORT.md`
  - 本报告。

没有修改 V3 kernel，没有实现 double buffering，没有 Git commit/push，也没有改动历史未跟踪报告。

## 3. 源码审查事实

### 3.1 V4_1 与 V3 的唯一 kernel 级差异

V3 每个 K tile 对 A/B 分别执行：

```cpp
uint4 data = *reinterpret_cast<const uint4*>(src);
*reinterpret_cast<uint4*>(dst) = data;
```

随后执行 `__syncthreads()`。V4_1 保留相同索引、tile、shared 数组、WMMA fragment、MMA 顺序和 C 写回，只把两条 staging 替换为：

```cpp
__pipeline_memcpy_async(dstA, srcA, 16);
__pipeline_memcpy_async(dstB, srcB, 16);
__pipeline_commit();
__pipeline_wait_prior(0);
__syncthreads();
```

循环末尾原有 `__syncthreads()` 保留，用于防止下一 K tile 覆盖仍在使用的单 buffer。

### 3.2 16-byte 对齐

V4_1 没有扩大 V3 的支持范围，仍要求 M/N 为 64 的倍数、K 为 16 的倍数。在这个前提下：

- `cudaMalloc` 返回的 A/B 基址满足至少 16-byte 对齐。
- A 的 row byte stride 为 `K * sizeof(half)`，至少是 32 B 的倍数；`col_A` 是 8 half 的倍数，即 16 B 的倍数。
- B 的 row byte stride 为 `N * sizeof(half)`，N 是 64 的倍数；`col_B` 同样是 8 half 的倍数。
- `As` 基址显式 32-byte 对齐，row stride 为 `16 half = 32 B`。
- `Bs` 基址显式 32-byte 对齐，row stride 为 `72 half = 144 B = 9 * 16 B`。

因此每个 16-byte async copy 的 source 和 destination 都满足对齐要求。

### 3.3 同步语义

`__pipeline_commit()` 提交本线程此前发起的两次 async copy；`__pipeline_wait_prior(0)` 等待本线程所有更早提交组完成；随后 CTA-wide `__syncthreads()` 保证所有线程完成等待并让整个 tile 可安全供所有 warp 使用。这个顺序对当前单 buffer 正确。

但 wait 紧跟 commit，WMMA 计算发生在 wait 之后，因此源码结构上没有把下一 tile 的 global-to-shared copy 与当前 tile 的 WMMA 计算重叠。这是源码可确认的限制，不是 profiler 推测。

## 4. 构建与正确性

### 4.1 环境与命令

实测环境：

- GPU：NVIDIA GeForce RTX 5070，compute capability 12.0
- CUDA compiler：13.2，nvcc 13.2.86
- Driver API：13050
- CUDA Runtime：13020
- cuBLAS Runtime：130401

构建命令：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu \
  -o build/tensor_core_gemm_v4_1 -lcublas
```

编译成功。唯一 warning 是既有 V0 代码中的未使用 `lane_id`，与 V4_1 无关。

### 4.2 ptxas 资源

| Kernel | Registers/thread | Static shared | Stack | Spill store/load |
|---|---:|---:|---:|---:|
| V3 | 78 | 4352 B | 0 B | 0 / 0 B |
| V4_1 | 72 | 4352 B | 0 B | 0 / 0 B |

`cuobjdump --dump-resource-usage` 报告二者均为 5376 B shared；其中包含 NCU 所示的 1024 B driver shared，kernel 自身 static shared 仍为 4352 B。

### 4.3 主程序 correctness

判定标准保持原工程不变：

```text
abs_error <= 1e-2 + 1e-2 * abs(FP16-quantized-input CPU reference)
```

| Shape | V4_1 quantized-ref max abs | max rel | V4_1 | V3 vs V4_1 |
|---|---:|---:|---|---|
| 1024×1024×1024 | 6.079674e-06 | 1.020690e-05 | PASS | max abs=0, max rel=0, exact comparison PASS |
| 2048×2048×2048 | 1.084805e-05 | 3.015384e-04 | PASS | max abs=0, max rel=0, exact comparison PASS |
| 128×192×64 | 8.344650e-07 | 8.325396e-07 | PASS | max abs=0, max rel=0, exact comparison PASS |

64×96×32 不满足 V3/V4_1 的 M/N 64-multiple 条件，按原支持范围明确 skip，没有扩大支持范围。

每次 `checkKernel()` 都执行 NaN output fill、kernel launch、`cudaGetLastError()`、`cudaDeviceSynchronize()` 和 D2H copy。获准访问 GPU 后的所有正式运行均以 exit code 0 完成，没有 CUDA Runtime error。沙箱内第一次执行无法访问 WSL GPU driver，报 driver/runtime 不可用；在授权的沙箱外 GPU 会话中驱动 13.5 高于 runtime 13.2，问题消失，因此不是 V4_1 代码错误。

### 4.4 原有随机回归

重新编译并运行 `tests/v3_random_correctness.cu`，5 个原有随机用例全部通过，最终输出 `FINAL_RESULT=PASS`。该历史测试文件只测试 V2.1、V3 和 cuBLAS，因此它证明 V3 回归未被破坏，但不能冒充 V4_1 随机覆盖；V4_1 的覆盖来自上面的集成 `runCase()`。

## 5. PTX / SASS 证据

PTX 由以下命令生成：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=compute_120 -ptx \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_v4_1.ptx
```

V4_1 PTX 中实际出现：

```text
cp.async.cg.shared.global [...], [...], 16, 16;
cp.async.commit_group;
cp.async.wait_group 0;
```

`sm_120` cubin 的 V4_1 SASS 中实际出现：

```text
LDGSTS.E.BYPASS.128 [...], desc[...][...]
LDGSTS.E.BYPASS.128 [...], desc[...][...]
LDGDEPBAR
DEPBAR.LE SB0, 0x0
BAR.SYNC.DEFER_BLOCKING 0x0
```

V3 对应路径仍为：

```text
LDG.E.128 R..., desc[...][...]
LDG.E.128 R..., desc[...][...]
STS.128 [...], R...
STS.128 [...], R...
BAR.SYNC.DEFER_BLOCKING 0x0
```

因此，V4_1 的 128-bit global-to-shared copy 确实被 lowering 为 `LDGSTS`；V3 的独立 `LDG.E.128` 和 `STS.128` staging 在 V4_1 中消失。PTX 的 commit/wait 在 SASS 序列中体现为 `LDGDEPBAR` 和 `DEPBAR.LE`，没有保留名为 `cp.async.commit_group` 的 SASS 文本。

## 6. CUDA Event Benchmark

计时协议沿用现有 interleaved framework：默认 stream、每个测量轮 10 次 warmup、100 次 kernel 的平均延迟作为该轮原始值、5 个 rotating-order measurement rounds，最终取 5 轮中位数。V3、V4_1 和 cuBLAS 使用相同 allocations、输入和计时条件；下列 cuBLAS 均来自本轮同进程实测，不使用历史数据。

### 6.1 1024×1024×1024

| Implementation | 每轮原始平均延迟（ms） | Median（ms） | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.055765, 0.058425, 0.059902, 0.045208, 0.046625 | 0.055765 | 38.509443 |
| V4_1 | 0.049679, 0.047566, 0.047623, 0.047431, 0.047561 | 0.047566 | 45.147376 |
| cuBLAS | 0.040140, 0.039862, 0.050571, 0.042252, 0.039776 | 0.040140 | 53.500051 |

- V4_1 / V3 speedup：1.1724x
- V4_1 / cuBLAS performance ratio：0.8439

V3 和 cuBLAS 的 5 轮离散明显高于 V4_1，1024³ 任务很短。因此 1.1724x 是本轮中位数事实，但证据不足以声称它是跨运行稳定加速。

### 6.2 2048×2048×2048

| Implementation | 每轮原始平均延迟（ms） | Median（ms） | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.301219, 0.300319, 0.304356, 0.304346, 0.295407 | 0.301219 | 57.034562 |
| V4_1 | 0.315474, 0.317330, 0.316649, 0.315030, 0.318963 | 0.316649 | 54.255252 |
| cuBLAS | 0.285987, 0.289666, 0.292140, 0.286382, 0.281632 | 0.286382 | 59.989262 |

- V4_1 / V3 speedup：0.9513x
- V4_1 / cuBLAS performance ratio：0.9044
- 等价表述：V4_1 的 median latency 比 V3 高约 5.12%。

2048³ 的各版本轮次更集中，V4_1 在这个尺寸上没有获得性能收益。

## 7. Nsight Compute：2048³ 一致配置对照

V3 和 V4_1 分别用相同 9 个 section、相同 2048³ 输入、`--launch-count 1` 采集，共 35 passes：

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
  --kernel-name 'regex:<wmmaGemmV3-or-wmmaGemmV4_1>' \
  --launch-count 1 ...
```

实际分两次运行，并分别把占位符替换为 `wmmaGemmV3` 和
`wmmaGemmV4_1`，以确保每份报告只采一个 kernel：

- `build/ncu/v4_1_v3_2048.ncu-rep`
- `build/ncu/v4_1_v4_1_2048.ncu-rep`

### 7.1 核心指标

| Metric | V3 | V4_1 | 直接观测 |
|---|---:|---:|---|
| NCU replay duration | 325.600 us | 340.128 us | 只作 profiler 内方向比较，不替代 Event |
| Registers/thread | 78 | 72 | V4_1 少 6 |
| Theoretical occupancy | 50.00% | 58.33% | V4_1 提高 8.33 pp |
| Achieved occupancy | 46.00% | 52.47% | V4_1 提高 6.47 pp |
| Local/shared spilling requests | 0 / 0 B | 0 / 0 B | 无 spill |
| Executed warp instructions | 14,270,464 | 15,839,232 | V4_1 增加约 11.0% |
| Normal global-load instructions | 1,048,576 | 0 | V4_1 改用 fused async copy |
| Normal shared-store instructions | 1,048,576 | 0 | V4_1 改用 fused async copy |
| `LDGSTS` instructions | 0 | 1,048,576 | V4_1 async staging |
| Global load requests | 1,048,576 | 1,048,576 | 请求数相同 |
| Global load sectors | 16,777,216 | 17,825,792 | V4_1 多 1,048,576，+6.25% |
| LDSM instructions | 2,097,152 | 2,097,152 | Shared→fragment 工作量相同 |
| HMMA instructions | 4,194,304 | 4,194,304 | Tensor 工作量相同 |
| CTA barrier warp instructions | 1,048,576 | 1,048,576 | 两个版本相同 |
| `LDGDEPBAR` / `DEPBAR` | 0 / 0 | 524,288 / 524,288 | V4_1 commit/wait lowering |
| Issue Slots Busy | 9.20% | 9.73% | 小幅提高 |
| Active warps/scheduler | 5.53 | 6.38 | 随 occupancy 提高 |
| Eligible warps/scheduler | 0.15 | 0.15 | 没有改善 |
| Warp cycles/issued instruction | 58.43 | 63.14 | V4_1 更高 |
| Long Scoreboard | 18.847 | 29.080 | cycles/issued，V4_1 明显更高 |
| Short Scoreboard | 3.637 | 3.979 | cycles/issued |
| Barrier stall | 7.399 | 8.666 | cycles/issued |
| Tensor Pipe Active | 87.316% | 82.768% | V4_1 降低 4.55 pp |
| HMMA subpipe active | 87.320% | 82.765% | 与 Tensor 汇总一致 |
| DRAM throughput | 14.93% | 15.28% | 均未接近带宽饱和 |
| L1/TEX throughput | 75.00% | 61.64% | V4_1 更低 |
| L2 throughput | 62.75% | 79.42% | V4_1 更高 |

### 7.2 LDSM wavefront / bank conflict

Source-correlated active SASS instances汇总：

| Fragment side | V3 actual / ideal / excessive | V4_1 actual / ideal / excessive |
|---|---:|---:|
| A：`LDSM.16.M88.4` | 8,388,608 / 4,194,304 / 4,194,304 | 8,388,608 / 4,194,304 / 4,194,304 |
| B：`LDSM.16.MT88.4` | 4,194,304 / 4,194,304 / 0 | 4,194,304 / 4,194,304 / 0 |

因此 V4_1 没有改变 V3 的 Shared Layout，也没有改变 A/B LDSM 的冲突模式：A 仍为每 request 8 actual / 4 ideal，B 仍为 4/4。

Aggregate shared-load counter 为：

- V3：12,584,094 wavefronts，4,195,734 bank conflicts。
- V4_1：12,627,567 wavefronts，4,238,786 bank conflicts。

aggregate 的小差异来自 replay/归因口径；source-correlated A/B 结构完全相同。V4_1 还记录到 13,893,632 个 `LDGSTS` shared-side wavefronts，`op_ldgsts` data-bank-conflict counter 为 0；不要把这个专用 async-copy wavefront 口径与普通 `STS.128` 指令数直接等同。

### 7.3 Global sector 现象

Aggregate NCU 显示 V4_1 在 global load request 数不变时多出 6.25% sectors。Source counter 的 theoretical-sector attribution 把额外 sectors 定位在 B-side（目标寄存器映射为 R48）的 `LDGSTS` 静态实例，而 A-side 为 ideal。Source theoretical-sector 汇总与 aggregate hardware sector counter 不是同一口径，因此报告只把“aggregate 多 6.25%”和“source attribution 指向 B-side”作为两条分别成立的观测，不把它们强行做数值相加。

## 8. 收益、瓶颈与推测边界

### 8.1 已由源码确认

- V4_1 是 single buffer；commit 后立即 wait，没有 load/compute overlap。
- V3/V4_1 的线程映射、Shared Layout、WMMA 计算和写回完全一致。
- V4_1 支持范围没有扩大。

### 8.2 已由 PTX/SASS/NCU 实测

- PTX 生成 `cp.async`，SASS 生成 `LDGSTS`、`LDGDEPBAR`、`DEPBAR`。
- V4_1 降低 6 registers/thread、提高 occupancy、无 spill。
- 2048³ 的 HMMA、LDSM 和 CTA barrier 动态工作量与 V3 相同。
- V4_1 动态总指令更多、global sectors 多 6.25%、Long Scoreboard 和 Barrier stall 更高、Tensor active 更低。
- 2048³ Event median 实际慢约 5.12%。

### 8.3 尚待验证的机制假设

- V4_1 的 2048³ 回退与“立即 wait 没有隐藏 copy latency”、额外 dependency instructions、B-side sector 增量一致，但现有对照不能把 5.12% 回退精确拆分给单一因素。
- 1024³ 的加速可能受寄存器/occupancy 或短任务的频率与 wave 调度影响；因 V3/cuBLAS 原始轮次离散较大，需要独立复测才能确认稳定性。
- Double Buffering 可能用跨 tile overlap 降低 async-copy 等待，但也会增加 shared 使用、同步复杂度，并可能改变寄存器/occupancy；当前数据不能预先保证它会加速。

## 9. 是否进入 V4_2 Double Buffering

建议：可以进入，但应把它定义为严格单变量、可回退的下一阶段实验，而不是默认会优于 V3 的既定优化。

进入条件已经满足：V4_1 正确、对齐和同步语义成立、PTX/SASS 确认真实 async copy、无 spill、已有 V3/V4_1 基线和同配置 NCU 报告。V4_2 应继续冻结 V3，并至少要求：

1. 只增加 ping-pong Shared Buffer、下一 tile prefetch 和必要的 wait/barrier，不改 WMMA/映射/layout/写回。
2. 保持 V3 的尺寸支持边界，并重复 V3/V4_2 精确输出对照。
3. 重新检查 `LDGSTS`、wait group 深度、register/spill 和 static shared。
4. 用 1024³、2048³ 同轮 interleaved Event 数据判定，不使用 NCU duration 代替。
5. 复查 B-side global sectors 是否仍增加，以及 Long Scoreboard、Tensor active 是否真正改善。

若 V4_2 不能在更稳定的 2048³ 上超过 V3，或引入 spill/同步风险，应保留 V3 为默认最优实现并记录负结果。
