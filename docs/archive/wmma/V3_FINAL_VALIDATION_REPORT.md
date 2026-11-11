# Tensor Core GEMM V3 最终独立复验与性能分析

日期：2026-10-14
对象：V3 128-bit Vectorized Staging
结论范围：RTX 5070（Compute Capability 12.0）、FP16 A/B、FP32 accumulation/output，以及 V3 当前支持的 M/N 为 64 倍数、K 为 16 倍数的矩阵。

## 1. 最终结论

**V3 已达到作为本项目“当前最佳自研 Kernel”正式冻结的状态。** 建议以当前源码 SHA-256 `d944fadcf2f323c7036d4d343143291b5234885a722529a4f8b3517032dd21df` 冻结为后续研究基线，而不是宣称它对任意形状都已全局最优。

依据如下：

- 五个指定形状、五个固定随机种子和五类分布全部通过原阈值；覆盖正数、负数、零、接近零和混合符号输入。V2.1、V3、默认同精度 cuBLAS 均通过 FP16 量化输入对应的 FP32 CPU reference。
- 两个规模各用两个独立普通进程得到 10 个 Event 样本。V3 聚合中位数为 0.048239 ms（1024³，44.518 TFLOPS）和 0.306693 ms（2048³，56.017 TFLOPS）。
- V3 相对 V2.1 加速 2.3998x / 2.2004x；达到同轮默认 cuBLAS 的 96.43% / 95.46%。没有挑选最快样本，所有有效原始值均在本报告列出。
- 2048³ NCU 显示 Tensor/HMMA pipe active 87.53%，Global load 和 Shared store 动态指令较 V2.1 各减少 8x，总动态指令减少 91.02%，且没有 spill/local-memory traffic。
- 剩余 A-side LDSM conflict、18.64 cycles/issued 的 Long Scoreboard、78 registers/thread 和固定 Tile 仍值得做独立研究，但现有数据不足以否定冻结，也不足以证明某一项继续优化必然带来端到端收益。

冻结结论的边界：V3 没有通用边界处理；非 64×64×16 tile 整除形状由 runner 跳过。1024³ 短任务本轮 CV 为 13.27%，说明其绝对微秒级结果对升频/系统状态敏感，应继续使用多进程、交替顺序和完整原始值，而不能只报单轮最快值。

## 2. 源码与环境

本轮没有修改 `src/tensor_core_gemm.cu`、任何 CUDA Kernel、`README.md` 或历史报告。只新增独立正确性驱动 `tests/v3_random_correctness.cu` 和本报告。

| 项目 | 值 |
|---|---|
| Git HEAD | `24e910728dd67dbb36befe06019ffb2f16a0c258` |
| `src/tensor_core_gemm.cu` SHA-256 | `d944fadcf2f323c7036d4d343143291b5234885a722529a4f8b3517032dd21df` |
| Benchmark binary SHA-256 | `55e6d943da2260a569dfab191363f3397775ab96a7d82737ffbbea39384add10` |
| GPU | NVIDIA GeForce RTX 5070 12 GB，48 SM，CC 12.0 |
| Driver | 596.49 |
| CUDA Toolkit / nvcc | 13.2 / 13.2.86 |
| CUDA runtime / driver API | 13020 / 13020 |
| cuBLAS runtime | 130401（13.4.1） |
| Nsight Compute | 2026.1.1.0 |
| OS | WSL2，Linux 6.18.33.2 |
| 编译目标 | `sm_120`，`-O3 -lineinfo` |

编译仍报告 V3 为 78 registers/thread、4352 B static shared memory、1 barrier、0-byte stack、0 spill load/store。独立正确性驱动通过预处理重命名原 runner 的 `main` 并直接包含当前编译单元，因此使用的是完全相同的 V2.1/V3 Kernel、launch path、CPU reference 和误差检查；它只提供新的主机输入分布和 case 列表。

## 3. 正确性独立复验

Pass/Fail 阈值保持不变：

```text
abs_error <= 1e-2 + 1e-2 * abs(FP16-quantized-input FP32 reference)
```

下表中误差为 V3 对 reference 的结果。三个 GPU 实现的各项误差在本轮相同；每个 case 的 V2.1↔V3 和 V3↔cuBLAS 最大差异也均为 0。逐位相同是本次观测，不是接口要求或未来编译器保证。

| Shape | Seed / 分布 | 对原始 FP32 ref max abs / rel | 对量化 ref max abs / rel | V2.1 | V3 | cuBLAS |
|---|---|---:|---:|---|---|---|
| 128×192×64 | 1101 / signed uniform，注入 0/近零 | 7.585883e-4 / 3.329361e+0 | 9.536743e-7 / 3.972653e-3 | PASS | PASS | PASS |
| 256×256×256 | 2202 / positive uniform，注入 0 | 3.452301e-4 / 8.858374e-5 | 8.106232e-6 / 2.131083e-6 | PASS | PASS | PASS |
| 512×512×512 | 3303 / negative uniform，注入 0 | 5.631447e-4 / 8.068137e-5 | 3.242493e-5 / 4.110150e-6 | PASS | PASS | PASS |
| 512×1024×256 | 4404 / near-zero signed uniform | 6.754817e-9 / 6.210144e-3 | 2.910383e-11 / 9.180212e-6 | PASS | PASS | PASS |
| 1024×512×1024 | 5505 / clipped normal，注入 0/近零 | 4.638731e-4 / 1.468945e+2 | 4.053116e-6 / 1.097644e-1 | PASS | PASS | PASS |

接近零 reference 会放大最大相对误差，例如最后一行；判定仍使用原有混合绝对/相对阈值，没有放宽或针对数据分布特判。

## 4. 非 NCU 性能稳定性复验

### 4.1 协议

- FP16 A/B、FP32 C/compute，cuBLAS 使用 `cublasGemmEx`、`CUBLAS_COMPUTE_32F`、`CUBLAS_GEMM_DEFAULT`。
- 同一进程内三实现使用相同输入、device allocations、默认 stream。
- 每个实现每轮预热 10 次，计时 100 次，共 5 轮；每轮按 V2.1/V3/cuBLAS 三元素循环轮转执行顺序。
- 每个规模运行两个独立普通进程，聚合 10 个样本；偶数样本中位数为排序后第 5、6 项的平均值。
- 标准差和 CV 使用 10 点总体标准差。NCU replay 和 NCU 包装进程中的打印时间不参与本节。

另有一次 2048³ 启动因终端会话只取回 CPU-reference 阶段、没有获得任何五轮 timing 输出，在看到任何 timing 前即判为不可用；它没有参与聚合。下面是两个完整独立进程的全部有效数据，不存在按快慢筛选。

### 4.2 全部原始时间

#### 1024×1024×1024

| 进程 | 实现 | 五轮时间（ms） |
|---|---|---|
| P1 | V2.1 | 0.121852, 0.102810, 0.103092, 0.121393, 0.121660 |
| P1 | V3 | 0.060963, 0.047176, 0.063731, 0.045578, 0.061079 |
| P1 | cuBLAS | 0.046511, 0.056029, 0.056901, 0.041269, 0.041140 |
| P2 | V2.1 | 0.113222, 0.113831, 0.118482, 0.115073, 0.116450 |
| P2 | V3 | 0.047537, 0.054379, 0.045490, 0.048940, 0.046023 |
| P2 | cuBLAS | 0.040822, 0.056600, 0.042943, 0.051172, 0.046526 |

#### 2048×2048×2048

| 进程 | 实现 | 五轮时间（ms） |
|---|---|---|
| P1 | V2.1 | 0.670998, 0.673482, 0.675759, 0.673957, 0.677999 |
| P1 | V3 | 0.298850, 0.300396, 0.307696, 0.316034, 0.314017 |
| P1 | cuBLAS | 0.281222, 0.284721, 0.297045, 0.297900, 0.298137 |
| P2 | V2.1 | 0.692816, 0.681574, 0.678543, 0.667721, 0.666701 |
| P2 | V3 | 0.305689, 0.315203, 0.301422, 0.304336, 0.310909 |
| P2 | cuBLAS | 0.292428, 0.289512, 0.291149, 0.293782, 0.293124 |

### 4.3 十点聚合

| Shape / 实现 | Median（ms） | Min–Max（ms） | Mean ± σ（ms） | CV | Median TFLOPS |
|---|---:|---:|---:|---:|---:|
| 1024³ V2.1 | 0.115761 | 0.102810–0.121852 | 0.114786 ± 0.006640 | 5.78% | 18.551 |
| 1024³ V3 | 0.048239 | 0.045490–0.063731 | 0.052090 ± 0.006913 | 13.27% | 44.518 |
| 1024³ cuBLAS | 0.046518 | 0.040822–0.056901 | 0.047991 ± 0.006337 | 13.20% | 46.165 |
| 2048³ V2.1 | 0.674858 | 0.666701–0.692816 | 0.675955 ± 0.007176 | 1.06% | 25.457 |
| 2048³ V3 | 0.306693 | 0.298850–0.316034 | 0.307455 ± 0.006015 | 1.96% | 56.017 |
| 2048³ cuBLAS | 0.292776 | 0.281222–0.298137 | 0.291902 ± 0.005277 | 1.81% | 58.679 |

| Shape | V3 / V2.1 speedup | V3 / cuBLAS throughput | 上一报告 V3 median | 本轮 V3 latency 变化 | 上一报告达成率 |
|---|---:|---:|---:|---:|---:|
| 1024³ | 2.3998x | 96.43% | 0.045612 ms | +5.76%（吞吐 -5.45%） | 90.28% |
| 2048³ | 2.2004x | 95.46% | 0.307923 ms | -0.40%（吞吐 +0.40%） | 94.16% |

1024³ 的 V3 与 cuBLAS 同时高离散，且本轮 cuBLAS 中位延迟相对上一报告慢 12.97%；因此 1024³ 达成率上升不能全部归因于 V3 改善。2048³ 三个实现均较稳定，本轮 V3 与历史值基本一致，是更可信的性能复现点。

### 4.4 频率、温度和功耗

NVML 以 100 ms 采样。活动阶段观察到的 SM clock 约 2.65–2.87 GHz、温度 45–47°C、板卡功耗约 41–63 W，power limit 为 250 W。后置快照中 `sw_power_cap` 为 Active，但 `sw_thermal_slowdown`、`hw_thermal_slowdown` 和 `hw_power_brake_slowdown` 均为 Not Active；同时实测功耗远低于 250 W。

因此没有明显热限频或达到板卡功耗上限的证据。`sw_power_cap` 快照和短任务的频率跃迁意味着不能排除驱动电源管理/升频状态对 1024³ 离散的影响，也不能仅凭这些低采样率数据将波动准确归因。

## 5. 2048³ NCU 定向分析

新报告：`build/ncu_v3_final_2048.ncu-rep`，只采集 SpeedOfLight、ComputeWorkloadAnalysis、SchedulerStats、WarpStateStats、InstructionStats、LaunchStats、Occupancy、MemoryWorkloadAnalysis_Tables 和 SourceCounters，共 35 passes；没有再次使用 full set。

下表的 V2.1 来自已有 `build/ncu_v2_1_v3diag_2048.ncu-rep`。两者不是同一 profiler 会话，NCU duration 只作方向检查，不能代替 Event benchmark。

| 指标 | V2.1 2048³ | V3 2048³ | 直接观测 |
|---|---:|---:|---|
| Tensor pipe active（elapsed） | 31.22% | 87.53% | V3 为 2.80x |
| HMMA subpipe active（elapsed） | 31.22% | 87.52% | 与 Tensor 汇总一致 |
| Issue Slots Busy | 33.49% | 9.22% | V3 更低；不能单独解释为退化 |
| Active warps/scheduler | 7.30 | 5.54 | 随 occupancy 下降 |
| Eligible warps/scheduler | 0.55 | 0.15 | 可立即发射 warp 更少 |
| Warp cycles/issued instruction | 21.81 | 58.53 | issued 指令分母与指令混合已改变 |
| Long Scoreboard | 13.414 | 18.645 | cycles/issued instruction |
| Short Scoreboard | 1.165 | 3.623 | cycles/issued instruction |
| Barrier stall | 0.892 | 7.589 | cycles/issued instruction |
| Math-pipe throttle | 1.996 | 19.298 | V3 最大 stall 类之一，与 Tensor 高利用一致 |
| DRAM throughput（elapsed） | 4.05% | 14.94% | 更短执行时间内完成相同 sectors |
| L1/TEX throughput（active） | 33.84% | 75.11% | 利用率显著提高 |
| Global load instructions/requests | 8,388,608 | 1,048,576 | 减少 8x |
| Global load sectors | 16,777,216 | 16,777,216 | 数据字节量不变 |
| Shared store instructions | 8,388,608 | 1,048,576 | 减少 8x |
| LDSM instructions | 2,097,152 | 2,097,152 | fragment load 工作量不变 |
| Total executed instructions | 158,863,360 | 14,270,464 | 减少 91.02% |
| Registers/thread | 56 | 78 | 增加 22 |
| Theoretical occupancy | 75.00% | 50.00% | V3 受 registers 限制 |
| Achieved occupancy | 60.91% | 45.93% | 下降 14.98 pp |
| Local load/store | 0 / 0 | 0 / 0 | 无 spill traffic |
| NCU replay duration | 957.44 us | 324.35 us | 只作方向参考，不进入性能结论 |

### 5.1 A0/A1、B0/B1 LDSM

编译器将 K loop 展开为多组静态指令。把同一逻辑 fragment 的四个活跃静态实例相加后：

| Fragment load | Requests | Actual wavefronts | Ideal wavefronts | Excessive | 每 request actual / ideal |
|---|---:|---:|---:|---:|---:|
| A0 `LDSM.16.M88.4` | 524,288 | 4,194,304 | 2,097,152 | 2,097,152 | 8 / 4 |
| A1 `LDSM.16.M88.4` | 524,288 | 4,194,304 | 2,097,152 | 2,097,152 | 8 / 4 |
| B0 `LDSM.16.MT88.4` | 524,288 | 2,097,152 | 2,097,152 | 0 | 4 / 4 |
| B1 `LDSM.16.MT88.4` | 524,288 | 2,097,152 | 2,097,152 | 0 | 4 / 4 |

Source-correlated excessive wavefronts 合计 4,194,304，全部来自 A0/A1。Aggregate shared-load wavefronts 为 12,584,077、bank conflicts 为 4,195,531，与 source 汇总的 12,582,912 / 4,194,304 略有差异，属于 replay/source attribution 口径差异；结论不变。Shared store conflicts 为 0。

### 5.2 解释边界

**直接测量：** V3 在相同 Global sectors 和相同 LDSM 工作量下大幅减少 staging/总指令；Tensor/HMMA pipe 已接近九成 active；A-side conflict 仍存在；寄存器限制 occupancy；无 spill。

**合理解释：** V3 的主要收益与 128-bit staging 降低指令开销、提高 Tensor work 的供给一致。Issue Slots Busy 从 33.49% 降到 9.22% 并不与性能提升矛盾：V3 删除了大量可计入 issue 的标量搬运/地址指令，同时剩余 HMMA 是多周期管线工作。Long/Short Scoreboard 和 Barrier 的 per-issued 值也受到 issued 指令分母缩小影响，不能直接判为退化。

**尚未证实：** A padding 是否能把 conflict counter 的下降转化为端到端 latency；`cp.async`/双缓冲是否能在 Tensor pipe 已达 87.53% 时继续获得显著收益；降低寄存器是否能抵消 ILP 损失；不同 Tile 对非方阵的性能收益。本次 NCU 不足以回答这些反事实问题。

## 6. 后续研究方向的证据评级

| 方向 | 证据强度 | 当前证据 | 风险与验证方法 |
|---|---|---|---|
| A Shared Padding / swizzle | **中等**（冲突存在为强证据，性能收益未证实） | A0/A1 均为 8 actual / 4 ideal，合计 4,194,304 excessive；B0/B1 已是 4/4 | padding 可能增加 shared、地址计算或 store conflict。只做单变量版本；要求 A LDSM excessive 下降且普通 Event latency 改善超过本轮 1.96% CV，不能只看 conflict counter |
| `cp.async` / Double Buffering | **中等偏弱** | Long Scoreboard 仍为 18.645、eligible warps 仅 0.15；但 Tensor/HMMA active 已达 87.53%，不像 V2.1 阶段那样明显缺 Tensor work | 额外 shared、barrier/wait 复杂度及寄存器可能进一步压低 occupancy。需拆分 async copy 与 double buffer，检查 SASS、正确性、spill、Tensor active 和两规模 Event 中位数 |
| Registers / Occupancy 调整 | **中等偏弱** | 78 registers 将 theoretical occupancy 限为 50%，achieved 45.93%；但零 spill且 Tensor pipe 已高利用 | 强制 `maxrregcount` 可能 spill 或破坏 ILP。只接受 local traffic 仍为 0 且 Event latency 改善的版本，不以 occupancy 单指标决策 |
| 不同矩阵形状的 Tile 调整 | **中等（作为专用路径）** | 1024³ 短任务波动高；固定 64×64 CTA 并未对长宽比做性能验证。2048³ V3 有 3.56 waves/SM 和 partial wave | 本轮非方阵只验证正确性，尚无性能数据。应保留冻结的 64×64 默认路径，另测形状专用 64×32/32×64 等，并同时统计额外 Global/LDSM 工作量与多个 aspect ratio |

优先级建议：先把当前 V3 冻结；若继续研究，先做 A-layout 单变量实验，其次才是异步双缓冲；寄存器限制和形状专用 Tile 作为独立分支验证。任何候选都必须与本报告的冻结二进制进行同轮对照。

## 7. 复现命令与产物

### 编译

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu \
  -o build/tensor_core_gemm_v3_validation -lcublas

nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  tests/v3_random_correctness.cu \
  -o build/v3_random_correctness -lcublas
```

### 正确性与普通 Benchmark

```bash
./build/v3_random_correctness

# 每条各执行两个独立进程；不要与历史样本混合求中位数
./build/tensor_core_gemm_v3_validation 1024 1024 1024
./build/tensor_core_gemm_v3_validation 1024 1024 1024
./build/tensor_core_gemm_v3_validation 2048 2048 2048
./build/tensor_core_gemm_v3_validation 2048 2048 2048
```

### 2048³ 定向 NCU

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
  --kernel-name regex:wmmaGemmV3 \
  --launch-count 1 \
  -o build/ncu_v3_final_2048 \
  ./build/tensor_core_gemm_v3_validation 2048 2048 2048

ncu --import build/ncu_v3_final_2048.ncu-rep --page details
ncu --import build/ncu_v3_final_2048.ncu-rep \
  --page source --print-source sass
```

相关文件：

- 最终报告：`V3_FINAL_VALIDATION_REPORT.md`
- 独立正确性驱动：`tests/v3_random_correctness.cu`
- 正确性二进制：`build/v3_random_correctness`
- 正式 Benchmark 二进制：`build/tensor_core_gemm_v3_validation`
- 新 2048³ NCU：`build/ncu_v3_final_2048.ncu-rep`
- 既有 V3 SASS：`build/tensor_core_gemm_v3.sass`
- 既有 2048³ V2.1 NCU：`build/ncu_v2_1_v3diag_2048.ncu-rep`

本轮未修改或覆盖 `V3_VALIDATION_REPORT.md`、`V3_PREOPT_PERFORMANCE_DIAGNOSIS.md` 及其他历史报告，未清理 untracked 文件，未执行 Git commit。
