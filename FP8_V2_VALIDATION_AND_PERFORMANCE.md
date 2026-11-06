# FP8 Tensor Core GEMM V2 集成、验证与性能对照

验证日期：2026-11-03（Asia/Shanghai）

## 1. 结论

用户实现的 `fp8_gemm_v2()` 已完成独立工程集成、逐元素正确性验证、PTX/SASS
检查、7 轮交替 CUDA Event benchmark 和 1024³ Nsight Compute 对照。

- V1 的冻结 Kernel 未修改。当前文件中 V1 的前 145 行与 `fp8-v1-complete`
  tag 中的 V1 SHA-256 均为
  `f90d3ef0f60fe6045a49f574819ab6ba87f69a508978e387cc1331f0e2ac1104`。
- 8 个正确性案例中，V1/reference、V2/reference、V2/V1 的最大绝对误差和
  最大相对误差全部为 0；没有超容差、非有限值、未覆盖输出或 guard 破坏。
- PTX 保留 `ld.global.b32`；SASS 实际生成 32-bit `LDG.E`，没有退化为 4 条
  `LDG.E.U8`，并保留相同的 `QMMA.16832.F32.E4M3.E4M3`。
- 每个 K tile 的 fragment loads 从 24 条 byte load 降为 6 条 32-bit load，
  18 条 packing 指令降为 0；1024³ 动态 global loads 和 load sectors 均精确下降
  75%。
- 三个形状的 V2/V1 中位数加速分别为 3.145×、3.794×、3.967×，各轮均为
  正向提升，不是单次偶然结果。
- NCU 中 LG Throttle 从 57.32 降至 0.28 cycles/issued instruction，Tensor
  Pipe Active 从 4.03% 升至 15.48%。V2 的主要等待转为 Long Scoreboard
  79.53 cycles/issued instruction；DRAM Active 只有 2.70%，因此仍不能把 V2
  判断为 DRAM 带宽受限。

本轮没有加入 Shared Memory、多 Warp CTA、流水线或布局修改，没有继续实现其他
优化，也没有执行 Git commit 或 tag。

## 2. 工程变更与 Git 状态

### 2.1 本轮新增/修改

| 文件 | 状态与用途 |
|---|---|
| `src/fp8_gemm.cu` | 用户已有 V2；V1 原样保留，本轮未改 Kernel 核心 |
| `src/fp8_gemm.cuh` | 增加 V2 声明及与 V1 相同的 contract 说明 |
| `tests/fp8_gemm_v2_compare.cu` | 新增同输入正确性、V1/V2 交替 benchmark、独立 NCU workload |
| `CMakeLists.txt` | 增加独立 target `fp8_gemm_v2_compare` |
| `FP8_V2_VALIDATION_AND_PERFORMANCE.md` | 本报告 |

没有修改冻结的 `tests/fp8_gemm_v1_test.cu`，其 SHA-256 仍为
`5285632a81827fbe333263b53219817e723eb84eff8a4a18c33f34e1e511d17d`。

### 2.2 最终 Git 状态

```text
## main...origin/main [ahead 1]
 M README.md
 M src/fp8_gemm.cu
 M src/fp8_gemm.cuh
?? CMakeLists.txt
?? FINAL_GEMM_PERFORMANCE_REPORT.md
?? FP8_V1_PERFORMANCE_DIAGNOSIS.md
?? FP8_V1_VALIDATION_REPORT.md
?? V1_1_BANK_CONFLICT_REPORT.md
?? V1_PERFORMANCE_REPORT.md
?? V2_1_BANK_CONFLICT_REPORT.md
?? V2_PERFORMANCE_REPORT.md
?? V3_FINAL_VALIDATION_REPORT.md
?? V3_PREOPT_PERFORMANCE_DIAGNOSIS.md
?? V3_VALIDATION_REPORT.md
?? V4_1_VALIDATION_REPORT.md
?? V4_2_INSTRUCTION_ANALYSIS.md
?? V4_2_VALIDATION_REPORT.md
?? V4_3_VALIDATION_REPORT.md
?? tests/fp8_gemm_v2_compare.cu
```

当前 HEAD 为：

```text
4b3b0cb (HEAD -> main, tag: fp8-v1-complete)
docs: complete FP8 V1 baseline and profiling report
```

本轮应由用户决定保存的 V2 文件至少包括：

- `src/fp8_gemm.cu`
- `src/fp8_gemm.cuh`
- `tests/fp8_gemm_v2_compare.cu`
- `CMakeLists.txt`
- `FP8_V2_VALIDATION_AND_PERFORMANCE.md`

`README.md` 和既有未跟踪报告是进入本轮前就存在的用户工作，均未覆盖或删除。
两个原始 NCU report 位于 `build/ncu/`，该目录被 Git 忽略；如果要长期保留原始
profile，需要另行归档：

```text
build/ncu/fp8_v1_1024_v2_compare_20261103.ncu-rep  572851 bytes
build/ncu/fp8_v2_1024_compare_20261103.ncu-rep     312024 bytes
```

## 3. 环境与构建

| 项目 | 结果 |
|---|---|
| 环境 | WSL Ubuntu |
| GPU | NVIDIA GeForce RTX 5070，CC 12.0 / SM120 |
| CUDA Runtime 报告显存 | 12,792,365,056 bytes |
| CUDA Toolkit | 13.2，nvcc 13.2.86 |
| PTX ISA | 9.2 |
| Nsight Compute | 2026.1.1.0 |
| 编译目标 | `sm_120` |

受限 shell 中 `nvidia-smi` 本次仍报告 `Failed to initialize NVML`；GPU 型号和
架构来自成功运行的 CUDA Runtime 及 NCU 目标信息，不据此虚构 NVML 检查结果。

构建命令：

```bash
cmake -S . -B build/cmake-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-sm120 --target \
  fp8_gemm_v1_test fp8_gemm_v2_compare -j 4
```

结果：两个 target 均构建成功。ptxas 资源结果：

| Kernel | Registers/thread | Spill | Barrier |
|---|---:|---:|---:|
| V1 | 39 | 0 | 0 |
| V2 | 40 | 0 | 0 |

V2 只增加 1 个 register，没有 local-memory spill。

## 4. 测试方法与相同条件

比较程序对 V1/V2 使用完全相同的：

- 实际量化后的 E4M3 A/B 数据；
- A row-major、B column-major、C row-major；
- `grid(N/8, M/16)`、`block(32)`；
- 16×8 CTA output tile 和 32-element K tile；
- `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`；
- FP32 FMA reference 和误差判据；
- 每个 shape 的输入 buffer、固定 seed、预热数和计时迭代数。

正确性判据沿用 V1：

```text
abs_error <= 1e-4 + 1e-4 * abs(reference)
```

每个 Kernel launch 后检查 `cudaGetLastError()` 和 `cudaDeviceSynchronize()`；C
先填 NaN，前后各放 64 个 float guard。失败路径会报告 C 坐标、reference、V1、
V2、对应 lane、accumulator 和 K tile 数量。

## 5. 正确性结果

先单独运行冻结 V1 全套案例：

```bash
./build/cmake-sm120/fp8_gemm_v1_test --correctness
```

结果：8/8 PASS，所有案例 `max_abs=0`、`max_rel=0`、`over_tol=0`、
`nonfinite=0`、guard OK。

随后在同一测试入口中以相同输入依次运行 V1 和 V2：

```bash
./build/cmake-sm120/fp8_gemm_v2_compare --correctness
```

| Case | Shape | 输入 | V1/ref max abs/rel | V2/ref max abs/rel | V2/V1 max abs/rel | 结果 |
|---|---:|---|---:|---:|---:|---|
| zero_single_tile | 16×8×32 | 全零 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| sparse_single_tile | 16×8×32 | 稀疏正负 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| distinct_single_tile | 16×8×32 | 按位置变化 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| mixed_three_k_tiles | 16×8×96 | 正负混合 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_multi_cta | 32×16×64 | seed 3001 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_64 | 64×64×128 | seed 4001 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_128 | 128×128×256 | seed 5001 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_256 | 256×256×512 | seed 6001 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |

全部比较均 `over_tol=0`、`nonfinite=0`、V1/V2 guard OK，程序返回 0。没有
失败案例需要进行 lane、fragment、K tile 或写回定位。由于结果逐 bit 对应的 FP32
数值完全相同，通过结论不依赖容差是否宽松。

## 6. PTX、SASS 与每 K tile 指令

### 6.1 PTX

生成和检查命令：

```bash
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_v2_probe/fp8_gemm.ptx
rg -n 'ld.global.b32|mma.sync.*e4m3' \
  build/fp8_v2_probe/fp8_gemm.ptx
```

V2 PTX 每个源代码 K tile 含 6 个 32-bit load 和 1 个 MMA：

```text
ld.global.b32 %r..., [%rd...];        # A × 4, B × 2
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ...
```

### 6.2 SASS

检查命令：

```bash
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v2_compare
```

关键结果：

```text
# V1
LDG.E.U8 ...
IMAD ... / LEA ...                  # byte shift/OR packing 的机器码形式
QMMA.16832.F32.E4M3.E4M3 ...

# V2
LDG.E ...                            # 32-bit destination register
QMMA.16832.F32.E4M3.E4M3 ...
```

V2 的 `LDG.E` 没有 `.U8`，每条写入一个 32-bit register；SASS 中没有 V1 的
packing `IMAD/LEA` 序列。编译器把 V2 steady-state K loop 展开 4 倍，形成 24 个
`LDG.E` 配 4 个 QMMA 的循环体。这个展开是同一 `-O3` 编译器对用户 V2 源码的
结果，不是本轮新增的 Kernel 优化。

以下统计是 steady-state SASS，地址项包括为 fragment 地址构造/更新而执行的
`MOV/SHF/IADD/UIADD`，不含 Kernel 公共 prologue、C 写回和 remainder：

| 每个 K tile | V1 | V2 |
|---|---:|---:|
| Global fragment load | 24 × `LDG.E.U8` | 6 × `LDG.E` |
| Fragment packing | 18（16 `IMAD` + 2 `LEA`） | 0 |
| 地址构造/指针更新 | 8 | 4（4-way loop 平均） |
| QMMA | 1 | 1 |

1024³ 的 grid 有 8192 个 CTA，每 CTA 32 个 K tiles，因此共有 262,144 个
动态 K tiles/QMMA。NCU 动态计数严格对应上述结果：

| 动态 SASS | V1 | V2 | 变化 |
|---|---:|---:|---:|
| QMMA | 262,144 | 262,144 | 不变 |
| Global loads | 6,291,456 | 1,572,864 | -75.0% |
| 核心 fragment packing | 4,718,592 | 0 | -100% |
| 全部 warp instructions | 14,843,904 | 3,637,248 | -75.5% |

V1 每条 QMMA 需要 24 load + 18 packing，即 42 条直接供数指令；V2 变为 6 load
加 0 packing。V2 仍有地址/循环和写回指令，因此总动态指令不会恰好只剩 1/4。

## 7. CUDA Event 性能对照

命令：

```bash
./build/cmake-sm120/fp8_gemm_v2_compare --benchmark
```

协议：每个 shape 7 轮；奇数轮 V1→V2，偶数轮 V2→V1；每个版本在每一轮都独立
执行相同预热和计时。输入在该 shape 内共用，H2D、分配和预热不计入 CUDA Event。
TFLOPS 按 `2*M*N*K/time` 计算。

### 7.1 512³（每轮 10 warmups / 100 iterations）

| 轮次 | 顺序 | V1 ms | V2 ms | V2 TFLOPS | 加速 |
|---:|---|---:|---:|---:|---:|
| 1 | V1,V2 | 0.053682 | 0.016938 | 15.848 | 3.169× |
| 2 | V2,V1 | 0.053489 | 0.016495 | 16.274 | 3.243× |
| 3 | V1,V2 | 0.053244 | 0.017370 | 15.454 | 3.065× |
| 4 | V2,V1 | 0.051486 | 0.016484 | 16.285 | 3.123× |
| 5 | V1,V2 | 0.051495 | 0.016932 | 15.854 | 3.041× |
| 6 | V2,V1 | 0.054652 | 0.017460 | 15.374 | 3.130× |
| 7 | V1,V2 | 0.051321 | 0.016511 | 16.258 | 3.108× |
| **中位数** | — | **0.053244** | **0.016932** | **15.853** | **3.145×** |

V1 中位数为 5.042 TFLOPS。

### 7.2 1024³（每轮 10 warmups / 50 iterations）

| 轮次 | 顺序 | V1 ms | V2 ms | V2 TFLOPS | 加速 |
|---:|---|---:|---:|---:|---:|
| 1 | V1,V2 | 0.385372 | 0.101002 | 21.262 | 3.815× |
| 2 | V2,V1 | 0.387470 | 0.100695 | 21.327 | 3.848× |
| 3 | V1,V2 | 0.383336 | 0.108873 | 19.725 | 3.521× |
| 4 | V2,V1 | 0.383339 | 0.101033 | 21.255 | 3.794× |
| 5 | V1,V2 | 0.387780 | 0.101306 | 21.198 | 3.828× |
| 6 | V2,V1 | 0.382493 | 0.105051 | 20.442 | 3.641× |
| 7 | V1,V2 | 0.383293 | 0.100966 | 21.269 | 3.796× |
| **中位数** | — | **0.383339** | **0.101033** | **21.255** | **3.794×** |

V1 中位数为 5.602 TFLOPS。

### 7.3 2048³（每轮 5 warmups / 20 iterations）

| 轮次 | 顺序 | V1 ms | V2 ms | V2 TFLOPS | 加速 |
|---:|---|---:|---:|---:|---:|
| 1 | V1,V2 | 2.995816 | 0.759979 | 22.606 | 3.942× |
| 2 | V2,V1 | 3.023547 | 0.759363 | 22.624 | 3.982× |
| 3 | V1,V2 | 3.004579 | 0.759088 | 22.632 | 3.958× |
| 4 | V2,V1 | 3.012235 | 0.769221 | 22.334 | 3.916× |
| 5 | V1,V2 | 3.023494 | 0.749749 | 22.914 | 4.033× |
| 6 | V2,V1 | 3.005847 | 0.757520 | 22.679 | 3.968× |
| 7 | V1,V2 | 3.035291 | 0.760285 | 22.597 | 3.992× |
| **中位数** | — | **3.012235** | **0.759363** | **22.624** | **3.967×** |

V1 中位数为 5.703 TFLOPS。

### 7.4 冻结基线说明

`FP8_V1_PERFORMANCE_DIAGNOSIS.md` 中冻结的跨会话 V1 Event 数值为 0.243956、
2.296136、18.061241 ms，本轮没有改写这些历史结果。当前同进程交替测试中的 V1
绝对时间明显更快，说明机器功耗/时钟/会话状态会显著影响跨会话绝对值；历史 NCU
中的 V1 profiler duration 约 0.42 ms 也曾明显快于当时的 Event 结果。

因此本报告的 V2 加速结论只使用本轮同一进程、相同输入、交替顺序的配对结果，
不拿 V2 当前值除以旧报告的 V1 绝对值。所有性能表中的时间都是 CUDA Event，
没有使用 NCU replay Duration。

## 8. 1024³ Nsight Compute 对照

### 8.1 采集方法

V1/V2 使用相同 section、输入生成和 `--cache-control none`。独立 profile workload
各发射 10 次 1024³，跳过前 5 次，采 1 次；每份报告完成 35 replay passes：

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
  --kernel-name-base function --kernel-name fp8_gemm_v1 \
  --launch-skip 5 --launch-count 1 --kill yes \
  --cache-control none --force-overwrite \
  -o build/ncu/fp8_v1_1024_v2_compare_20261103 \
  ./build/cmake-sm120/fp8_gemm_v2_compare --profile-v1

# V2 命令只把 kernel-name、输出名和 mode 改为：
# fp8_gemm_v2 / fp8_v2_1024_compare_20261103 / --profile-v2
```

目标 launch 均为 `grid=(128,64,1)`、`block=(32,1,1)`，8192 CTAs。
`--cache-control none` 用于贴近重复 Event 的 cache-hot 条件，但 NCU 明确警告多次
replay 的 cache 状态不受控制。因此 V2 表格中的 replay-derived L2 hit rate
101.74% 是无效结果，不用于结论；指令计数、global sectors 和主要 stall 转移仍
相互一致。

### 8.2 SOL、cache、memory 和 occupancy

| 指标 | V1 | V2 | 解释 |
|---|---:|---:|---|
| NCU Duration | 419.52 us | 112.80 us | 仅 profiler 环境，不作 Event 时间 |
| L1/TEX throughput / active | 98.23% | 98.88% | 两者都接近 L1 数据路径上限 |
| Memory throughput / elapsed | 97.04% | 94.09% | 由 L1/TEX 主导，不等于 DRAM |
| L2 throughput | 15.08% | 51.73% | V2 在更短运行期内请求率更高 |
| DRAM active | 1.50% | 2.70% | 都远未达到 DRAM 带宽上限 |
| DRAM bytes/s | 10.09 GB/s | 18.13 GB/s | V2 更快完成，瞬时速率上升 |
| L1 overall hit | 93.36% | 75.15% | V1 的重复 byte loads 制造了额外 hit |
| L1 global-load hit | 93.57% | 76.13% | V2 去掉重复请求后 hit rate 降低属预期 |
| Global-load sectors | 50,331,648 | 12,582,912 | 精确 -75% |
| Global-load instructions | 6,291,456 | 1,572,864 | 精确 -75% |
| Useful bytes/load sector | 4 / 32 | 16 / 32 | 12.5% → 50% sector utilization |
| Tensor Pipe Active / elapsed | 4.03% | 15.48% | 3.84× |
| Registers/thread | 39 | 40 | +1，无 spill |
| Theoretical occupancy | 50.00% | 50.00% | 不变 |
| Achieved occupancy | 47.76% | 48.73% | 基本不变 |

32-bit load 没有把 sector utilization 提到 100%，原因是 fragment lane 映射下一个
warp 的 32 个 lane 仍覆盖 8 个 32-byte sectors；每条 V2 load 取得 128 useful
bytes，平均每 sector 使用 16 bytes。但相同 sectors 不再被四条独立 byte-load
指令重复请求，因此请求数和 sectors 均下降到 1/4。

V2 的 L1 hit rate 下降不是退化证据：V1 的四条 byte loads 对相邻字节重复触达同一
cache line，后三次更容易命中；V2 用一条 32-bit load 合并后消除了这类冗余 hit。

### 8.3 Scheduler 与 warp stall

| 指标 | V1 | V2 |
|---|---:|---:|
| Active warps / scheduler | 5.74 | 5.86 |
| Eligible warps / scheduler | 0.20 | 0.07 |
| No eligible cycles | 92.80% | 93.01% |
| Warp cycles / issued instruction | 79.66 | 83.79 |
| LG Throttle | 57.32 | 0.28 |
| Long Scoreboard | 17.09 | 79.53 |
| Wait | 1.85 | 2.03 |
| Not selected | 1.74 | 很低 |

V1 的主要问题是过多细粒度 LSU/L1 请求导致 LG Throttle。V2 的宽加载把该项几乎
完全消除，同时把 18 条 packing 移走，使 QMMA 发射密度和 Tensor Pipe Active
大幅提高。V2 仍有约 5.86 个 active warps/scheduler，却只有 0.07 个 eligible
warps，新的主导等待是 global load dependency 的 Long Scoreboard。

这表明 V2 已完成预期目标，但仍是“直接 global load → QMMA、数据复用和延迟隐藏
不足”的 V1 架构延续。DRAM Active 仅 2.70%，不能把 Long Scoreboard 简化为
DRAM 带宽饱和；它表示在等待 L1TEX/global-memory 依赖返回，而不是证明 DRAM
吞吐达到上限。

## 9. 理论预测与实测区分

### 9.1 修改前的理论预测

仅根据源代码替换可以确定：

- 每 K tile A/B load 数应从 `(4+2)×4=24` 降到 `4+2=6`；
- 手工移位/OR packing 应从 18 条机器指令附近降到 0；
- QMMA 数、计算量、layout 和写回应保持不变；
- 最理想情况下供数指令数减少约 7×，但端到端不应直接假定获得 7×，因为仍有
  地址计算、global-load latency、QMMA、loop 和 store。

### 9.2 实测

- SASS 和动态 NCU 计数完全验证了 24→6 loads、18→0 packing、QMMA 不变。
- Global load sectors 同步下降 75%，不是只有指令变少而 memory transaction 不变。
- Event 中位数实际提升 3.15×到 3.97×，小于直接供数指令减少倍数，符合剩余
  global-load dependency 和固定开销的预期。
- 瓶颈实测从 LG Throttle 转到 Long Scoreboard；因此不能继续假设再做同类 packing
  优化还能获得相同幅度收益。

## 10. 当前 V2 局限与后续方向（本轮未实现）

V2 保留 V1 的全部功能限制：仅紧密 A row-major / B column-major / C row-major，
M/N/K 分别要求 16/8/32 的倍数，无边界 tile、scale、batch、任意 leading
dimension 或融合 epilogue。一个 CTA 仍只有一个 warp、输出 16×8；A/B 跨 CTA
没有数据复用。

若进入下一阶段，最值得验证的方向不是继续做 packing，而是针对已显现的 Long
Scoreboard 设计延迟隐藏/数据复用实验，例如扩大 CTA 的 warp/tile 复用，或使用
shared-memory staging。代价会是 shared memory、更多 registers、同步和潜在
occupancy 下降。应一次只引入一种机制，与本报告的 V2 做相同输入、交替 Event 和
同一 NCU 指标对照。本轮按要求没有实施这些优化。

## 11. 可复现命令摘要

```bash
# 构建
cmake --build build/cmake-sm120 --target \
  fp8_gemm_v1_test fp8_gemm_v2_compare -j 4

# 冻结 V1 与同输入 V1/V2 正确性
./build/cmake-sm120/fp8_gemm_v1_test --correctness
./build/cmake-sm120/fp8_gemm_v2_compare --correctness

# 七轮交替 CUDA Event
./build/cmake-sm120/fp8_gemm_v2_compare --benchmark

# PTX/SASS
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_v2_probe/fp8_gemm.ptx
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v2_compare

# 读取保存的 NCU 报告
ncu --import build/ncu/fp8_v1_1024_v2_compare_20261103.ncu-rep
ncu --import build/ncu/fp8_v2_1024_compare_20261103.ncu-rep
```
