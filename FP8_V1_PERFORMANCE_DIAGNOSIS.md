# FP8 Tensor Core GEMM V1 性能瓶颈诊断

诊断日期：2026-11-03（Asia/Shanghai）

## 1. 结论

FP8 V1 在 1024³ 上的首要瓶颈不是 DRAM 带宽，也不是 Tensor Core 吞吐上限，
而是 **24 条细粒度 `LDG.E.U8`/K-tile 造成的 L1/LSU 请求压力，以及紧随其后的
18 条寄存器 packing 指令**。

主要证据：

- L1/TEX throughput 98.24%，但 DRAM throughput 仅 1.79%，L2 throughput 15.23%。
- Global load 每个 32-byte sector 平均只使用 4 bytes，即 12.5% sector 有效利用率。
- NCU 统计 50,593,792 个 global theoretical sectors，其中 44,171,264 个
  excessive，约占 87%。
- 每个 scheduler 平均有 5.74 个 active warps，却只有 0.20 个 eligible warps；
  92.80% cycles 没有 eligible warp。
- 每条 issued instruction 对应 79.66 warp cycles，其中 LG Throttle 为 57.00
  cycles（71.6%），Long Scoreboard 为 17.47 cycles（21.9%）。
- Tensor pipe active 仅 4.01%，Math Pipe Throttle 仅 0.02 cycles/issued inst。
- 262,144 条动态 QMMA 对应 6,291,456 条 global byte loads 和 4,718,592 条
  fragment packing 指令；每条 QMMA 需要 42 条直接供数指令。

因此 V2 的首选方向应是：**在保持相同 fragment 映射的前提下，将每四个连续
E4M3 元素直接作为一个对齐 32-bit word 加载到 fragment register**。这一步应先于
shared-memory tiling、`cp.async` 或 double buffering，以最小改动直接验证当前诊断。

本轮没有修改 `fp8_gemm_v1()` 或启动任何 V2 实现，也没有创建 Git commit。

## 2. 冻结基线

### 2.1 正确性

冻结结果来自 `FP8_V1_VALIDATION_REPORT.md`：

- 16×8×32：全零、稀疏、distinct-value 全部 PASS。
- 16×8×96、32×16×64、64×64×128、128×128×256、256×256×512 全部 PASS。
- 所有案例 max absolute error = 0、max relative error = 0。
- `over_tol=0`、`nonfinite=0`、C guard 完整。

### 2.2 CUDA Event

以下结果保持为 V1 冻结性能基线。本轮没有重新解释 profiler replay 时间为
benchmark 时间。

| 形状 | 预热 | 计时迭代 | 平均延迟 | TFLOPS |
|---:|---:|---:|---:|---:|
| 512³ | 10 | 100 | 0.243956 ms | 1.100343 |
| 1024³ | 10 | 50 | 2.296136 ms | 0.935260 |
| 2048³ | 5 | 20 | 18.061241 ms | 0.951201 |

### 2.3 冻结文件哈希

本轮诊断前记录的 SHA-256：

```text
f90d3ef0f60fe6045a49f574819ab6ba87f69a508978e387cc1331f0e2ac1104  src/fp8_gemm.cu
0a0f977206ec6a47304be75334ecd07fb1d6bf173ba42de9dc33e96183939e9b  src/fp8_gemm.cuh
5285632a81827fbe333263b53219817e723eb84eff8a4a18c33f34e1e511d17d  tests/fp8_gemm_v1_test.cu
7cdc8988c167dfe308379141db23a61a6681e7447dc7c0633ba45822f676a5c4  FP8_V1_VALIDATION_REPORT.md
124f1bf74459f14d38daeace9da5c90effee3e028a0bd265935230a498eb7bdf  CMakeLists.txt
b27bf5e8347dd186beb645c536321bbad4a46b74da947c1b01adfd3fc4601349  README.md
```

## 3. Git 文件状态

诊断开始时：

```text
## main...origin/main [ahead 3]
 M README.md
?? CMakeLists.txt
?? FP8_V1_VALIDATION_REPORT.md
?? src/fp8_gemm.cu
?? src/fp8_gemm.cuh
?? tests/fp8_gemm_v1_test.cu
```

此外还有多份此前已有的未跟踪 FP32/FP16 实验报告，与本轮 FP8 诊断无关，未改动。

在本轮最后核验时，Git 状态已被外部操作更新为：

```text
9d8b55e (HEAD -> main, tag: fp8-v1, origin/main)
feat: add validated FP8 Tensor Core GEMM V1
```

该提交只新增了：

```text
src/fp8_gemm.cu
tests/fp8_gemm_v1_test.cu
```

本诊断没有执行任何 `git add`、`commit`、`tag`、`push` 或 `reset`。当前工作树仍有：

```text
 M README.md
?? CMakeLists.txt
?? FP8_V1_VALIDATION_REPORT.md
?? FP8_V1_PERFORMANCE_DIAGNOSIS.md
?? src/fp8_gemm.cuh
```

需要特别注意：提交中的 `src/fp8_gemm.cu` 和测试程序都包含
`#include "fp8_gemm.cuh"`，但 `src/fp8_gemm.cuh` 尚未被提交。因此当前 `fp8-v1`
tag 单独 checkout 后缺少必需头文件，尚不是完整可复现的冻结基线；CMake、验证报告
和 README 集成也仍在工作树中。

后续若由用户手动保存基线到 Git，FP8 路径至少需要包含：

- `src/fp8_gemm.cu`
- `src/fp8_gemm.cuh`
- `tests/fp8_gemm_v1_test.cu`
- `CMakeLists.txt`
- `README.md`
- `FP8_V1_VALIDATION_REPORT.md`
- `FP8_V1_PERFORMANCE_DIAGNOSIS.md`

`src/fp8_gemm.cu` 和测试程序现在已 tracked；其余上述文件仍需由用户决定如何保存。

原始 NCU report 位于：

```text
build/ncu/fp8_v1_1024_diagnosis_20261103.ncu-rep
```

大小 281,411 bytes。`build/` 和 `*.ncu-rep` 已被 `.gitignore` 排除，不会出现在
Git 提交中；如需长期保存原始 profile，应单独归档。

## 4. Nsight Compute 采集方法

环境：RTX 5070 / CC 12.0、CUDA 13.2、Nsight Compute 2026.1.1。

测试程序的 `--benchmark` 顺序为 512³、1024³、2048³。512³ 有 10 次预热和
100 次计时 launch。为采到已经预热的 1024³ kernel，跳过：

```text
110 个 512³ launch + 5 个 1024³ warmup launch = 115
```

实际命令：

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
  --kernel-name-base function \
  --kernel-name fp8_gemm_v1 \
  --launch-skip 115 --launch-count 1 --kill yes \
  --cache-control none --force-overwrite \
  -o build/ncu/fp8_v1_1024_diagnosis_20261103 \
  ./build/fp8_gemm_v1_test --benchmark
```

采集成功，共 36 replay passes。目标 launch：

```text
grid=(128,64,1), block=(32,1,1), grid size=8192
registers/thread=39, shared memory=0, CC=12.0
```

使用 `--cache-control none` 是为了接近重复 CUDA Event benchmark 的 cache-hot 状态。
NCU 明确警告 uncontrolled caches 可能使跨 replay 指标不一致。特别是 tables 中个别
分项 L2 hit-rate 超过 100%，不能作为有效百分比；本报告只采用 summary hit rate、
请求/sector 计数和相互一致的主要 stall 指标。

NCU 报告的 Duration 为 421.54 us。该数值来自多 pass profiler 环境，且 cache/clocks
行为与 CUDA Event 路径不同，因此不用于更新或替代 2.296136 ms 的冻结基线。

## 5. Speed of Light 与 pipeline

| 指标 | 1024³ NCU 结果 |
|---|---:|
| Memory throughput | 96.69% |
| L1/TEX throughput | 98.24% |
| L2 throughput | 15.23% |
| DRAM throughput | 1.79% |
| DRAM bytes/s | 12.00 GB/s |
| Compute (SM) throughput | 24.18% |
| Tensor pipe active / elapsed | 4.01% |
| Tensor pipe active / active | 4.08% |
| LSU pipe | 24.18% |
| Issue slots busy | 7.08% |

`Compute (SM) throughput=24.18%` 不是 Tensor Core 利用率。NCU breakdown 表明它由
LSU pipe 的 24.18% 主导，Tensor pipe 只有 4.01%。Memory SOL 的 96.69% 也不是
DRAM 带宽，而是 L1 data-pipe wavefront 达到 96.69%。

因此不能把低 TFLOPS 归因于 DRAM：DRAM 只达到 1.79%，而请求压力已经在更靠近
SM 的 L1/LSU 路径达到极限。

## 6. Global memory 效率与 cache 行为

### 6.1 请求与 sector

| 指标 | 结果 |
|---|---:|
| Dynamic global-load warp instructions | 6,291,456 |
| L1 global-load requests | 6,291,456 |
| L1 global-load sectors | 50,331,648 |
| Sectors per load request | 8 |
| Average useful bytes per load sector | 4 / 32 bytes |
| Global-load sector utilization | 12.5% |
| Average useful bytes per store sector | 16 / 32 bytes |
| Total theoretical global sectors | 50,593,792 |
| Ideal global sectors | 6,422,528 |
| Excessive global sectors | 44,171,264（约 87.3%） |

一条 `LDG.E.U8` 是 warp-level 指令：32 lanes 共请求 32 useful bytes，但 lane 映射
把它们分布到 8 个 32-byte sectors，即请求 256 bytes 的 sector capacity，只使用
其中 32 bytes。

当前每个 lane 的四个连续 FP8 元素由四条独立 load 读取。即使四条指令最终访问
同一批 cache lines，coalescing 也不会跨四条独立指令合并，所以同一 sectors 会被
重复请求四次。

### 6.2 Cache

| 指标 | 结果 |
|---|---:|
| L1/TEX overall hit rate | 93.35% |
| L1/TEX global-load hit rate | 93.40% |
| L2 summary hit rate | 90.82% |
| L2 sectors | 3,797,164 |
| Local-memory spill requests | 0 |
| Shared-memory spill requests | 0 |

高 L1/L2 hit rate 和低 DRAM throughput 表明大部分数据已被 cache 捕获；问题是每个
K tile 产生了过多低宽度请求和 wavefront，而不是 miss 后从 DRAM 搬运的数据量。

## 7. Scheduler 与 Warp State

| 指标 | 结果 |
|---|---:|
| Theoretical occupancy | 50.00% |
| Achieved occupancy | 47.80% |
| Active warps / scheduler | 5.74 |
| Eligible warps / scheduler | 0.20 |
| Cycles with one or more eligible warps | 7.20% |
| Cycles with no eligible warp | 92.80% |
| Issued warp / scheduler / cycle | 0.07 |
| Warp cycles / issued instruction | 79.66 |

主要 warp state：

| Stall reason | cycles / issued instruction | 占 79.66 cycles |
|---|---:|---:|
| LG Throttle | 57.00 | 71.6% |
| Long Scoreboard | 17.47 | 21.9% |
| Wait | 1.85 | 2.3% |
| Not Selected | 1.74 | 2.2% |
| MIO Throttle | 0.28 | 0.4% |
| Math Pipe Throttle | 0.02 | <0.1% |

Source Counter sampling 也得到相同结论：25,173 个 samples 中 LG Throttle 为
17,828，Long Scoreboard 为 5,633。

V1 一个 block 只有一个 warp，硬件的 24 blocks/SM 上限把 theoretical active
warps 限制在 24/SM，即 50% occupancy。实际 47.80% 已接近这个理论值。增加更多
active warps 可能改善 latency hiding，但当前 L1 请求队列已经接近饱和，简单增加
warps 也可能加重 LG Throttle。因此 occupancy 不是首先需要解决的根因。

## 8. SASS 与动态指令关系

### 8.1 编译器是否合并 load/packing

K loop 中可见 24 条静态：

```text
LDG.E.U8
```

分别对应每 lane 的 16 个 A bytes 和 8 个 B bytes。编译器没有把每四个连续 byte
loads 合并为 `LDG.E.32`。

C++ 的 shift/OR 表达式没有逐字翻译为一串独立 SHL 和 OR。ptxas 把 packing 合并
为 18 条 `IMAD`/`LEA`，例如：

```text
IMAD R11, R10, 0x100, R11
IMAD R11, R34, 0x10000, R11
IMAD R20, R20, 0x1000000, R11
LEA  R36, R29, R30, 0x8
```

这说明算术 packing 已做 peephole 优化，但 load 宽度没有优化。无论用 IMAD 还是
shift/OR，packing 本身仍要占据 issue slots，并依赖 byte-load 返回。

### 8.2 动态指令

1024³ 有 8192 CTAs，每个 CTA 执行 32 个 K tiles，因此：

```text
8192 * 32 = 262,144 K-tile warp iterations
```

关键动态 warp-level 指令：

| 指令/类别 | 动态数量 | 占全部 14,843,904 |
|---|---:|---:|
| `LDG.E.U8` | 6,291,456 | 42.38% |
| Fragment packing（18/K-tile） | 4,718,592 | 31.79% |
| `QMMA.16832.F32.E4M3.E4M3` | 262,144 | 1.77% |
| `STG.E` | 32,768 | 0.22% |
| 全部 warp instructions | 14,843,904 | 100% |

基本 opcode 的完整大项包括：

```text
LDG.E.U8       6,291,456
IMAD           4,251,648
IADD.64        1,376,256
UIADD3           786,432
LEA              532,480
IADD             286,720
BRA              270,336
SHF.R.S32.HI     262,144
ISETP.LE.AND     262,144
QMMA             262,144
STG.E             32,768
```

每次 QMMA 前有：

```text
24 scalar global loads + 18 packing instructions = 42 feed instructions
```

仅这两类供数指令就占全部动态 warp instructions 的 74.17%。包含地址计算和 loop
control 后，平均每条 QMMA 对应约 56.6 条总指令。Tensor Core 大部分时间并没有
可发射的 QMMA，而是在等待和构造它的 6 个输入寄存器。

## 9. 瓶颈判定

按优先级排列：

1. **L1/LSU request-path saturation。** 24 条 byte loads/K-tile、12.5% sector
   利用率，使 L1/TEX 达到 98.24%，产生 71.6% LG Throttle。
2. **Load-to-pack dependency chain。** Packing 依赖多个 byte loads，Long
   Scoreboard 占 21.9%，eligible warps 仅 0.20/scheduler。
3. **整数 packing 和地址指令开销。** 18 条 packing/K-tile 占动态指令的
   31.79%，显著稀释 QMMA issue rate。
4. **单 warp CTA 的 occupancy 上限。** 理论 occupancy 只有 50%，会降低 latency
   hiding，但在 L1 队列已经饱和时不是首要矛盾。

以下假设与数据不符：

- **不是 DRAM-bandwidth bound：** DRAM throughput 仅 1.79%，12 GB/s。
- **不是 Tensor-pipe bound：** Tensor active 仅 4.01%，Math Pipe Throttle 几乎为 0。
- **不是 register spill：** 39 registers/thread，local/shared spill 都为 0。
- **不是 branch divergence：** branch efficiency 100%，divergent branches=0。

## 10. V2 优化候选与对照实验

### 候选 1：对齐的 32-bit fragment load（首选）

保持现有 lane/fragment 映射，每个 `a_frag[r]`/`b_frag[r]` 直接加载四个连续 E4M3
bytes，而不是四次 `.__x` 和手工 packing。

预计效果：

- 每 K tile global loads 从 24 条降到 6 条。
- 18 条 packing 指令理论上可全部消除。
- 每条 32-bit warp load 的 useful bytes/sector 预计从 4 提高到 16，sector 利用率
  从 12.5% 提高到约 50%。
- 同样 128-byte fragment 数据所需 L1 sector requests 预计下降约 4×。
- 直接降低 LG Throttle、Long Scoreboard 和总动态指令，并提高 QMMA issue 比例。

资源/风险：

- 必须显式保证 4-byte alignment 和合法 aliasing；当前 K 是 32 的倍数，fragment
  column/row offset 是 4 的倍数，地址条件满足，但实现仍需使用安全的类型/加载方式。
- 预计寄存器不增加，甚至可能因移除 byte temporaries 而减少；需要以 ptxas 验证。
- 需要重新跑所有 distinct/random correctness，防止 byte order 变化。

对照实验：

1. 只改变 load/packing，不改变 grid、block、tile、MMA 或 writeback。
2. 重跑 V1 全部正确性和 512³/1024³/2048³ Event。
3. 对 1024³ 采同一 NCU sections。
4. 重点比较 `LDG` 数、global sectors、bytes/sector、LG Throttle、Long
   Scoreboard、Tensor pipe active、register count。

### 候选 2：多 warp CTA + shared-memory 跨输出 tile 复用

例如让一个 CTA 的多个 warps 计算相邻 M/N tiles，并共同 staging A/B，使相同 A tile
在多个 N tiles 间复用、相同 B tile 在多个 M tiles 间复用。

预计效果：

- 降低每个输出元素对应的 global requests。
- 多 warp/block 可绕开 24 blocks/SM 导致的 50% warp occupancy 上限。
- 将供数压力从 global/L1 path 转移到 shared-memory path。

资源/风险：

- 增加 shared memory、barrier、地址和 warp mapping 复杂度。
- 可能引入 shared bank conflicts，并提高 register/CTA 资源占用。
- 当前数据已高比例命中 L1/L2；若 shared 设计只减少 DRAM bytes 而未减少 LSU/L1
  请求，收益可能不及候选 1。

对照实验：

- 应以完成候选 1 后的版本为基线，而不是同时混入两个变化。
- 固定每 warp 的 16×8 MMA，分别测试 2-warps 和 4-warps CTA。
- 比较 global sectors/QMMA、shared transactions、bank conflicts、barrier stall、
  theoretical/achieved occupancy 和 Event latency。

### 候选 3：vector-load 后的软件流水

在候选 1 降低请求数后，预取下一 K tile 的 A/B fragment，与当前 QMMA 交错，尝试
隐藏剩余 load latency。

预计效果：

- 主要针对 Long Scoreboard，而不是 DRAM bandwidth。
- 增加 eligible warps/instructions，并提高 Tensor pipe active。

资源/风险：

- 需要额外一组 A/B fragment registers、地址和 loop state，可能提高 registers/thread。
- 若在 scalar-load V1 上直接做，更多 outstanding loads 可能进一步恶化 LG Throttle；
  因此不应作为第一步。
- register 增长可能降低 occupancy，需与 latency hiding 收益权衡。

对照实验：

- 以候选 1 为唯一基线，增加单级 prefetch，暂不引入 shared memory。
- 比较 ptxas register count、occupancy、Long Scoreboard、LG Throttle、Tensor active
  和 Event latency；若 Long Scoreboard 不降或 occupancy 明显恶化则回退。

## 11. 推荐顺序

推荐 V2 只做候选 1：**32-bit vectorized fragment load，消除手工 packing**。

理由是它直接对应本轮最强证据链：

```text
LDG.E.U8 ×24 + packing ×18
    -> 12.5% sector utilization
    -> L1/TEX 98.24%
    -> LG Throttle 71.6% + Long Scoreboard 21.9%
    -> Tensor pipe active 4.01%
```

该方向改动面最小、可单独归因、无需改变 MMA fragment 或输出映射，也最适合作为
V1→V2 的严格对照。shared-memory reuse 和软件流水应分别作为后续独立实验。

## 12. 本轮未采集或不可直接使用的指标

- 没有把 NCU replay Duration 当作 CUDA Event benchmark 时间。
- 没有运行 Compute Sanitizer；上一轮已确认当前 WSL/WDDM memcheck 不支持。
- `--cache-control none` 下个别 replay-derived L2 分项 hit rate 超过 100%，已排除。
- 未采集 cuBLASLt 对照；本轮目标是诊断 V1 自身。
- 未执行 PM Sampling 独立 set；Warp State/Source Counters 已提供 stall sampling，足以
  识别 LG Throttle 和 Long Scoreboard。若后续需要时间轴，可额外采 `pmsampling`，
  但不应把 sampling duration 用作性能时间。

