# Tensor Core WMMA V2.1：V3 优化前性能瓶颈诊断

诊断日期：2026-10-13

## 1. 结论摘要

本轮没有修改 `src/tensor_core_gemm.cu` 或任何核心 kernel。诊断基于当前源码重新编译的独立二进制、已有 1024³ NCU/SASS 证据、新增的 2048³ NCU full-set 报告，以及与同精度默认 `cublasGemmEx` 的 CUDA Event 复测。

当前最值得优先验证的 V3 方向是 **Global→Shared 软件流水化**：先把标量 staging 改成可核查的 16 B 向量搬运，再引入双缓冲和异步 copy，使下一 K tile 的搬运与当前 tile 的 WMMA 重叠。证据不是单看 `LDG.E.U16` 数量或 Stall 百分比，而是以下组合：

- 源码中每个 K tile 先完成全部 Global→Shared，再做 block barrier、LDSM/HMMA、第二次 barrier，没有跨 tile 重叠。
- SASS staging 为标量 `LDG.E.U16`→依赖的 `STS.U16`；2048³ source sampling 的 Long Scoreboard 主要落在这些依赖 `STS.U16` 上。
- 2048³ Long Scoreboard 仍为 13.414 cycles/issued instruction，而 DRAM throughput 仅 4.05%，更像搬运依赖延迟和不足的 latency hiding，不像 DRAM 带宽打满。
- 2048³ 只有 0.55 eligible warps/scheduler，Issue Slots Busy 33.49%，Tensor/HMMA pipe active 约 31.2%，计算管线仍明显吃不满。

但这仍是“证据最强的候选瓶颈”，不是已经由 V3 对照实验确认的唯一主因。A-side LDSM 的剩余冲突、1024³ 小网格欠填充、56 registers/thread 和固定 64×64 CTA tile 也有直接证据，应通过单变量版本逐项验证。

## 2. 测量环境与范围

| 项目 | 本轮值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 12 GB，48 SM |
| Compute Capability / 编译目标 | 12.0 / `sm_120` |
| Driver | 596.49 |
| CUDA Toolkit / nvcc | 13.2 / 13.2.86 |
| CUDA runtime / driver API | 13020 / 13020 |
| cuBLAS runtime | 130401（13.4.1） |
| Nsight Compute | 2026.1.1.0 |
| OS | WSL2，Linux 6.18.33.2 |
| Git HEAD | `24e9107` |
| 当前 Tensor 源码 SHA-256 | `92d187c958219d7fd9053e7d7d214b2914fcec75b74f7f2ca15edf419865581b` |

编译选项：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_v3diag -lcublas
```

ptxas 对 V2.1 报告 56 registers/thread、4352 B static shared memory、1 barrier、0-byte stack、0 spill load/store。测试只覆盖当前支持范围：M/N 为 64 的倍数、K 为 16 的倍数。

## 3. Benchmark 复测

### 3.1 协议

- V2.1 与 cuBLAS 使用相同 FP16 A/B、FP32 accumulation/output、相同 device allocations 和默认 stream。
- cuBLAS 为 `cublasGemmEx(..., CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)`；没有 cuBLASLt heuristic 调参。
- 每个实现每轮 10 次 warmup、100 次 timed launches，共 5 轮；相邻轮次反转 V2.1/cuBLAS 顺序。
- CUDA Event 只包围 GPU GEMM；初始化、CPU 双 reference、H2D/D2H 和 handle 创建均不计时。
- 1024³ 为一个独立进程的 5 个样本。2048³ 为两个不在 profiler 包装下的独立进程，共 10 个样本的聚合中位数；没有选最快值。

### 3.2 本轮结果

| 尺寸 | V2.1 median | V2.1 | cuBLAS median | cuBLAS | V2.1/cuBLAS |
|---|---:|---:|---:|---:|---:|
| 1024³ | 0.104478 ms | 20.554 TFLOPS | 0.040603 ms | 52.890 TFLOPS | 38.86% |
| 2048³ | 0.623202 ms | 27.567 TFLOPS | 0.272638 ms | 63.013 TFLOPS | 43.75% |

1024³ 五轮：

- V2.1：0.105626, 0.106310, 0.102828, 0.104478, 0.102675 ms
- cuBLAS：0.040603, 0.040018, 0.040671, 0.042813, 0.040423 ms

2048³ 两个纯 Event 进程：

- V2.1：0.648571, 0.720746, 0.665750, 0.611108, 0.666274 ms
- V2.1：0.625940, 0.615309, 0.608444, 0.613316, 0.620463 ms
- cuBLAS：0.297615, 0.304396, 0.272484, 0.269723, 0.292904 ms
- cuBLAS：0.267551, 0.271710, 0.274275, 0.271047, 0.272792 ms

相对 2026-10-12 最终报告，本轮 1024³ 的 V2.1/cuBLAS latency 分别变化 +1.64%/+0.46%；2048³ 聚合中位数分别变化 +2.58%/+1.39%。V2.1 达成率仍约 39%/44%，历史结论复现。

正确性方面，1024³、2048³ 和自动补跑的 64×96×32、128×192×64 均通过双 reference、NaN/finite 和 launch/synchronize 检查。本组确定性输入中 V2、V2.1、cuBLAS 逐元素差为 0；这不是接口保证的 bitwise 等价性。

## 4. NCU 与 SASS 关键证据

1024³ 复用已有 `build/ncu_v2_1.ncu-rep`；2048³ 新采 `build/ncu_v2_1_v3diag_2048.ncu-rep`。两者均为 `--set full`、只匹配一个 V2.1 launch、40 replay passes。NCU replay duration 不参与上面的 Event 性能排名。

### 4.1 利用率、stall 与资源

| 指标 | 1024³ | 2048³ | 解释边界 |
|---|---:|---:|---|
| Grid blocks | 256 | 1024 | 直接观测 |
| Waves/SM | 0.59 | 2.37 | 1024³ 明确欠填充；2048³ 不再是小于一 wave |
| Theoretical occupancy | 75.00% | 75.00% | 56 registers/thread 限制 |
| Achieved occupancy | 44.55% | 60.91% | 直接观测，不等于性能上限 |
| Active warps/scheduler | 5.32 | 7.30 | 直接观测 |
| Eligible warps/scheduler | 0.29 | 0.55 | 很少 warp 可立即发射 |
| Issue Slots Busy | 20.34% | 33.49% | 规模增大后改善但仍低 |
| Tensor pipe active（elapsed） | 17.88% | 31.22% | 直接观测 |
| HMMA subpipe active（elapsed） | 17.89% | 31.22% | 直接观测 |
| Warp cycles/issued instruction | 25.94 | 21.81 | 直接观测 |
| Long Scoreboard | 18.376 | 13.414 | cycles/issued instruction |
| Short Scoreboard | 1.284 | 1.165 | cycles/issued instruction |
| Barrier | 0.694 | 0.892 | cycles/issued instruction；不是最大 stall |
| MIO Throttle | 0.183 | 0.286 | cycles/issued instruction |
| LG Throttle | 0.082 | 0.050 | cycles/issued instruction |
| DRAM throughput | 3.53% | 4.05% | 未显示 DRAM 带宽饱和 |
| L1/TEX throughput | 20.75% | 33.84% | 直接观测 |

不能仅凭 Long Scoreboard 占比断定根因。这里更有力的是 source sampling：2048³ 最大的 Long Scoreboard 样本位于四条 `STS.U16`，这些 store 正在等待前序 `LDG.E.U16` 的寄存器结果；源码和 SASS 同时显示搬运与本 tile 计算串行，且没有下一 tile 的预取。

### 4.2 Global→Shared

2048³ 直接计数：

| 指标 | 数值 |
|---|---:|
| Global-load instructions/requests | 8,388,608 |
| Global-load sectors | 16,777,216 |
| Sectors/request | 2.0 |
| Shared-store instructions | 8,388,608 |
| Total executed instructions | 158,863,360 |

Source counters 对每条有效 `LDG.E.U16` 都给出 actual sectors = ideal sectors，因此当前 Global load 虽是标量 half 指令，但访问是合并的。结论是：

- **直接事实：** 标量 staging 带来大量 load/store 指令和地址依赖；并非 sector 利用率问题。
- **合理推测：** 向量化可减少动态搬运/地址指令，异步双缓冲可隐藏 `LDG→STS` 依赖延迟。
- **尚未证实：** `LDG.E.U16` 数量本身不是“主要瓶颈”的充分证据；必须由 V3 单变量版本的 latency、动态指令和 Long Scoreboard 共同验证。

### 4.3 Shared→WMMA Fragment

V2.1 每个 K-loop 的核心 SASS 为：

```text
2 x LDSM.16.M88.4       # A0/A1
2 x LDSM.16.MT88.4      # B0/B1
8 x HMMA.16816.F32      # four WMMA calls
2 x BAR.SYNC.DEFER_BLOCKING
```

2048³ 每条 LDSM 执行 524,288 次：

| 指令 | Source actual / ideal / excessive wavefronts |
|---|---:|
| A0 `LDSM.16.M88.4` | 4,194,304 / 2,097,152 / 2,097,152 |
| A1 `LDSM.16.M88.4` | 4,194,304 / 2,097,152 / 2,097,152 |
| B0 `LDSM.16.MT88.4` | 2,097,152 / 2,097,152 / 0 |
| B1 `LDSM.16.MT88.4` | 2,097,152 / 2,097,152 / 0 |

B padding 72 的效果在 2048³ 继续成立；剩余 excessive wavefront 全部来自 A0/A1。aggregate load conflict 为 4,375,537，略高于 source-correlated excessive 4,194,304，属于 replay/归因差异。归一化后仍约为每个 fragment-load request 2 个 excessive wavefront。

2048³ aggregate counter 报告 349,702 个 shared-store conflicts，但逐条 source counters 对八条有效 `STS.U16` 都给出 actual=ideal、excessive=0，且 1024³ aggregate store conflicts 为 0。因此本轮不把 shared store conflict 判为瓶颈；若 V3 改动 store layout，应使用 focused metric 重采确认这处不一致。

### 4.4 Tensor pipe、barrier、动态工作量

2048³ 动态计数包括 2,097,152 条 LDSM、4,194,304 条 HMMA、1,048,576 条 block barrier 和 65,536 条 global store。HMMA 总工作量符合算法预期，没有“Tensor 指令不足”或 spill 的证据；问题是 Tensor pipe 只有约 31.2% active，更多时候没有可发射的 Tensor 工作。

Barrier stall 只有 0.892 cycles/issued，不能说 barrier 本身是首要瓶颈。但每 K tile 的两个 full-block barrier 固定了“load→compute→下一 tile”的串行阶段，是阻止跨 tile 重叠的结构性约束。V3 的目标应是改变流水依赖，而不只是机械减少 barrier 计数。

## 5. 瓶颈分级

### 直接观测事实

1. V2.1 与默认同精度 cuBLAS 仍有约 2.3–2.6× latency 差距。
2. 1024³ 只有 0.59 waves/SM；2048³ 提升到 2.37 waves/SM 后性能和 Tensor 利用率改善，但 Tensor/HMMA active 仍仅约 31.2%。
3. Global loads 完全合并，但 staging 是标量 `LDG.E.U16`/`STS.U16`，没有 async copy 或双缓冲。
4. Long Scoreboard 是最大 stall；2048³ source sampling 将主要样本定位到等待 global-load 数据的 shared stores。
5. B-side LDSM 已达到 actual=ideal；A0/A1 仍各为 8 actual / 4 ideal。
6. 56 registers/thread、75% theoretical occupancy、0 spill；2048³ achieved occupancy 60.91%。
7. 每 K tile 有两个 full-block barrier，且当前 K stage 深度为 1。

### 合理推测

1. **最可能的首要限制是 Global→Shared latency/依赖没有被计算覆盖**，而不是 DRAM 峰值带宽不足。
2. A-side LDSM conflict 是真实但次一级的 Shared→Fragment 开销；其 Short Scoreboard/MIO 量级明显低于 Long Scoreboard。
3. 1024³ 还受到 CTA 数不足和 tail/underfill 影响；同一个 64×64 tile 不太可能同时最适合所有尺寸/长宽比。
4. 56-register footprint 限制可用 warps，但盲目压寄存器未必有益，因为可能降低 ILP 或产生 spill。

### 尚未证实

1. async copy、向量化或双缓冲各自能带来多少 speedup。
2. A padding/swizzle 是否能转化为可测 latency 收益，而不只是 conflict counter 下降。
3. cuBLAS 的具体内部 tile、pipeline、TMA 或 warp-specialization 策略；本轮没有 profile cuBLAS 内部 kernel。
4. 单独减少 barrier、LDG 指令数或寄存器数是否足以提升性能。

## 6. V3 候选优化方向

| 优先级 | 方向 | 预期作用 | 主要风险 | 必须验证 |
|---|---|---|---|---|
| P0 | Global→Shared 16 B 向量搬运，随后做双缓冲 + async copy | 降低 staging/地址动态指令；用当前 tile HMMA 覆盖下一 tile 搬运；降低 Long Scoreboard，提高 eligible warps 和 Tensor active | 对齐和 tile 边界；wait/commit/barrier 正确性；shared/寄存器增加；可能只改变指令数却没隐藏延迟 | 先做“只向量化”单变量，再做“流水化”单变量；检查 SASS 是否出现预期宽 load/store 或 async copy、Long Scoreboard、总指令、Tensor active、ptxas spill，以及两尺寸 Event 中位数 |
| P1 | A shared stride padding/skew（从合法的 8-half 倍数中小范围实验） | 消除 A0/A1 的 8/4 LDSM，降低剩余约 2 excessive wavefront/request | 额外 shared、地址运算或新 store conflict；WMMA ldm/对齐约束 | 逐条 A LDSM actual/ideal/excessive；aggregate load/store conflicts；Short Scoreboard/MIO；Event speedup 必须大于波动 |
| P2 | 1024³/形状专用 CTA tile（如 64×32、32×64），保留 64×64 作为大方阵候选 | 增加小矩阵 block 数，缓解 0.59 waves/SM；改善非方形适配 | fragment/global reuse 下降，动态 load 和 barrier 增加；可能伤害 2048³ | 同一协议分别测 1024³/2048³/2048×1024×2048；比较 waves/SM、global/LDSM/HMMA 工作量和 cuBLAS 达成率 |
| P3 | 在流水化后再做寄存器/warp tile 调整 | 尝试把理论 occupancy 从 75% 提高并增加 eligible warps | 破坏四 accumulator reuse、增加指令或 spill；occupancy 提高不保证性能提高 | ptxas/NCU resources、local traffic=0、eligible warps、issue busy、Event latency；不以 occupancy 单指标决策 |
| P4 | 更深 K stage、TMA 或 producer/consumer warp specialization | 进一步降低同步和搬运开销，接近库级流水 | 实现复杂、Blackwell 专用、资源和正确性风险高，难以保持单变量归因 | 仅在 P0 证明搬运流水有效后开展；逐阶段 ablation，并保留普通 async-double-buffer 基线 |

推荐实施顺序是 P0→P1→P2→P3；P4 暂不作为首个 V3。P0 内部也应拆成“向量化”和“异步双缓冲”两个可独立回退的提交/实验版本，避免把收益错误归因。

## 7. 可复现命令

```bash
# 编译独立诊断二进制，不覆盖历史产物
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_v3diag -lcublas

# 普通 Event benchmark；2048³ 建议跑两个独立进程并聚合 10 个样本
./build/tensor_core_gemm_v3diag 1024 1024 1024
./build/tensor_core_gemm_v3diag 2048 2048 2048
./build/tensor_core_gemm_v3diag 2048 2048 2048

# 只 profile 首个匹配的 V2.1 launch；不要把 replay duration 当 benchmark
ncu --set full --kernel-name-base demangled \
  --kernel-name regex:wmmaGemmSharedV2_1 --launch-count 1 \
  -o build/ncu_v2_1_v3diag_2048_rerun \
  ./build/tensor_core_gemm_v3diag 2048 2048 2048

# 查看报告和当前二进制 SASS
ncu --import build/ncu_v2_1_v3diag_2048_rerun.ncu-rep --page details
cuobjdump --dump-sass build/tensor_core_gemm_v3diag
cuobjdump --dump-resource-usage build/tensor_core_gemm_v3diag
```

本轮新增产物为 `build/tensor_core_gemm_v3diag` 和 `build/ncu_v2_1_v3diag_2048.ncu-rep`。没有覆盖历史报告、没有实现 V3、没有提交 Git，也没有清理任何既有 untracked Markdown。
