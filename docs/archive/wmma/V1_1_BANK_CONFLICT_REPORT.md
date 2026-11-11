# Tensor Core V1.1：Shared B Padding 40 Bank Conflict 实验报告

日期：2026-10-11（Asia/Shanghai）

## 1. 实验结论

本实验只改变一个变量：V1.1 将 Shared B 的物理行跨度从 32 个 `half` 改为 40 个 `half`。V0、V1 的 kernel 保持不变；MMA tile、block tile、warp mapping、A layout、同步和计算逻辑均未改变。

在 NVIDIA GeForce RTX 5070（CC 12.0）上，Padding 40 有效：

- B 的 `LDSM.16.MT88.4` 从 16 个 actual wavefront/request 降到理想的 4 个，B 的 excessive wavefronts 从 3,145,728 降为 0。
- A 的 `LDSM.16.M88.4` 保持 8 个 actual wavefront/request、4 个 ideal wavefront/request，A excessive wavefronts 保持 1,048,576；说明 A 没被 B padding 改变。
- shared-load bank-conflict counter 从 4,265,347 降到 1,110,154，下降 73.97%；shared-load actual wavefronts 从 6,362,363 降到 3,210,162，下降 49.54%。
- 普通 CUDA Event benchmark 中，V1.1 相对 V1 的中位延迟 speedup 为：1024³ 1.0992x，2048³ 1.1118x。
- V1.1 与 V0/V1 在三种尺寸上的结果一致并通过两套 reference；NaN 预填没有发现遗漏的 C tile。
- 代价为每 block 多 256 B static shared memory。aggregate shared-store 指标出现约 0.93% 的 wavefront 增量，但逐条 `STS.U16` source correlation 仍为 1-way、0 excessive；详见第 7 节，不能据此断言产生了稳定的结构性 store bank conflict。

因此，本次结果是“冲突显著减少且性能改善”，不是持平或退化。

## 2. 环境与方法

- GPU：NVIDIA GeForce RTX 5070，compute capability 12.0，12227 MiB
- Driver：596.49
- CUDA：13.2，`nvcc` 13.2.86
- Nsight Compute：2026.1.1
- 编译：`nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v src/tensor_core_gemm.cu`
- 普通 benchmark：每个 kernel 每轮 10 次 warmup、100 次计时 launch，共 5 轮；CUDA Event 只包围 kernel launch，报告五轮 per-launch average 的中位数。
- Nsight Compute：1024³、`--set full`、每个 kernel 只采一个匹配 launch、40 replay passes；profiler duration 不用于普通性能对比。
- 对照报告：`build/ncu_v1_new.ncu-rep`
- V1.1 报告：`build/ncu_v1_1_new.ncu-rep`
- SASS：`build/tensor_core_gemm_v1_1.sass`
- PTX：`build/tensor_core_gemm_v1_1.ptx`

Compute Sanitizer 的 memcheck 在该 WDDM/RTX 5070 环境报告 “Device not supported / Failed to initialize WDDM debugger interface”，所以不能将其作为通过证据。有效 correctness 证据来自逐元素双 reference、NaN 预填、finite 检查以及每次 launch 后的 CUDA error/synchronize 检查。

## 3. V1 原始冲突分析

### 3.1 文档支持的事实

NVIDIA 的 [CUDA C Programming Guide 13.0（WMMA）](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-matrix-functions) 规定：

- `load_matrix_sync` 的 `mptr` 必须 256-bit 对齐；
- 对 `__half`，`ldm` 必须是 8 个元素的倍数，即字节跨度是 16 B 的倍数；
- row-major 时 `ldm` 是相邻逻辑行之间的元素跨度；整个 warp 必须使用相同参数并共同执行。

NVIDIA 的 [PTX ISA `ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix) 规定：

- `ldmatrix` 由整个 warp 协作从 shared memory 加载一个或多个矩阵；
- `.m8n8` 的每个元素为 16 bit，`.x4` 表示四个矩阵；
- `.x4` 时 lanes 0–7、8–15、16–23、24–31 分别提供四个 8×8 矩阵的八个行地址；
- 每四个连续线程共同加载 16 B，行地址必须自然对齐；`.trans` 表示以 column-major 形式装载。

[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) 说明：shared memory 有 32 banks，相邻 32-bit word 映射到相邻 bank；同一 request 的不同地址落到同一 bank 时需要序列化。一个 request 可以产生多个 wavefront，不同 wavefront 在不同 cycle 串行处理；因此 actual wavefronts 相对 ideal wavefronts 的增加是判断额外序列化的直接指标。

### 3.2 PTX/SASS 实测

V1 PTX 是：

```text
wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 ..., ldm=16
wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 ..., ldm=32
wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 ...
```

在 `sm_120` 上，ptxas 将 Shared-to-Fragment load 生成为：

```text
LDSM.16.M88.4  R16, [R16]   // A
LDSM.16.MT88.4 R20, [R20]   // B
HMMA.16816.F32 ...
HMMA.16816.F32 ...
```

`LDSM.16.MT88.4` 是当前工具链为 B row-major WMMA load 生成的实际 SASS；`MT` 与 PTX 的 transpose 型矩阵装载一致，`.4` 与四个 8×8 子矩阵的形式一致。但 NVIDIA 没有在公开 SASS ISA 中给出 `LDSM.16.MT88.4` 的逐 lane 编码语义。本报告只用公开的 PTX `ldmatrix` lane/address 规则建立解释模型，不把它当作对该 Blackwell SASS 指令逐 bit 解码的证明。

### 3.3 简化 bank mapping 推导

V1 的 B 是 `half Bs[16][32]`。以文档中的 32 banks、相邻 32-bit word 轮转映射为模型，行 `r` 的起始 bank 为：

```text
bank_start(r) = floor((base + r * stride * sizeof(half)) / 4) mod 32
```

忽略共同的 base 项：

```text
stride=32: bank_start(r) = 16r mod 32 = 0,16,0,16,...
```

一行 32 个 `half` 是 64 B，相邻行只移动 16 banks，两个行起点后就重复。一个 8-element half row fragment 是 16 B、覆盖四个 32-bit bank word；当 `LDSM` 协作处理多个 8×8 子矩阵的行时，大量行片段反复落入相同的 bank 组，容易产生序列化。

这部分是理论模型，不足以单独断言“必然 16-way”。真正的冲突度来自 profiler：V1 的 B `LDSM.16.MT88.4` 每个 request 有 16 actual wavefronts、4 ideal wavefronts，产生 3,145,728 excessive wavefronts；A 则是 8 actual、4 ideal，产生 1,048,576 excessive wavefronts。两条指令各执行 262,144 次，共 524,288 个 shared-load requests。

V1 的合计实测为：

| 指标 | V1 |
|---|---:|
| Shared-load requests | 524,288 |
| Ideal load wavefronts | 2,097,152 |
| Actual load wavefronts | 6,362,363 |
| Source-correlated excessive wavefronts | 4,194,304 |
| Shared-load bank-conflict counter | 4,265,347 |
| NCU 报告的平均 actual wavefront/request | 12.1 |

aggregate counter 与 source-correlated 的整数差异来自不同 replay pass/指标定义；判断具体 A/B 指令时，以 source correlation 的 actual/ideal/excessive 三元组为准。

## 4. Padding 40 的依据、合法性与局限

V1.1 使用 40 个 `half` 的物理行跨度，即 80 B：

```text
stride=40: bank_start(r) = 20r mod 32
             = 0,20,8,28,16,4,24,12,...
```

`gcd(20, 32)=4`，所以起始 bank 在八行之后才重复，而不是 V1 的两行周期。这会把各 16 B 行片段分散到更多 bank 组，是选择 40 的理论依据。

合法性检查：

- 40 是 8 的倍数，满足 half WMMA 的 `ldm` 约束；80 B 是 16 B 的倍数。
- `Bs` 本身 `__align__(32)`；`warp_col=0/1` 的子 tile 起点分别为 base+0 B/base+32 B，均满足 256-bit 对齐。
- cooperative load 仍以 `BLOCK_K * BLOCK_N = 16 * 32` 为逻辑范围，padding 元素不读取、不参与计算，也无需初始化。
- Shared B 从 16×32×2=1024 B 增加到 16×40×2=1280 B；A 仍为 1024 B。

局限性：简化 bank 公式解释“为何值得测试”，但不能替代 `LDSM` 的真实内部 transaction/lane grouping；最终结论必须来自 actual/ideal wavefronts 和 bank-conflict counter。

## 5. 实际代码修改

新增常量和编译期约束：

```cpp
constexpr int B_SHARED_STRIDE = 40;

static_assert(B_SHARED_STRIDE >= BLOCK_N && B_SHARED_STRIDE % 8 == 0,
              "The padded half WMMA leading dimension must be >= BLOCK_N "
              "and a multiple of 8 elements");
```

新增独立 kernel `wmmaGemmSharedV1_1`，核心差异仅为：

```cpp
__shared__ __align__(32) half As[BLOCK_M][BLOCK_K];
__shared__ __align__(32) half Bs[BLOCK_K][B_SHARED_STRIDE];

for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += blockDim.x) {
    int row_Bs = idx / BLOCK_N;
    int col_Bs = idx % BLOCK_N;
    // ...
    Bs[row_Bs][col_Bs] = B[row_B * N + col_B];
}

wmma::load_matrix_sync(b_frag, B_frag_ptr, B_SHARED_STRIDE);
```

其余 block tile 32×32×16、2×2 warps、16×16×16 MMA、A layout、两个 `__syncthreads()`、accumulation 和 C store 与 V1 相同。host runner 增加 V1.1 的 correctness、五轮 benchmark 和相对 V1 speedup 输出。

## 6. Correctness

Reference 1 是原始 FP32 输入直接进行 FP32 GEMM，包含 FP32→FP16 输入量化误差；Reference 2 将 FP16 输入转回 FP32 后进行 FP32 GEMM，用于隔离 kernel 算术并按 `abs_error <= 1e-2 + 1e-2 * abs(reference)` 判定。

每次 kernel 前将整个 `d_C` 填为 NaN；结果必须全部 finite 且通过 Reference 2。三种 kernel 在每个尺寸上的误差统计完全相同：

| M×N×K | Kernel | Ref.1 max abs | Ref.1 max rel | Ref.2 max abs | Ref.2 max rel | 结果 |
|---|---|---:|---:|---:|---:|---|
| 64×96×32 | V0 | 5.761981e-4 | 1.952723e+2 | 1.192093e-7 | 5.948920e-6 | Pass |
| 64×96×32 | V1 | 5.761981e-4 | 1.952723e+2 | 1.192093e-7 | 5.948920e-6 | Pass |
| 64×96×32 | V1.1 | 5.761981e-4 | 1.952723e+2 | 1.192093e-7 | 5.948920e-6 | Pass |
| 1024³ | V0 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 1024³ | V1 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 1024³ | V1.1 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 2048³ | V0 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |
| 2048³ | V1 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |
| 2048³ | V1.1 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |

64×96×32 的 grid 为 `(3,2)` blocks，每 block 2×2 warps，覆盖六个互不重叠的 32×32 C block tiles。该非方形案例通过且没有残留 NaN，说明没有交换 block/warp 行列，也没有遗漏 C tile。小尺寸 Ref.1 的大相对误差来自 reference 接近零；其最大绝对误差仍为 5.76e-4，且 Ref.2 很小，未放宽容差。

## 7. Benchmark

### 7.1 中位数

“相对 V1 speedup”定义为 `V1 median latency / 当前 kernel median latency`。

| 尺寸 | Kernel | Median latency (ms) | Median TFLOPS | 相对 V1 speedup |
|---:|---|---:|---:|---:|
| 1024³ | V0 | 0.148014 | 14.508644 | 0.9526x |
| 1024³ | V1 | 0.140994 | 15.231071 | 1.0000x |
| 1024³ | V1.1 | 0.128270 | 16.741849 | **1.0992x** |
| 2048³ | V0 | 1.099120 | 15.630562 | 0.9022x |
| 2048³ | V1 | 0.991578 | 17.325783 | 1.0000x |
| 2048³ | V1.1 | 0.891864 | 19.262880 | **1.1118x** |

### 7.2 五轮原始 per-launch average

- 1024 V0：0.146328, 0.148014, 0.148523, 0.148218, 0.146578 ms
- 1024 V1：0.142429, 0.138280, 0.140994, 0.141095, 0.140532 ms
- 1024 V1.1：0.124980, 0.129320, 0.125986, 0.128790, 0.128270 ms
- 2048 V0：1.099363, 1.105298, 1.097623, 1.099120, 1.098807 ms
- 2048 V1：0.989468, 0.986022, 0.994482, 0.991578, 1.000136 ms
- 2048 V1.1：0.910419, 0.891572, 0.891864, 0.891588, 0.895525 ms

两种尺寸上 V1 最快一轮仍慢于 V1.1 最慢一轮，因此 9.9%/11.2% 的差距明显超过本组测量波动。2048 V1.1 第一轮 0.910419 ms 偏慢，但不改变中位数结论。

## 8. SASS 与 Nsight Compute：V1 vs V1.1

### 8.1 指令与资源

| 指标（1024³） | V1 | V1.1 | 解释 |
|---|---:|---:|---|
| Registers/thread | 40 | 40 | 不变，无 spill |
| Static shared memory/block | 2048 B | 2304 B | +256 B，来自 B padding |
| Theoretical occupancy | 100% | 100% | 不变 |
| Achieved occupancy | 87.94% | 88.81% | 采样值，小幅波动/改善 |
| Executed instructions | 45,965,312 | 45,965,312 | 动态总数不变 |
| Global-load instructions/requests | 2,097,152 | 2,097,152 | 不变，`LDG.E.U16` |
| Global-load sectors | 4,194,304 | 4,194,304 | 不变 |
| Global-store instructions/requests | 16,384 | 16,384 | 不变，C 使用 `STG.E.64` |
| Global-store sectors | 131,072 | 131,072 | 不变 |
| Shared instructions total | 2,621,440 | 2,621,440 | 不变 |
| Shared-store instructions | 2,097,152 | 2,097,152 | 不变，`STS.U16` |
| Fragment shared-load requests | 524,288 | 524,288 | 两条 LDSM 各 262,144 |

核心 SASS mnemonic 和编码不变：两者都有同样的 `LDSM.16.M88.4`、`LDSM.16.MT88.4` 和两条 `HMMA.16816.F32`。发生变化的是 B 地址生成常量：物理 byte stride 从 `0x40` 变为 `0x50`，传给 B WMMA load 的 element stride 从 32 (`0x20`) 变为 40 (`0x28`)。所以“生成的 SASS 是否变化”的准确回答是：访存/计算指令类型和数量不变，B 地址计算立即数发生预期变化。

### 8.2 Shared load/store

| 指标（1024³） | V1 | V1.1 | 变化 |
|---|---:|---:|---:|
| Shared-load requests | 524,288 | 524,288 | 0 |
| Shared-load ideal wavefronts | 2,097,152 | 2,097,152 | 0 |
| Shared-load actual wavefronts | 6,362,363 | 3,210,162 | -49.54% |
| Shared-load source excessive wavefronts | 4,194,304 | 1,048,576 | -75.00% |
| Shared-load bank conflicts | 4,265,347 | 1,110,154 | -73.97% |
| NCU average actual wavefront/request | 12.1 | 6.1 | 改善 |
| Shared-store actual wavefronts | 2,097,152 | 2,116,676 | +0.93% |
| Shared-store bank-conflict counter | 1,274 | 20,829 | +19,555 |

逐指令 source correlation：

| 指令 | V1 actual / ideal / excessive | V1.1 actual / ideal / excessive |
|---|---:|---:|
| A `LDSM.16.M88.4` | 2,097,152 / 1,048,576 / 1,048,576（8-way） | 相同（8-way） |
| B `LDSM.16.MT88.4` | 4,194,304 / 1,048,576 / 3,145,728（16-way） | 1,048,576 / 1,048,576 / 0（4-way，即该指令的 intrinsic ideal） |

注意：这里 V1.1 B 的“4-way”不是还存在四倍冲突；这条 512 B 级矩阵装载本身的 ideal 就是每 request 四个 wavefront。actual=ideal 且 excessive=0 才是关键证据。

V1.1 的 aggregate store counter 确实比 V1 高，并伴随 19,524 个额外 store wavefront，必须记录为小代价。但 source correlation 中每条 `STS.U16` 仍显示 1-way、actual=ideal、0 excessive，且增量不是 kernel 固定执行次数的整倍数。因此现有证据不足以把它归因成某一条稳定的结构性 store bank conflict；它可能涉及 replay/counter 变化或 aggregate 归因差异。无论原因，该量级只占约 0.93%，远小于消除的 3,145,728 个 B-load excessive wavefronts。

### 8.3 Stall、Tensor pipe 与 profiler duration

| 指标（1024³） | V1 | V1.1 | 观察 |
|---|---:|---:|---|
| Long Scoreboard | 8.7571 | 9.0136 | +2.93%，没有改善；其占比因总 issue 间隔缩短而上升 |
| MIO Throttle | 1.2186 | 0.6007 | -50.71% |
| LG Throttle | 0.0310 | 0.0201 | 很低，进一步下降 |
| Barrier stall | 0.9905 | 0.7418 | 下降；同步指令未改变 |
| Warp cycles / issued instruction | 18.06 | 17.15 | 下降 |
| Eligible warps/scheduler | 1.54 | 1.63 | 上升 |
| Tensor pipe cycles active (elapsed) | 23.78% | 25.40% | +1.62 percentage points（相对 +6.83%） |
| HMMA subpipe cycles active (elapsed) | 23.76% | 25.45% | 上升 |
| NCU replay duration | 166.56 us | 158.02 us | -5.13%，仅辅助证据 |

MIO throttle 减半、每条 issued instruction 的 warp cycles 下降、eligible warps 增加，与 shared-load 序列化减少一致。Long Scoreboard 没有下降，说明 padding 只处理了 shared/L1TEX 路径中的一个瓶颈，并未消除 global/L1TEX dependency。Tensor pipe 的指令数不变，但等待减少后其 active 比例提高；这支持“Tensor Core 获得了更连续的供数”，而不是算力指令变多。

## 9. 对五个核心问题的回答

1. **B Padding 是否真正降低了 Bank Conflicts？** 是。B `LDSM` 的 actual wavefronts 从 ideal 的 4 倍降到 actual=ideal，B excessive wavefronts 清零；总 shared-load bank conflicts 下降 73.97%。
2. **是否产生新的 Shared Store 冲突或其他代价？** aggregate 指标出现 0.93% store wavefront 增量和更高的 store-conflict counter，但逐指令没有定位到 N-way/excessive `STS`；应记录为小的实测代价/不确定项，不能夸大为确定的结构性冲突。确定代价是每 block +256 B shared memory；register、occupancy 上限和指令数不变。
3. **生成的 SASS 是否发生变化？** 核心 mnemonic、编码和动态指令数不变；B 的地址计算 stride 立即数按预期从 32/64 B 改为 40/80 B。
4. **Tensor Core 利用率是否提高？** 是。Tensor pipe elapsed active 从 23.78% 提到 25.40%，HMMA subpipe 从 23.76% 提到 25.45%；HMMA 指令数不变。
5. **最终性能如何？为什么？** 改善。1024³/2048³ 分别比 V1 快 9.92%/11.18%。主因是 B-load 的 3,145,728 个 excessive wavefronts 被消除，使 MIO throttle 和 issue 间隔下降；A 冲突与 Long Scoreboard 仍在，所以提升不是无限的。

## 10. 下一步建议（本次不实施）

Padding 40 已验证有效，当前代码不应继续混入其他变量。建议后续分别作为独立实验：

1. 重复采集 V1/V1.1 的 shared-store counters，确认约 0.93% 增量是否稳定，并用 source-only metric set 尝试提高同 pass 归因一致性。
2. 以单变量实验研究 A 的 8-way `LDSM.16.M88.4`，不要同时改 B 或 warp tile。
3. 在独立阶段再考虑 swizzle、转置布局、double buffering 或 async copy；它们不属于本次 V1.1。

本实验未提交 Git commit，也未开始 V2。
