# Tensor Core V2：Fragment Reuse 集成与性能分析报告

日期：2026-10-11（Asia/Shanghai）

## 1. 结论摘要

本轮只测试用户编写的无 Padding V2：`64x64x16` block tile、`2x2` warps、每 warp 计算 `32x32` C tile，并以两个 A fragment、一个复用的 B fragment 和四个 FP32 accumulator 完成四次 WMMA。没有修改 V0/V1/V1.1 的核心计算逻辑，也没有加入 B/A padding、swizzle、double buffering、async copy 或手写 PTX。

主要结论如下：

- **正确性通过。** V0、V1、V1.1、V2 在 `128x192x64`、`1024^3`、`2048^3` 上都通过 FP16-quantized input reference；每次运行前用 NaN 预填 C，所有输出均为 finite。V1.1 与 V2 在三个 V2 合法尺寸上逐元素完全相同，最大绝对/相对差均为 0。
- **V2 的 fragment reuse 确实生效。** 在 1024³ 的 profiler 实测中，V2 相对 V1.1 将 global-load requests、shared-store instructions 和 Shared-to-Fragment requests 都减少 50%，动态总指令减少 56.65%；总 HMMA 动态次数仍为 524,288，计算量没有减少。
- **V2 的 B layout 有严重 bank conflict。** `Bs[16][64]` 的行跨度为 128 B，即 32 个 4-byte bank word，简化模型中相邻行起始 bank 完全相同。实测两个 B `LDSM.16.MT88.4` 都是 32 actual wavefront/request、4 ideal wavefront/request；V2 shared-load bank-conflict counter 为 4,208,214，约为 V1.1 的 3.79 倍。
- **寄存器压力明显增加但没有 spill。** V1.1 为 40 registers/thread，V2 为 56（+40%）；V2 registers/block 为 7,168，理论 occupancy 从 100% 降到 75%。ptxas 和 NCU 均显示 0 spill、0 local-memory load/store。
- **性能具有明显尺寸依赖。** 普通 CUDA Event benchmark 中：
  - 1024³：V2 聚合中位数 0.129789 ms、16.546 TFLOPS，相对 V1.1 为 0.9787x，即约慢 2.13%。两次独立进程均未显示 V2 获益，但差距较小且有波动。
  - 2048³：V2 中位数 0.769729 ms、22.319 TFLOPS，相对本轮重新测得的 V1.1 提升 **1.1816x（18.16%）**，且五轮区间完全分离。
- **1024³ 的 Tensor pipe 利用率没有提高。** 相同条件的 NCU 采样中，V2 HMMA subpipe active 从 V1.1 的 25.11% 降到 15.51%；这是 256-block 小网格仅有 0.59 waves/SM、occupancy 下降及 B LDSM 严重序列化共同作用下的实测结果。不能把它外推成 2048³ 的 Tensor pipe 数据。

因此，对“V2 是否优于 V1.1”的准确回答是：**1024³ 上没有，且小幅退化；2048³ 上明显优于 V1.1。双向 Fragment Reuse 带来的工作量削减是真实的，但当前无 Padding B layout 和更高寄存器占用显著限制了小尺寸表现。**

## 2. 证据分类与实验环境

本文使用以下标签区分证据：

- **[文档事实]**：NVIDIA 公开 CUDA/PTX/NCU 文档明确支持。
- **[代码推导]**：可由索引、tile 数量和循环次数直接推出，但不是性能计数器。
- **[Profiler 实测]**：本轮 `.ncu-rep`、SASS、ptxas 或 resource usage 的直接结果。
- **[解释/假设]**：用来解释实测现象，尚未通过独立对照实验验证。

环境：

- GPU：NVIDIA GeForce RTX 5070，compute capability 12.0，12,227 MiB
- Driver：596.49
- CUDA compiler：13.2，`nvcc` 13.2.86
- Nsight Compute：2026.1.1
- 编译目标：`sm_120`
- 编译选项：`-std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v --ptxas-options=-warn-spills`

普通 benchmark 方法：

- 相同 device allocations、输入和编译产物；CPU reference、H2D/D2H 不在计时区间。
- 每个 kernel 每轮 10 次 warmup、100 次 timed launches；CUDA Event 只包围 kernel launches。
- 每轮轮换 V0/V1/V1.1/V2 的测试次序，降低固定顺序导致的温度/频率偏差。
- 1024³ 进行了两次独立进程级测量，共 10 轮并报告聚合中位数；2048³ 进行 5 轮，报告中位数。
- NCU replay duration 不与普通 CUDA Event benchmark 混用。

Profiler 方法：

- 尺寸：1024³；V1.1 与 V2 使用同一个最终二进制和相同输入。
- 命令核心：`ncu --set full --kernel-name regex:<kernel> --launch-count 1`。
- 每个 kernel 40 replay passes。
- 报告文件：`build/ncu_v1_1_v2_compare.ncu-rep`、`build/ncu_v2.ncu-rep`。
- SASS/PTX：`build/tensor_core_gemm_v2.sass`、`build/tensor_core_gemm_v2.ptx`。

## 3. V2 源码审查

### 3.1 Mapping 与覆盖范围

**[代码推导]** V2 block 起点为：

```cpp
block_row = blockIdx.y * 64;
block_col = blockIdx.x * 64;
```

四个 warp 的 `(warp_row, warp_col)` 依次为 `(0,0)`、`(0,1)`、`(1,0)`、`(1,1)`；其 C tile 起点分别是 block 内的 `(0,0)`、`(0,32)`、`(32,0)`、`(32,32)`。每个 warp 写四个 `16x16` tile，正好覆盖一个互不重叠的 `32x32` 区域；四个 warp 完整且无重叠地覆盖 `64x64` block tile。

V2 第一版只在 `M,N` 为 64 的倍数且 `K` 为 16 的倍数时启动，因此当前无边界判断的 global load/store 不会越界。Host 端已显式检查这个约束。

### 3.2 Cooperative Loading

**[代码推导]** A、B 的逻辑 shared tiles 都有 1,024 个 half 元素：

```text
A: 64 x 16 = 1024
B: 16 x 64 = 1024
```

128 threads 各搬 8 个 A 和 8 个 B 元素；`idx += blockDim.x` 覆盖每个合法 `idx` 恰好一次。满足尺寸约束时，A 的 global 行范围是 `[block_row, block_row+63]`，B 的 global 列范围是 `[block_col, block_col+63]`，都在边界内。每个 shared 元素只有一个 writer，第一次 barrier 后所有 warp 才读取，因此没有数据竞争。

### 3.3 Fragment 指针、ldm 与对齐

每个 warp 的指针映射正确：

| Fragment | shared 起点（block 内） | 逻辑 tile |
|---|---:|---:|
| A0 | row `warp_row*32`，col 0 | `16x16` |
| A1 | row `warp_row*32+16`，col 0 | `16x16` |
| B0 | row 0，col `warp_col*32` | `16x16` |
| B1 | row 0，col `warp_col*32+16` | `16x16` |

**[文档事实]** NVIDIA [CUDA Programming Guide 的 WMMA 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)要求 `load_matrix_sync` 的指针 256-bit 对齐；对 `half`，`ldm` 必须是 8 个元素的倍数。

**[代码推导]** 当前布局满足要求：

- `As`/`Bs` 均 `__align__(32)`。
- A `ldm=16`，相邻行 32 B；A0/A1 起点相对 base 均为 32 B 的整数倍。
- B `ldm=64`，相邻行 128 B；B0/B1 的列偏移为 0、16、32、48 个 half，即 0、32、64、96 B，均 32 B 对齐。
- 16 和 64 都是 8 的倍数。

为防止参数以后失配，源码加入了 tile、warp 数量和 half-WMMA stride 的 `static_assert`。没有改变 V2 的布局或计算设计。

### 3.4 MMA、同步与 C store

四次 MMA 的 accumulator 对应关系正确：

```text
c00 += a0 * b0
c10 += a1 * b0
c01 += a0 * b1
c11 += a1 * b1
```

`b` 在完成 B0 的两次 MMA 后才被 B1 覆盖；A0/A1 在整个 K iteration 内保持有效。四个 accumulator 都在 K loop 前用 `fill_fragment(0)` 初始化。

第一次 `__syncthreads()` 位于 cooperative load 之后，防止未完成写入就读 shared；第二次位于当前 tile 所有 WMMA 读取和 MMA 之后，防止较快 warp 在较慢 warp 使用完之前覆盖下一轮 shared tile。位置正确且所有 threads 走一致控制流，不存在 barrier divergence。

C00/C01/C10/C11 的地址分别加 `(0,0)`、`(0,16)`、`(16,0)`、`(16,16)`，与四个 accumulator 对应。结合 warp 的 32x32 起点，四个 tile 不重叠且没有遗漏。

### 3.5 审查结果与最小修改

V2 kernel 的计算逻辑本身不需要修正。所做最小改动是：

1. 删除未被 kernel 使用、且容易让人误以为 V2 已 padding 的 `B_SHARED_STRIDE_V2=72` 常量；实际 V2 始终保持 `Bs[16][64]` 和 `ldm=64`。
2. 添加上述 V2 编译期约束。
3. 完成 `KernelVersion`、名称、二维 grid、correctness、benchmark 和 V2/V1.1 对比的 Host 集成。
4. 保留 `64x96x32` 原有非方形测试并为 V2 打印 skip；新增 V2 合法的 `128x192x64`。
5. benchmark 改为每轮轮换 kernel 顺序；warmup 和 timed-launch 数量对所有版本相同。

没有改写用户的 V2 算法，也没有给 V2 加 padding。

## 4. Correctness

Reference 1 使用原始 FP32 输入执行 FP32 CPU GEMM，用于观察输入量化误差。Reference 2 将实际 FP16 输入转回 FP32 后执行 CPU GEMM，用于检查 Tensor Core kernel；判定条件保持为：

```text
abs_error <= 1e-2 + 1e-2 * abs(reference)
```

每次 kernel 前以 `cudaMemset(..., 0xff, ...)` 将整个 C 填为 NaN；比较函数要求 reference/result/error 都为 finite。下面每个尺寸内，四个 kernel 的误差统计完全相同：

| MxNxK | Kernel | Ref.1 max abs | Ref.1 max rel | Ref.2 max abs | Ref.2 max rel | 结果 |
|---|---|---:|---:|---:|---:|---|
| 128x192x64 | V0/V1/V1.1/V2 | 6.663799e-4 | 5.642738e-3 | 8.344650e-7 | 8.325396e-7 | Pass |
| 1024³ | V0/V1/V1.1/V2 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 2048³ | V0/V1/V1.1/V2 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |

V1.1 与 V2 的直接逐元素比较：

| MxNxK | Max absolute difference | Max relative difference |
|---|---:|---:|
| 128x192x64 | 0 | 0 |
| 1024³ | 0 | 0 |
| 2048³ | 0 | 0 |

原有 `64x96x32` 测试中 V0/V1/V1.1 继续通过，V2 按设计跳过，因为 `N=96` 不是 64 的倍数。新增 `128x192x64` 的 V2 grid 为 `(3,2)`，覆盖六个 `64x64` block tiles；该非方形测试无 NaN、结果正确，验证了 block/warp 行列映射。

## 5. 编译资源、寄存器与 Occupancy

ptxas 输出：

```text
V1.1: 40 registers, 2304 bytes smem, 0-byte stack,
      0 spill stores, 0 spill loads
V2:   56 registers, 4096 bytes smem, 0-byte stack,
      0 spill stores, 0 spill loads
```

`cuobjdump --dump-resource-usage` 同时报告两个 kernel 的 `LOCAL:0`。NCU 的 local load/store SASS counters 也都是 0。

| 资源 | V1.1 | V2 | 变化/说明 |
|---|---:|---:|---|
| Registers/thread | 40 | 56 | +16，+40% |
| Registers/block（128 threads） | 5,120 | 7,168 | +2,048 |
| Spill loads/stores | 0 / 0 | 0 / 0 | 无实际 spill |
| Stack / local allocation | 0 / 0 B | 0 / 0 B | 无 local allocation |
| Kernel static shared memory | 2,304 B | 4,096 B | V2 的 A/B 各 2,048 B |
| Driver reserved shared/block | 1,024 B | 1,024 B | NCU Launch Statistics |
| Theoretical occupancy | 100% | 75% | V2 受 registers 限制 |
| Register block limit/SM | 12 blocks | 9 blocks | NCU occupancy calculator |
| Achieved occupancy（1024³） | 88.09% | 44.64% | V2 小网格也限制驻留工作量 |

**[Profiler 实测]** 四个 FP32 accumulator 与额外 A fragment 将寄存器从 40 提到 56，并把理论 active warps/SM 从 48 降到 36；这是明显的 register pressure 增加。但没有出现 spill，因此当前代价表现为 occupancy 降低，而不是 local-memory traffic。

## 6. 普通 CUDA Event Benchmark

### 6.1 中位结果

“相对 V1.1 speedup”定义为 `V1.1 median latency / 当前 kernel median latency`，因此 V2 大于 1 才表示 V2 更快。

| 尺寸 | Kernel | Median latency (ms) | Median TFLOPS | 相对 V1.1 speedup |
|---:|---|---:|---:|---:|
| 1024³ | V0 | 0.148898 | 14.423 | 0.8531x |
| 1024³ | V1 | 0.141761 | 15.149 | 0.8961x |
| 1024³ | V1.1 | 0.127031 | 16.905 | 1.0000x |
| 1024³ | V2 | 0.129789 | 16.546 | **0.9787x** |
| 2048³ | V0 | 1.105506 | 15.540 | 0.8227x |
| 2048³ | V1 | 0.992527 | 17.309 | 0.9163x |
| 2048³ | V1.1 | 0.909481 | 18.890 | 1.0000x |
| 2048³ | V2 | 0.769729 | 22.319 | **1.1816x** |

1024³ 使用两次独立进程、共 10 轮的聚合中位数。两次进程内的 V2/V1.1 speedup 分别为 0.9617x 和 0.9925x；方向一致，但退化幅度从 0.75% 到 3.83% 变化。因此结论是“没有 V2 收益，聚合结果小幅退化约 2.13%”，而不是宣称一个高精度的固定退化比例。

2048³ 的 V2 五轮范围是 0.761212–0.782643 ms，V1.1 是 0.901481–0.915171 ms，两区间完全分离；18.16% 的提升明显超过本轮波动。

### 6.2 原始 per-launch average

1024³，第一次独立进程：

- V0：0.147790, 0.152660, 0.147052, 0.148810, 0.148986 ms
- V1：0.142482, 0.138602, 0.142859, 0.138339, 0.141598 ms
- V1.1：0.126772, 0.129335, 0.124973, 0.129046, 0.126867 ms
- V2：0.133190, 0.127906, 0.131925, 0.131138, 0.132198 ms

1024³，第二次独立进程：

- V0：0.149183, 0.148731, 0.146683, 0.149155, 0.152947 ms
- V1：0.139915, 0.140333, 0.144034, 0.142595, 0.141924 ms
- V1.1：0.127194, 0.124876, 0.128135, 0.127262, 0.124948 ms
- V2：0.127498, 0.127511, 0.131829, 0.128159, 0.128440 ms

2048³：

- V0：1.102189, 1.101996, 1.105506, 1.108097, 1.121025 ms
- V1：0.989401, 1.008842, 0.992527, 0.991917, 1.007216 ms
- V1.1：0.909481, 0.911825, 0.901481, 0.905869, 0.915171 ms
- V2：0.765100, 0.761212, 0.772799, 0.769729, 0.782643 ms

## 7. PTX/SASS 与 Fragment Reuse

### 7.1 PTX 结构

V2 K-loop 中保留了预期的四条 WMMA load 和四条 WMMA MMA：

```text
wmma.load.a.sync.aligned.row.m16n16k16.shared.f16  // A0
wmma.load.a.sync.aligned.row.m16n16k16.shared.f16  // A1
wmma.load.b.sync.aligned.row.m16n16k16.shared.f16  // B0
wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32    // C00
wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32    // C10
wmma.load.b.sync.aligned.row.m16n16k16.shared.f16  // B1
wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32    // C01
wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32    // C11
```

### 7.2 SASS 结构与动态次数

ptxas 在 `sm_120` 上生成：

| 每个 K-loop 的静态核心指令 | V1.1 | V2 |
|---|---:|---:|
| A `LDSM.16.M88.4` | 1 | 2 |
| B `LDSM.16.MT88.4` | 1 | 2 |
| `HMMA.16816.F32` | 2 | 8 |

每个 WMMA `m16n16k16` MMA 在本工具链下生成两条 `HMMA.16816.F32`。V2 每 warp 每轮四个 WMMA，因此是八条 HMMA；这与 C00/C10/C01/C11 结构一致。

1024³ 动态计数解释了为何“V2 静态指令更多”并不代表总工作更多：

```text
V1.1: 1024 blocks x 4 warps x 64 K-iterations = 262,144 warp-iterations
V2:     256 blocks x 4 warps x 64 K-iterations =  65,536 warp-iterations
```

所以：

| 动态指标（1024³） | V1.1 | V2 | 变化 |
|---|---:|---:|---:|
| Fragment shared-load requests | 524,288 | 262,144 | -50% |
| HMMA SASS instructions | 524,288 | 524,288 | 0% |
| Global-load instructions/requests (`LDG.E.U16`) | 2,097,152 | 1,048,576 | -50% |
| Global-load sectors | 4,194,304 | 2,097,152 | -50% |
| Shared-store instructions (`STS.U16`) | 2,097,152 | 1,048,576 | -50% |
| Shared instructions total | 2,621,440 | 1,310,720 | -50% |
| Global-store instructions/requests (`STG.E.64`) | 16,384 | 16,384 | 0% |
| Global-store sectors | 131,072 | 131,072 | 0% |
| Executed instructions | 45,965,312 | 19,923,968 | -56.65% |
| Local load/store instructions | 0 / 0 | 0 / 0 | 无 spill traffic |

**[Profiler 实测]** 这组计数直接证明 A/B 双向 reuse 达到了目标：对相同 C 输出和相同 HMMA 总数，输入 staging 和 fragment-load 请求都减半。

## 8. Shared Memory Bank Conflict

### 8.1 文档语义与理论模型

**[文档事实]** NVIDIA [PTX ISA 的 `ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)说明 `ldmatrix.sync.aligned.m8n8.x4{.trans}.shared.b16` 由整个 warp 协作加载四个 8x8、16-bit 矩阵。当前 SASS 的 `LDSM.16.M88.4`/`LDSM.16.MT88.4` 分别对应非转置/转置型 shared matrix load，但 NVIDIA 没有公开 Blackwell SASS 的逐 bit、逐 lane 编码；本文不把 PTX 规则当作 SASS 私有编码的完整证明。

**[文档事实]** [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)说明：每条 shared instruction 形成一个 request，request 可被拆成多个串行 wavefront；不同地址映射到同一 bank 会产生 bank conflict。因而 source correlation 的 actual/ideal/excessive wavefronts 是本报告判断冲突的直接证据。

**[代码推导]** 在 32 banks、每 bank word 4 B 的简化模型中，V2 B 相邻行的起始 bank 偏移是：

```text
64 half x 2 B / 4 B = 32 bank words
32 mod 32 = 0
```

也就是说 16 行的起始 bank 都相同。这只能说明布局具有最短、最差的行起点周期；“实际是 32-way”仍必须由 profiler 证明。

### 8.2 逐条 LDSM 实测

| Kernel / 指令 | 动态 requests | N-way | Actual wavefronts | Ideal wavefronts | Excessive |
|---|---:|---:|---:|---:|---:|
| V1.1 A `LDSM.16.M88.4` | 262,144 | 8 | 2,097,152 | 1,048,576 | 1,048,576 |
| V1.1 B `LDSM.16.MT88.4` | 262,144 | 4 | 1,048,576 | 1,048,576 | 0 |
| V2 A0 `LDSM.16.M88.4` | 65,536 | 8 | 524,288 | 262,144 | 262,144 |
| V2 A1 `LDSM.16.M88.4` | 65,536 | 8 | 524,288 | 262,144 | 262,144 |
| V2 B0 `LDSM.16.MT88.4` | 65,536 | 32 | 2,097,152 | 262,144 | 1,835,008 |
| V2 B1 `LDSM.16.MT88.4` | 65,536 | 32 | 2,097,152 | 262,144 | 1,835,008 |

这里 V1.1 B 的 “4” 是该 `.4` matrix load 的 intrinsic ideal wavefront 数，不是四倍额外冲突；关键是 actual=ideal、excessive=0。V2 B 则是 actual 32、ideal 4，每 request 多 28 个 wavefront。

### 8.3 Aggregate shared 指标

| 指标（1024³） | V1.1 | V2 | 变化 |
|---|---:|---:|---:|
| Shared-load requests | 524,288 | 262,144 | -50% |
| Source ideal load wavefronts | 2,097,152 | 1,048,576 | -50% |
| Source actual load wavefronts | 3,145,728 | 5,242,880 | +66.67% |
| Source excessive load wavefronts | 1,048,576 | 4,194,304 | +300% |
| HW shared-load actual wavefront counter | 3,207,754 | 5,257,140 | +63.89% |
| HW shared-load bank conflicts | 1,109,956 | 4,208,214 | +279.13%（3.79x） |
| NCU avg. actual wavefront/request | 6.1 | 20.1 | 明显恶化 |
| Shared-store actual wavefronts | 2,118,127 | 1,048,576 | 约 -50.5% |
| Shared-store bank conflicts | 18,835 | 0 | 未产生新 store conflict |

source correlation 的整数结果与 aggregate HW counter 有小幅差异，这是 replay passes/指标归因差异；逐条 A/B 判断使用 source actual/ideal/excessive，整体比较同时保留 HW counter。

结论：**V2 虽把 fragment-load requests 减半，但无 Padding 的 stride 64 使 B load 严重序列化，最终 actual load wavefront 反而增加约 64%。没有产生新的 shared-store bank conflict；V2 的 store 数量减半且本次 counter 为 0。**

## 9. Stall、Tensor Pipe 与瓶颈

下表均为 1024³、相同 NCU 条件；stall 数值是 average warp cycles per issued instruction 对应的 ratio：

| 指标 | V1.1 | V2 | 观察 |
|---|---:|---:|---|
| Long Scoreboard | 9.2810 | 19.0244 | +105%；V2 latency hiding 更差 |
| Short Scoreboard | 1.0423 | 3.9123 | +275%；与 shared dependency/conflict 一致 |
| MIO Throttle | 0.5909 | 0.7354 | +24.4% |
| LG Throttle | 0.0227 | 0.1068 | 绝对值仍小，但上升 |
| Barrier | 0.7454 | 1.0621 | 上升；barrier 位置和每 K 轮次数未改变 |
| Tensor pipe cycles active | 25.07% | 15.51% | -9.56 percentage points |
| HMMA subpipe cycles active | 25.11% | 15.51% | 未提高 |
| Active warps/scheduler | 10.58 | 5.31 | 约减半 |
| Eligible warps/scheduler | 1.61 | 0.26 | V2 经常没有可发射 warp |
| Issue slots busy | 60.36% | 18.06% | 小网格/低 occupancy 明显欠填充 |
| Grid blocks | 1,024 | 256 | V2 每 block 计算 4 倍 C 面积 |
| Waves per SM | 1.78 | 0.59 | V2 1024³ grid 太小 |

**[文档事实]** NCU 将 long scoreboard 定义为等待 L1TEX（包括 global/local）依赖，将 short scoreboard 通常关联到 MIO/shared-memory 依赖；MIO throttle 可由 shared instruction 扩展为大量内部操作引起。该定义支持把上表与 global/shared load 延迟联系起来，但单一 stall counter 不能独立证明因果。

**[Profiler 实测]** 1024³ 的主要限制是：

1. V2 grid 只有 256 blocks，NCU 明确报告 0.59 full waves/SM，不能填满 48 个 SM。
2. 56 registers/thread 把理论 occupancy 限到 75%，实测 achieved occupancy 仅 44.64%。
3. 两条 B LDSM 的 32-way pattern 使实际 shared-load wavefront 数高于 V1.1，即使请求数已减半。
4. long/short scoreboard 都显著增加，可发射 warps/scheduler 下降到 0.26。

**[解释/假设]** 2048³ 上 V2 有 1,024 blocks，较 1024³ 的 256 blocks 更能摊薄 tail/underfill；同时输入 staging、fragment requests 和总动态指令减半的收益开始超过 occupancy 与 B conflict 的代价。这与 2048³ 的 18.16% Event speedup一致，但本轮 NCU 只采集了 1024³，因此不把 2048³ Tensor-pipe/occupancy 的具体数值作为实测结论。

## 10. 回答关键问题

1. **V2 是否减少重复 Fragment Load？** 是。Shared-to-Fragment requests 从 524,288 降到 262,144，正好减半；global input load 和 shared staging store 也减半。
2. **HMMA 数量是否符合四次 MMA？** 是。每个 V2 warp iteration 有 4 个 WMMA，每个生成 2 条 HMMA，共 8 条；动态总数与 V1.1 一样是 524,288。
3. **Register pressure 是否增加？是否 spill？** 从 40 增到 56 registers/thread，理论 occupancy 100% 降到 75%；但 spill loads/stores 和 local memory 均为 0。
4. **Occupancy 是否下降？** 是。1024³ achieved occupancy 从 88.09% 降到 44.64%，同时受到 register limit 与 0.59 waves/SM 小网格影响。
5. **无 Padding `Bs[16][64]` 是否有严重 conflict？** 是。两个 B LDSM 都是 32 actual / 4 ideal wavefronts，每条产生 1,835,008 excessive wavefronts。
6. **是否产生新 shared-store conflict？** 没有。本次 V2 store bank-conflict counter 为 0；store instructions/wavefronts约减半。
7. **Tensor Core 利用率是否提高？** 在本轮 1024³ profiler 中没有，HMMA subpipe active 从 25.11% 降到 15.51%。2048³ 未采 NCU，不能给出该尺寸的数值结论。
8. **主要瓶颈是什么？** 1024³ 是小网格欠填充、register-limited occupancy 和 B LDSM bank conflict/scoreboard latency 的组合。2048³ 上复用带来的 load/instruction 削减已足以产生明显总收益。
9. **V2 最终是否优于 V1.1？** 取决于尺寸：1024³ 小幅退化约 2.13%；2048³ 明显提升 18.16%。

## 11. 下一步建议（未实施）

本轮没有继续修改 V2。若开展下一轮独立实验，优先候选是只改变 V2 B 的物理 stride，例如单独验证一个满足 WMMA 对齐约束的 padding stride，并保持 tile、warp mapping、fragment 组织和同步全部不变。这样可以隔离回答：去除当前两个 32-way B LDSM conflict 后，V2 在 1024³ 是否转为获益、2048³ 是否进一步提升。

应先做 B padding 单变量对照，再单独研究 register pressure；不要在同一版本同时重组 accumulators、加 padding 或引入 pipeline，否则无法归因。A 的两个 `LDSM.16.M88.4` 仍各有 8 actual/4 ideal wavefront，也可作为后续独立研究对象，但不应与 B padding 混在同一实验。

本报告只提出候选，没有实现 V2.1，也没有提交 Git commit。
