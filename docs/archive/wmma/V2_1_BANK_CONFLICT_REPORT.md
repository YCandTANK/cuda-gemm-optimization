# Tensor Core V2.1：B Padding 72 Bank Conflict 实验报告

日期：2026-10-12（Asia/Shanghai）

## 1. 实验结论

本实验只改变一个变量：V2.1 将 V2 Shared B 的物理行跨度从 64 个 `half` 改为 72 个 `half`。逻辑 B tile 仍为 `16x64`；block/warp/MMA tile、A layout、cooperative loading 的有效元素、fragment reuse、四个 accumulator、同步和 C store 全部保持不变。

结论明确：**Padding 72 消除了两个 B `LDSM.16.MT88.4` 的额外 wavefront，并带来显著性能提升；建议把 V2.1 保留为当前项目中已测试版本的最佳实现。**

关键实测结果：

- V2.1 在 `128x192x64`、`1024^3`、`2048^3` 上正确；与 V2 逐元素最大绝对/相对差均为 0。
- 两个 B `LDSM.16.MT88.4` 均从 V2 的 `32 actual / 4 ideal` 降为 `4 actual / 4 ideal`，每条指令的 excessive wavefronts 从 1,835,008 降为 0。
- 两个 A `LDSM.16.M88.4` 均保持 `8 actual / 4 ideal`，说明 B padding 没有改变 A 的访问行为。
- aggregate shared-load bank conflicts 从本轮 V2 的 4,208,358 降到 530,059，下降 87.40%；按 shared-load request 归一化后从 16.054 降到 2.022 conflicts/request。
- shared-load actual wavefronts 从 5,256,856 降到 1,578,561，下降 69.97%；source-correlated actual/request 从精确的 20 降到 6，ideal/request 均为 4。
- shared-store bank conflicts 在 V2 和 V2.1 中均为 0，没有引入新 store conflict。
- V2/V2.1 均为 56 registers/thread、0 spill、0 local-memory traffic、75% theoretical occupancy。V2.1 仅增加 256 B static shared memory，理论和实测 occupancy 未下降。
- 普通 CUDA Event benchmark：
  - 1024³：V2.1 0.102633 ms、20.924 TFLOPS，相对 V2 为 **1.2628x**。
  - 2048³：V2.1 0.611262 ms、28.106 TFLOPS，相对 V2 为 **1.2702x**。
- 1024³ NCU 中 HMMA subpipe active 从 15.40% 增到 17.89%；Short Scoreboard 降 67.12%，MIO Throttle 降 74.80%。

上述性能差距在两种尺寸的五轮数据中都完全分离，不属于接近正常波动的小差异。

## 2. 证据分类与实验范围

本文使用以下标签：

- **[文档事实]**：NVIDIA 公开 CUDA/PTX/NCU 文档支持。
- **[理论推导]**：由简化 bank mapping 或代码结构推出，不能替代 profiler。
- **[实测]**：来自本轮 CUDA Event、ptxas、SASS 或 Nsight Compute。
- **[解释]**：对指标与性能的因果解释，明确保留不确定性。

本轮没有实施或搜索其他 padding，也没有加入 A padding、XOR swizzle、转置 layout、double buffering、async copy、手写 PTX或其他优化。

## 3. 环境与方法

- GPU：NVIDIA GeForce RTX 5070，compute capability 12.0，12,227 MiB
- Driver：596.49
- CUDA compiler：13.2，`nvcc` 13.2.86
- Nsight Compute：2026.1.1
- 编译目标：`sm_120`
- 编译参数：`-std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v --ptxas-options=-warn-spills`

普通 benchmark：

- V1.1、V2、V2.1 使用同一最终二进制、相同输入和 device allocations。
- 每个 kernel 每轮 10 次 warmup、100 次 timed launches，共 5 轮。
- 每轮旋转 V0/V1/V1.1/V2/V2.1 的测试顺序，降低固定执行顺序偏差。
- CUDA Event 只包围 kernel launches；CPU reference、传输和 profiler replay 不计入普通性能排名。

Nsight Compute：

- V2/V2.1 均分析 1024³，同一最终二进制和 `--set full`，每个 kernel 只采一个匹配 launch，40 replay passes。
- V2 报告：`build/ncu_v2_v2_1_compare.ncu-rep`
- V2.1 报告：`build/ncu_v2_1.ncu-rep`
- SASS：`build/tensor_core_gemm_v2_1.sass`
- PTX：`build/tensor_core_gemm_v2_1.ptx`

Profiler duration 只作辅助观察，不用于普通性能排名。

## 4. Padding 72 的理论依据与合法性

### 4.1 简化 bank mapping

**[理论推导]** 以 32 banks、每个 bank word 4 B 的简化模型：

```text
V2 stride 64 half:
64 x 2 B / 4 B = 32 bank words
32 mod 32 = 0

V2.1 stride 72 half:
72 x 2 B / 4 B = 36 bank words
36 mod 32 = 4
```

V2 的相邻行起始 bank 不移动；V2.1 连续八行起始 bank 为：

```text
0, 4, 8, 12, 16, 20, 24, 28
```

这给出了测试 stride 72 的理由，但不能提前证明 `LDSM` 会达到 ideal；真实结果必须看 source correlation 的 actual/ideal/excessive wavefronts。

### 4.2 WMMA 合法性

**[文档事实]** NVIDIA [CUDA Programming Guide 的 WMMA 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)要求 `load_matrix_sync` 的 `mptr` 具有 256-bit 对齐；`half` 的 `ldm` 必须是 8 个元素的倍数。

**[代码推导]** V2.1 满足这些条件：

- `Bs` 使用 `__align__(32)`。
- `B_SHARED_STRIDE_V2_1=72` 是 8 的倍数，物理行跨度为 144 B，也是 16 B 的倍数。
- B0/B1 的逻辑列起点仍为 0/16 或 32/48 个 half；相对 shared base 的偏移均为 32 B 的整数倍。
- A 的 base、layout、指针和 `ldm=16` 完全不变。
- 编译期 `static_assert` 同时检查 stride 至少覆盖 64 个有效列且满足 half WMMA 的 ldm 约束。

## 5. 核心源码差异与 Host 集成

V2.1 相对 V2 的 kernel 差异仅为：

```cpp
constexpr int B_SHARED_STRIDE_V2_1 = 72;

__shared__ __align__(32) half As[BLOCK_M_V2][BLOCK_K_V2];
__shared__ __align__(32) half Bs[BLOCK_K_V2][B_SHARED_STRIDE_V2_1];

for (int idx = tid; idx < BLOCK_K_V2 * BLOCK_N_V2;
     idx += blockDim.x) {
    int row_Bs = idx / BLOCK_N_V2;
    int col_Bs = idx % BLOCK_N_V2;
    // Global B addressing remains unchanged.
    Bs[row_Bs][col_Bs] = B[row_B * N + col_B];
}

wmma::load_matrix_sync(b, B0_ptr, B_SHARED_STRIDE_V2_1);
// same two MMA operations
wmma::load_matrix_sync(b, B1_ptr, B_SHARED_STRIDE_V2_1);
// same two MMA operations
```

因此只写入每行 `[0,63]` 的 64 个有效元素；每行 `[64,71]` 的 padding 未初始化，但没有任何 fragment 读取这些列。

Host 端增加了：

- `KernelVersion::V2_1` 与名称。
- 与 V2 相同的 `(N/64, M/64)` grid 和 128-thread launch。
- 两套 reference、NaN 预填、V2/V2.1 直接逐元素比较。
- V2.1 benchmark 输出和 V2.1/V2 speedup。
- 原有 `64x96x32` 继续跳过 V2/V2.1；`128x192x64` 同时检查 V2/V2.1 的非方形 mapping。

V0、V1、V1.1、V2 的核心 kernel 未修改。

## 6. Correctness 与内存安全检查

Reference 1 使用原始 FP32 输入执行 FP32 CPU GEMM，包含输入 FP16 量化误差；Reference 2 将实际 FP16 输入转换回 FP32 后执行 CPU GEMM，用于判定 kernel。容差规则没有修改：

```text
abs_error <= 1e-2 + 1e-2 * abs(reference)
```

每次 kernel 前将整个 C 用 NaN pattern 填充，结果必须全部 finite。V2 与 V2.1 使用相同输入，误差统计如下：

| MxNxK | Kernel | Ref.1 max abs | Ref.1 max rel | Ref.2 max abs | Ref.2 max rel | 结果 |
|---|---|---:|---:|---:|---:|---|
| 128x192x64 | V2 | 6.663799e-4 | 5.642738e-3 | 8.344650e-7 | 8.325396e-7 | Pass |
| 128x192x64 | V2.1 | 6.663799e-4 | 5.642738e-3 | 8.344650e-7 | 8.325396e-7 | Pass |
| 1024³ | V2 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 1024³ | V2.1 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 2048³ | V2 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |
| 2048³ | V2.1 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |

V2/V2.1 直接比较：

| MxNxK | Max absolute difference | Max relative difference |
|---|---:|---:|
| 128x192x64 | 0 | 0 |
| 1024³ | 0 | 0 |
| 2048³ | 0 | 0 |

三种尺寸均无残留 NaN、非有限结果、launch error 或 synchronize error。代码索引审查表明，在 Host 限制的 `M,N` 为 64 倍数、`K` 为 16 倍数时，global/shared 访问均在范围内。

Compute Sanitizer memcheck 也被尝试，但当前 WDDM/RTX 5070 环境报告 `Failed to initialize WDDM debugger interface` 和 `Device not supported`，因此不能把 sanitizer 作为通过证据。这里的有效证据是边界推导、三尺寸逐元素 reference、NaN/finite 检查和每次 launch 后的 CUDA error/synchronize 检查。

## 7. 编译资源与 SASS

### 7.1 资源使用

| 指标 | V2 | V2.1 | 变化 |
|---|---:|---:|---:|
| Registers/thread | 56 | 56 | 0 |
| Registers/block（128 threads） | 7,168 | 7,168 | 0 |
| Spill loads/stores | 0 / 0 | 0 / 0 | 0 |
| Stack/local allocation | 0 / 0 B | 0 / 0 B | 0 |
| Kernel static shared memory | 4,096 B | 4,352 B | +256 B |
| Driver reserved shared/block | 1,024 B | 1,024 B | 0 |
| Theoretical occupancy | 75% | 75% | 0 |
| Register block limit/SM | 9 | 9 | 0 |
| Shared-memory block limit/SM | 20 | 19 | -1，但非首要限制 |
| Achieved occupancy（1024³） | 44.49% | 44.55% | 实质不变 |

寄存器仍是 theoretical occupancy 的限制项，因此增加 256 B shared memory 没有进一步降低 occupancy。NCU local load/store SASS counters 也都是 0。

### 7.2 指令形态与动态计数

V2 与 V2.1 每个 K-loop 都生成相同的核心 SASS 形态：

```text
2 x LDSM.16.M88.4       // A0, A1
2 x LDSM.16.MT88.4      // B0, B1
8 x HMMA.16816.F32      // four WMMA MMA operations
```

Global/shared staging 和 C store 仍使用相同类型的 `LDG.E.U16`、`STS.U16`、`STG.E.64`。变化位于 B shared 地址/stride 计算，而不是访存或 MMA 指令种类。

| 动态指标（1024³） | V2 | V2.1 |
|---|---:|---:|
| Executed instructions | 19,923,968 | 19,923,968 |
| Global-load instructions/requests | 1,048,576 | 1,048,576 |
| Global-load sectors | 2,097,152 | 2,097,152 |
| Shared instructions total | 1,310,720 | 1,310,720 |
| Shared-store instructions | 1,048,576 | 1,048,576 |
| Fragment shared-load requests | 262,144 | 262,144 |
| HMMA instructions | 524,288 | 524,288 |
| Global-store instructions/requests | 16,384 | 16,384 |
| Global-store sectors | 131,072 | 131,072 |
| Local load/store instructions | 0 / 0 | 0 / 0 |

**[实测]** Padding 没有增加寄存器、spill 或动态指令数量；除 256 B shared allocation 和预期的地址 stride 外，没有新的明显资源/指令开销。

## 8. CUDA Event Benchmark

### 8.1 中位结果

V2.1/V2 speedup 定义为 `V2 median latency / V2.1 median latency`。

| 尺寸 | Kernel | Median latency (ms) | Median TFLOPS | V2.1/V2 speedup |
|---:|---|---:|---:|---:|
| 1024³ | V1.1 | 0.127006 | 16.909 | - |
| 1024³ | V2 | 0.129607 | 16.569 | 1.0000x |
| 1024³ | V2.1 | 0.102633 | 20.924 | **1.2628x** |
| 2048³ | V1.1 | 0.905552 | 18.972 | - |
| 2048³ | V2 | 0.776408 | 22.127 | 1.0000x |
| 2048³ | V2.1 | 0.611262 | 28.106 | **1.2702x** |

作为版本曲线补充，V2.1 相对本轮重新测得的 V1.1 分别为 1.2375x（1024³）和 1.4814x（2048³）。没有使用历史 V1.1 数据。

### 8.2 五轮原始 per-launch average

1024³：

- V1.1：0.128941, 0.125129, 0.127006, 0.128680, 0.125155 ms
- V2：0.129607, 0.128627, 0.127815, 0.134226, 0.134286 ms
- V2.1：0.102580, 0.104474, 0.104814, 0.102453, 0.102633 ms

2048³：

- V1.1：0.900926, 0.905552, 0.914392, 0.913034, 0.898243 ms
- V2：0.761735, 0.769247, 0.776408, 0.776781, 0.784608 ms
- V2.1：0.606502, 0.604428, 0.628735, 0.611262, 0.613844 ms

1024³ 中 V2.1 最慢一轮 0.104814 ms 仍比 V2 最快一轮 0.127815 ms 快；2048³ 也同样区间完全分离。因此 26%–27% 的收益远大于本轮波动，无需追加更多普通测量来判断方向。

## 9. Nsight Compute：V2 与 V2.1

### 9.1 逐条 LDSM source correlation

**[文档事实]** NVIDIA [PTX ISA 的 `ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)描述了 warp 协作的 `.m8n8.x4{.trans}.shared.b16` matrix load；[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)说明一个 shared request 可拆成多个串行 wavefront，actual 相对 ideal 的增加代表额外服务工作。公开文档未给出 Blackwell `LDSM` SASS 的完整私有编码，因此逐 lane 解释不超出公开 PTX 语义和实际 source counters。

| 指令 | V2 actual / ideal / excessive | V2.1 actual / ideal / excessive |
|---|---:|---:|
| A0 `LDSM.16.M88.4` | 524,288 / 262,144 / 262,144（8-way） | 相同 |
| A1 `LDSM.16.M88.4` | 524,288 / 262,144 / 262,144（8-way） | 相同 |
| B0 `LDSM.16.MT88.4` | 2,097,152 / 262,144 / 1,835,008（32-way） | 262,144 / 262,144 / 0（4-way ideal） |
| B1 `LDSM.16.MT88.4` | 2,097,152 / 262,144 / 1,835,008（32-way） | 262,144 / 262,144 / 0（4-way ideal） |

每条指令执行 65,536 次。V2.1 的两个 B load 都达到 actual=ideal；这里的 “4-way” 是 `.4` matrix load 固有的四个 ideal wavefront，不是四倍额外 bank conflict。

### 9.2 Aggregate 与归一化 shared 指标

| 指标（1024³） | V2 | V2.1 | 变化 |
|---|---:|---:|---:|
| Shared-load requests | 262,144 | 262,144 | 0 |
| Source ideal load wavefronts | 1,048,576 | 1,048,576 | 0 |
| Source actual load wavefronts | 5,242,880 | 1,572,864 | -70.00% |
| Source excessive load wavefronts | 4,194,304 | 524,288 | -87.50% |
| HW actual load wavefront counter | 5,256,856 | 1,578,561 | -69.97% |
| HW shared-load bank conflicts | 4,208,358 | 530,059 | -87.40% |
| HW actual wavefronts/load request | 20.053 | 6.022 | -69.97% |
| Source ideal wavefronts/load request | 4.000 | 4.000 | 0 |
| Source excessive wavefronts/load request | 16.000 | 2.000 | -87.50% |
| HW bank conflicts/load request | 16.054 | 2.022 | -87.40% |
| Shared-store requests/instructions | 1,048,576 | 1,048,576 | 0 |
| Shared-store actual wavefronts | 1,048,576 | 1,048,576 | 0 |
| Shared-store bank conflicts | 0 | 0 | 无新增冲突 |

绝对值和归一化指标给出同一结论，因为两者的 shared-load request 数完全相同。V2.1 仍有约 2 excessive wavefronts/request，全部来自两个未改变的 A load；B 的 excessive 已归零。

aggregate HW counter 与 source-correlated 整数存在小幅 replay/归因差异，因此逐条 A/B 结论使用 source actual/ideal/excessive，整体变化同时保留 HW counter。

### 9.3 Stall、Tensor pipe 与其他指标

stall 数值为 NCU 的 average warp cycles per issued instruction ratio：

| 指标（1024³） | V2 | V2.1 | 变化 |
|---|---:|---:|---:|
| Long Scoreboard | 19.1817 | 18.3760 | -4.20% |
| Short Scoreboard | 3.9055 | 1.2843 | -67.12% |
| MIO Throttle | 0.7270 | 0.1832 | -74.80% |
| LG Throttle | 0.1023 | 0.0819 | -19.91% |
| Barrier | 1.0455 | 0.6939 | -33.63% |
| Tensor pipe cycles active | 15.40% | 17.88% | +2.48 percentage points |
| HMMA subpipe cycles active | 15.40% | 17.89% | +2.49 percentage points（相对 +16.15%） |
| Achieved occupancy | 44.49% | 44.55% | 实质不变 |
| Active warps/scheduler | 5.290 | 5.323 | 实质不变 |
| Eligible warps/scheduler | 0.264 | 0.286 | +8.25% |
| Issue slots busy | 17.93% | 20.34% | +2.41 percentage points |
| Local load/store | 0 / 0 | 0 / 0 | 不变 |
| Global load requests/sectors | 1,048,576 / 2,097,152 | 相同 | 不变 |
| Global store requests/sectors | 16,384 / 131,072 | 相同 | 不变 |

**[文档事实]** NCU 将 Short Scoreboard 通常关联到 MIO/shared-memory dependency，并指出 highly scattered shared access 会增加 MIO queue 压力。当前 B excessive wavefronts 归零，同时 Short Scoreboard 和 MIO Throttle 大幅下降，方向和机制一致。

Long Scoreboard 只下降 4.2%，说明 global/L1TEX dependency 仍是主要延迟来源之一；Padding 72 没有改变 global load 数量，也不应期望消除这一项。1024³ 的 grid 仍只有 256 blocks、0.59 waves/SM，所以 Tensor pipe 尚未接近饱和。即便如此，去掉 B serialization 后 HMMA active 和 issue-slot utilization 都提高。

## 10. 因果分析与不确定性

支持“性能提升主要来自 B bank-conflict 降低”的证据链：

1. **控制变量明确：** kernel 中唯一计算相关变化是 B shared stride 64→72 以及对应 WMMA ldm。
2. **工作量不变：** global/shared requests、dynamic instructions、HMMA 数量、registers 和 occupancy 相同。
3. **逐指令定位：** B0/B1 从 32 actual 降到 4 ideal，A0/A1 完全不变。
4. **没有转移代价：** shared-store conflicts 保持 0，spill/local traffic 保持 0，theoretical occupancy 保持 75%。
5. **相关 stall 同步改善：** Short Scoreboard -67.12%、MIO Throttle -74.80%。
6. **普通性能改善稳定：** 两个尺寸上均约 1.27x，五轮区间完全分离。

因此，把主要收益归因于 B LDSM serialization 的消除具有很强证据。不过仍有两点不确定性：

- NCU 使用 replay passes，不同 counter 会在不同 pass 收集；aggregate 与 source 数值有小幅差异。
- 当前结论覆盖指定的三种正确性尺寸和两种 benchmark 尺寸，不自动代表所有矩阵形状或其他 GPU。

本轮没有出现“冲突下降但性能不升”的情况。如果未来在其他尺寸出现这种现象，首先应检查小 grid/tail effect、register-limited occupancy、A 的剩余 conflict、global-load Long Scoreboard 和频率波动，而不能只看 bank-conflict 总数。

## 11. 是否推荐保留 V2.1

**推荐保留 V2.1，作为当前已测试版本的最佳实现。** 理由是：

- 正确性与 V2 完全一致。
- B 两条 LDSM 都达到 actual=ideal。
- aggregate load conflicts 降低 87.40%，没有新 store conflict。
- registers、spill、dynamic instruction count 和 occupancy 均未恶化。
- 唯一资源代价是每 block 增加 256 B static shared memory，当前不改变 occupancy 限制项。
- 1024³ 和 2048³ 分别稳定提升 26.28% 和 27.02%，并且都优于本轮重新测量的 V1.1。

这是一项针对当前 RTX 5070/sm_120 和指定尺寸的实验结论，而不是对任意架构、尺寸或 stride 的普遍保证。

本轮未提交 Git commit，未开始 V2.2，也未测试其他 padding 或 cuBLAS。
