# CUDA GEMM 最终性能与 cuBLAS 对照报告

实验日期：2026-10-12

## 1. 结论摘要

本项目完成了两条彼此独立的优化路线：FP32 CUDA-Core GEMM，以及 FP16 输入、FP32 累加/输出的 WMMA Tensor Core GEMM。两条路线的数据类型和硬件执行单元不同，因此不把它们的 TFLOPS 直接用于算法优劣比较。

本轮新增的是公平的 `cublasGemmEx()` 基线，没有继续修改任何 GEMM kernel。V2.1 和 cuBLAS 使用相同 FP16 A/B、相同 FP32 C、相同 device allocations、相同默认 stream，并在每一轮反转测试顺序。结果如下：

| 尺寸 M×N×K | V2.1 median | V2.1 | cuBLAS median | cuBLAS | V2.1/cuBLAS |
|---|---:|---:|---:|---:|---:|
| 512×512×512 | 0.048391 ms | 5.547 TFLOPS | 0.012660 ms | 21.203 TFLOPS | 26.16% |
| 1024×1024×1024 | 0.102796 ms | 20.891 TFLOPS | 0.040419 ms | 53.130 TFLOPS | 39.32% |
| 2048×2048×2048 | 0.607552 ms | 28.277 TFLOPS | 0.268906 ms | 63.888 TFLOPS | 44.26% |
| 2048×1024×2048 | 0.393454 ms | 21.832 TFLOPS | 0.138450 ms | 62.044 TFLOPS | 35.19% |

2048³ 的主结果采用两次独立进程共 10 个样本的聚合中位数；原因是首次进程的 V2.1 第一轮出现 0.702986 ms 离群值。第二组五轮全部位于 0.601490–0.607746 ms，验证了总体结论。这里没有选取最快一轮。

所有指定尺寸均通过正确性检查，没有 NaN/Inf、漏写 tile、越界或 Row-major 映射错误。对本组确定性输入，cuBLAS 与 V2.1 恰好逐元素一致；这只是实测结果，不是接口所保证的 bitwise 等价性。

## 2. 环境与版本状态

### 2.1 硬件和工具链

| 项目 | 实测/记录值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 12 GB |
| Compute Capability | 12.0 |
| NVIDIA Driver | 596.49 |
| CUDA Toolkit / nvcc | 13.2 / 13.2.86 |
| CUDA runtime / driver API | 13020 / 13020 |
| cuBLAS runtime | 130401（13.4.1） |
| cuBLAS header | 13.4.1 |
| Nsight Compute | 2026.1.1.0 |
| 编译目标 | `sm_120` |

编译命令：

```bash
nvcc -O3 -lineinfo -arch=sm_120 -Xptxas=-v \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_final -lcublas

nvcc -O3 -lineinfo -arch=sm_120 -Xptxas=-v \
  src/gemm.cu -o build/gemm_fp32_final -lcublas
```

Tensor Core 编译资源保持不变：V1.1 为 40 registers/thread、2304 B static shared memory；V2.1 为 56 registers/thread、4352 B static shared memory；两者均为 0 spill loads/stores、0-byte stack frame。

### 2.2 Git 状态

实验开始时 `HEAD` 为 `02e3805 perf: optimize tensor core GEMM with B padding`。已有四份 Tensor Core 报告处于 untracked 状态：

- `V1_PERFORMANCE_REPORT.md`
- `V1_1_BANK_CONFLICT_REPORT.md`
- `V2_PERFORMANCE_REPORT.md`
- `V2_1_BANK_CONFLICT_REPORT.md`

本轮保留了这些文件，没有 commit、reset、clean 或覆盖无关内容。最终状态见本文末尾及任务交付摘要。

## 3. cuBLAS Baseline 的公平性

### 3.1 数值配置

主要基线严格使用：

```cpp
cublasGemmEx(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_B, CUDA_R_16F, N,
    d_A, CUDA_R_16F, K,
    &beta,
    d_C, CUDA_R_32F, N,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT);
```

其中 `alpha=1.0f`、`beta=0.0f`。没有使用 FP32 输入的 TF32 模式，也没有使用 FP16 输出或 FP16 累加。没有加入 cuBLASLt 或算法调参，因此本文的“cuBLAS”始终指上述默认 `cublasGemmEx` 基线。

### 3.2 为什么参数顺序是 B、A

项目矩阵在内存中是 Row-major，而传统 cuBLAS 将同一片线性内存解释为 Column-major。利用：

```text
C_row = A_row B_row
C_row^T = B_row^T A_row^T
```

Row-major 的 `B[K,N]` 缓冲区可被视为 Column-major 的 `B^T[N,K]`，Row-major 的 `A[M,K]` 可被视为 Column-major 的 `A^T[K,M]`。因此 cuBLAS 计算一个 `N×M` 的 Column-major 结果 `B^T A^T`，其线性内存正好就是所需 `M×N` Row-major C。传入 `B, A` 和 `N, M, K` 即可，不需要实际转置或额外数据搬运。

### 3.3 计时边界

**[代码事实]** 每个进程只创建一次 handle，并在计时前设置默认 stream 和 Host pointer mode。输入初始化、FP16 转换、两套 CPU reference、H2D/D2H、handle 创建均不在 Event 区间内。

**[实测方法]** 每个实现每轮先 warmup 10 次，再在两个 CUDA Event 之间连续执行 100 次，记录 per-launch average；共 5 轮。版本数组为 `[V2.1, cuBLAS]`，轮次每增加一次就旋转顺序，因此测试顺序是 V2.1→cuBLAS、cuBLAS→V2.1 交替。两者使用同一 `d_A/d_B/d_C` 和同一默认 stream。若 cuBLAS 内部需要多个 GPU kernel，它们也都处于同一对 Event 之间，计时覆盖完成 GEMM 的全部 GPU stream 工作。

## 4. Correctness

测试先用原始 FP32 输入做 FP32 CPU GEMM，再把 FP16 量化输入转回 FP32做第二套 CPU GEMM。第二套 reference 用于判定 Tensor Core/cuBLAS 正确性，混合容差为：

```text
abs_error <= 1e-2 + 1e-2 * abs(reference)
```

每次 GPU 运行前都用 `0xff` 将 C 填成 NaN，随后要求所有 reference、结果和误差均为 finite。

| 尺寸 | 实现 | 对原始 FP32 Ref 最大绝对/相对误差 | 对量化 Ref 最大绝对/相对误差 | V2.1/cuBLAS 最大差异 | 结论 |
|---|---|---:|---:|---:|---|
| 128×192×64 | V2.1 | 6.663799e-4 / 5.642738e-3 | 8.344650e-7 / 8.325396e-7 | — | Pass |
| 128×192×64 | cuBLAS | 6.663799e-4 / 5.642738e-3 | 8.344650e-7 / 8.325396e-7 | 0 / 0 | Pass |
| 1024³ | V2.1 | 7.154942e-4 / 1.094607e-3 | 6.079674e-6 / 1.020690e-5 | — | Pass |
| 1024³ | cuBLAS | 7.154942e-4 / 1.094607e-3 | 6.079674e-6 / 1.020690e-5 | 0 / 0 | Pass |
| 2048³ | V2.1 | 9.616613e-4 / 8.727074e-2 | 1.084805e-5 / 3.015384e-4 | — | Pass |
| 2048³ | cuBLAS | 9.616613e-4 / 8.727074e-2 | 1.084805e-5 / 3.015384e-4 | 0 / 0 | Pass |
| 2048×1024×2048 | V2.1 | 6.392002e-4 / 1.440107e-2 | 1.204014e-5 / 3.733509e-4 | — | Pass |
| 2048×1024×2048 | cuBLAS | 6.392002e-4 / 1.440107e-2 | 1.204014e-5 / 3.733509e-4 | 0 / 0 | Pass |

原始 FP32 reference 的误差主要包括 FP32→FP16 输入量化；量化 reference 的误差才反映 GEMM 累加差异。较大的最大相对误差出现在接近零的 reference 元素，绝对误差仍很小。一般情况下 cuBLAS 和 WMMA 的 reduction/累加顺序可以不同，因此正确要求是合理容差内一致，而不是 bitwise 相等。

## 5. V2.1 与 cuBLAS Benchmark

### 5.1 五轮原始时间

单位均为 ms/launch。

- 512³：
  - V2.1：0.047971, 0.048391, 0.048429, 0.048083, 0.049848
  - cuBLAS：0.012660, 0.012729, 0.012474, 0.012889, 0.012623
- 1024³：
  - V2.1：0.102061, 0.105619, 0.102796, 0.108484, 0.102424
  - cuBLAS：0.044121, 0.039836, 0.040539, 0.040367, 0.040419
- 2048³，首次进程：
  - V2.1：0.702986, 0.615830, 0.608515, 0.610121, 0.604988；进程内 median 0.610121
  - cuBLAS：0.266216, 0.266000, 0.271051, 0.268617, 0.267522；进程内 median 0.267522
- 2048³，确认进程：
  - V2.1：0.606064, 0.607357, 0.607746, 0.606079, 0.601490；进程内 median 0.606079
  - cuBLAS：0.269779, 0.266159, 0.272863, 0.269194, 0.275270；进程内 median 0.269779
- 2048×1024×2048：
  - V2.1：0.392159, 0.388868, 0.393454, 0.393885, 0.393810
  - cuBLAS：0.133405, 0.138587, 0.136386, 0.138450, 0.138538

1024³ 中 cuBLAS 第一轮偏慢，2048³ 首次进程中 V2.1 第一轮明显偏慢；额外的 2048³ 完整复测说明结论对该离群点不敏感。最终差距为 2.26–3.82 倍，不属于正常测量波动能够逆转的“小差距”。

`nvidia-smi` 在测试前后空闲状态均报告 41°C，memory clock 14001 MHz；空闲 SM clock 分别为 390/367 MHz。这里没有连续采样负载期间的 boost clock，因此不能把首轮离群值确定归因于温度或频率，只能通过追加完整测量确认它不改变统计结论。

### 5.2 对结果的解释

**[实测]** V2.1 在 1024³ 和 2048³ 分别达到默认 cuBLAS 的 39.32% 和 44.26%；cuBLAS 延迟分别是 V2.1 的 0.3932× 和 0.4426×。所以当前自写 kernel 没有超过 cuBLAS，也不存在 cuBLAS 异常低的问题。

**[已有 profiler 实测]** V2.1 已消除 B 的 LDSM excessive wavefront，但 A0/A1 各仍是 8 actual / 4 ideal；Long Scoreboard 仍为 18.376 warp cycles/issued instruction。1024³ 只有 256 blocks、0.59 waves/SM，且 56 registers/thread 将理论 occupancy 限为 75%。Global→Shared 仍是标量 `LDG.E.U16`/`STS.U16`，每个 K tile 有两次 block barrier。

**[合理推测，不是 cuBLAS 内部实测]** cuBLAS 的优势可能来自更成熟的多级 tiling、流水化、数据搬运、形状专用 kernel 与算法选择。本文没有 profile cuBLAS 内部 kernel，也没有做 cuBLASLt heuristic 搜索，因此不把某一具体内部策略写成已验证事实。

## 6. FP32 CUDA-Core 优化路线

`src/gemm.cu` 中真实存在的实现如下：

| 输出名称 | Kernel | 关键变化 |
|---|---|---|
| Naive | `gemmNaive` | 每线程一个 C 元素，直接读取 global A/B |
| Tiled | `gemmTiled` | 16×16 shared-memory tiling |
| Thread Tiled | `gemmThreadTiled` | 一维 register blocking，每线程计算多个列 |
| 2D Thread Tiled | `gemm2DThreadTiled` | 64×64×16 tile、4×4 thread micro-tile |
| Vectorized | `gemm2DThreadTiledVectorized` | `float4` coalesced/global vector loads |
| V5 | `gemm2DThreadTiledVectorizedSharedStore` | 向 shared 的向量化 store 实验 |
| V6 | `gemm2DThreadTiledVectorizedGlobalStore` | 向 global C 的 `float4` store |
| V7 | `gemm2DThreadTiledDoubleBuffered` | 双缓冲 shared tile，重叠 next-tile load 与当前计算 |
| V7_2 | `gemm2DThreadTiledDoubleBufferedRegisterPrefetch` | 在 V7 上增加 register prefetch |
| V8_1 | `gemmWarpTiledDoubleBufferedV8_1` | 显式 warp tile layout |
| V8_2 | `gemmWarpTiledDoubleBufferedV8_2` | 仅给双缓冲 A shared stride 增加 `+1` padding |

本轮在同一 RTX 5070、`sm_120 -O3` 下运行现有 1024³ 程序得到以下 **FP32 当前快照**。原程序使用 10 次 warmup、100 次 Event 区间循环，但不是本轮 Tensor/cuBLAS 的五轮中位数协议，因此仅用于路线总结。

| 版本 | Latency | FP32 性能 | 相对前一步 |
|---|---:|---:|---:|
| Naive | 1.030340 ms | 2.084 TFLOPS | — |
| Tiled | 0.778897 ms | 2.757 TFLOPS | 1.323× |
| Thread Tiled | 0.670893 ms | 3.201 TFLOPS | 1.161× |
| 2D Thread Tiled | 0.275297 ms | 7.801 TFLOPS | 2.437× |
| Vectorized | 0.166094 ms | 12.929 TFLOPS | 1.657× |
| V5 | 0.157825 ms | 13.607 TFLOPS | 1.052× |
| V6 | 0.157570 ms | 13.629 TFLOPS | 1.002× |
| V7 | 0.149161 ms | 14.397 TFLOPS | 1.056× |
| V7_2 | 0.145540 ms | **14.755 TFLOPS** | 1.025× |
| V8_1 | 0.173926 ms | 12.347 TFLOPS | 0.858×（相对 V7） |
| V8_2 | 0.152652 ms | 14.068 TFLOPS | 1.139×（相对 V8_1） |
| FP32 `cublasSgemm` | 0.109040 ms | 19.694 TFLOPS | V7_2 达成率 74.92% |

从 Naive 到当前最佳 V7_2 为约 7.08×。V8_2 的 padding 修复了 V8_1 的一部分 warp-layout/bank-conflict 代价，但仍没有超过 V7_2，因此不能把“更复杂”当成“更快”。所有 1024³ 主版本最大绝对误差为 `4.76837e-6`；V5–V8 还通过了非整 tile/小 K 控制流测试。

仓库已有 NCU 报告为这条路线提供了局部证据：V6 报告显示 L1/TEX throughput 74.29%、Compute throughput 49.84%；V7 为 88.73%/54.91%，但理论 occupancy 因资源占用降至 50%；V7_2 报告显示 L1/TEX throughput 96.28%、Compute throughput 69.94%，且 MIO throttle 成为主要 stall（5.3 cycles/issued instruction）。这些 profiler duration 受 replay 影响，不与普通 Event latency 混用。

## 7. FP16 Tensor Core 优化路线

下表的性能取自各版本对应报告中的同轮对照；不同报告之间不是单次统一运行。每个 speedup 都只使用该报告内重测的直接基线。

| 版本 | 1024³ | 2048³ | 同轮直接结论 |
|---|---:|---:|---|
| V0 WMMA Baseline | 14.547 TFLOPS | 15.615 TFLOPS | 起点 |
| V1 Shared Tiling | 15.327 TFLOPS | 17.318 TFLOPS | 相对 V0：1.0536× / 1.1090× |
| V1.1 B Padding 40 | 16.742 TFLOPS | 19.263 TFLOPS | 相对 V1：1.0992× / 1.1118× |
| V2 Fragment Reuse | 16.546 TFLOPS | 22.319 TFLOPS | 相对 V1.1：0.9787× / 1.1816× |
| V2.1 B Padding 72 | 20.924 TFLOPS | 28.106 TFLOPS | 相对 V2：1.2628× / 1.2702× |

### V0：Global → WMMA Fragment

- 动机：建立最简单、可验证的 Tensor Core 基线。
- 代码：每 warp 计算一个 16×16 C tile；A/B fragment 直接从 global memory 加载。
- NCU：V0 global-load sector 平均只有 16/32 B 有效，Long Scoreboard 49.98、MIO Throttle 26.19、LG Throttle 19.90。
- 新瓶颈：跨 warp 输入重复加载、sector 利用率低、global dependency stall 高。

### V1：32×32 Shared Tiling

- 动机：让 2×2 warps 复用 block 级 A/B tile并改善合并访问。
- 代码：`As[32][16]`、`Bs[16][32]`，Global→Shared cooperative load，再由 WMMA shared load 取 fragment。
- NCU：global-load sectors 相对 V0 降 4×，每 sector 有效字节从 16/32 提到 32/32；Long Scoreboard 49.98→8.96。
- 实际效果：1024³/2048³ 提升 5.36%/10.90%。
- 新瓶颈：A `LDSM` 为 8 actual/4 ideal，B 为 16 actual/4 ideal；合计约 426 万 shared-load bank conflicts。

### V1.1：B Padding 40

- 动机：隔离验证 V1 的 B 行跨度 32 half 所造成的 bank 周期性。
- 代码：只将 `Bs[16][32]` 改为物理 `Bs[16][40]`，逻辑 tile和 MMA 不变。
- NCU：B `LDSM.16.MT88.4` 从 16 actual/4 ideal 降到 4/4，B excessive wavefronts 清零；总 load conflicts -73.97%。
- 实际效果：1024³/2048³ 相对 V1 提升 9.92%/11.18%。
- 新瓶颈：A 的 8/4 pattern 未变，Long Scoreboard 未下降；每 block 多 256 B shared memory。

### V2：64×64 Block + 双向 Fragment Reuse

- 动机：每 warp 计算 32×32 C，通过 A0/A1 与 B0/B1 组合减少重复 fragment/global staging load。
- 代码：四个 accumulator 对应 C00/C01/C10/C11；每 K tile 四次 WMMA，A、B fragment 都各复用两次。
- NCU：fragment requests、global loads 和 shared stores均减半，总动态指令 -56.65%，HMMA 总数不变；证明 reuse 生效。
- 代价：registers/thread 40→56，理论 occupancy 100%→75%；无 spill。
- 反作用：未 padding 的 `Bs[16][64]` 让两个 B LDSM 均达到 32 actual/4 ideal，load conflicts 约为 V1.1 的 3.79×。
- 实际效果：1024³ 小幅退化 2.13%，2048³ 提升 18.16%；小尺寸还受 256-block grid 欠填充影响。

### V2.1：B Padding 72

- 动机：在不改 V2 reuse/tiling 的前提下，只修复 B stride 64 的 LDSM serialization。
- 代码：物理 stride 64→72，`load_matrix_sync` ldm 同步改为 72；逻辑 B tile仍为 16×64。
- NCU：两个 B LDSM 都从 32 actual/4 ideal 降到 4/4；B excessive 归零。总 load conflicts 4,208,358→530,059（-87.40%），shared-store conflicts 保持 0。
- 资源：仍为 56 registers/thread、0 spill、75% theoretical occupancy；只增加 256 B shared memory。
- stall/利用率：Short Scoreboard -67.12%、MIO Throttle -74.80%，HMMA subpipe active 15.40%→17.89%。
- 实际效果：相对 V2 在 1024³/2048³ 提升 26.28%/27.02%，成为已测试自写 Tensor Core 版本中的最佳实现。

## 8. 当前瓶颈和未解决问题

- V2.1 只支持 M/N 为 64 的倍数、K 为 16 的倍数，没有通用边界处理。
- A0/A1 LDSM 仍各有 8 actual/4 ideal，V2.1 仍约有 2.022 bank conflicts/shared-load request。
- Global→Shared staging 仍为标量 half load/store，没有 async copy 或 double buffering。
- 56 registers/thread 限制理论 occupancy；1024³ 的 grid 也不足以产生完整 waves。
- 两次 `__syncthreads()`/K tile、Long Scoreboard 和固定 64×64 tile 仍限制吞吐。
- 大非方形达成率只有 35.19%，说明单一固定 tile 对形状适应性有限。
- 本轮未 profile cuBLAS 内部 kernel，也未尝试 cuBLASLt；因此只报告 API 设置和端到端 GPU stream 时间，不猜测内部算法。
- 本项目是教学/实验实现，不支持任意 stride、batched GEMM、融合 epilogue 或生产级错误恢复。

## 9. 复现命令

```bash
mkdir -p build
nvcc -O3 -lineinfo -arch=sm_120 -Xptxas=-v \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_final -lcublas

./build/tensor_core_gemm_final 512 512 512
./build/tensor_core_gemm_final 1024 1024 1024
./build/tensor_core_gemm_final 2048 2048 2048
./build/tensor_core_gemm_final 2048 1024 2048
```

每个命令都会运行两套 CPU reference、全部自写 Tensor Core kernel 与 cuBLAS correctness，然后只对 V2.1/cuBLAS 执行五轮交替 benchmark；程序还会自动补跑 `64×96×32` 和 `128×192×64` 映射检查。

详细单阶段证据见：

- [V1 Shared Tiling](V1_PERFORMANCE_REPORT.md)
- [V1.1 Bank Conflict](V1_1_BANK_CONFLICT_REPORT.md)
- [V2 Fragment Reuse](V2_PERFORMANCE_REPORT.md)
- [V2.1 Bank Conflict](V2_1_BANK_CONFLICT_REPORT.md)

## 10. 证据边界

- **硬件/API 事实**：数据类型、cuBLAS 参数、WMMA/LDSM 公开语义及工具版本。
- **代码推导**：Row-major 映射、tile 覆盖、reuse 关系和 shared stride bank 周期。
- **Profiler 实测**：各 Tensor 报告中的 request/wavefront/conflict/stall/occupancy 指标。
- **普通实测**：CUDA Event latency、TFLOPS、CPU reference 误差。
- **推测**：cuBLAS 领先的具体内部原因。未 profile 或未公开的算法细节不作为实测结论。

项目 kernel 的核心学习和实现由项目作者完成；Codex 辅助了 Host 测试集成、Profiler 数据整理和文档撰写。本轮未创建新 kernel、未提交 Git commit。
