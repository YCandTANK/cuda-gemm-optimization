# FP8 Tensor Core GEMM V3 集成、正确性与性能报告

验证日期：2026-11-03（Asia/Shanghai）

## 1. 结论

用户实现的 `fp8_gemm_v3()` 已完成工程集成、映射审计、严格正确性验证、
PTX/SASS 检查、7 轮交替 CUDA Event benchmark、1024³ Nsight Compute 对照，
并独立复核了 V1 历史跨会话计时差异。

- 冻结的 V1/V2 Kernel 未修改。当前 `src/fp8_gemm.cu` 前 288 行与 `fp8-v2`
  tag 完全一致，SHA-256 均为
  `5e600b1d0ab1fc5c424c78f2eb235f266bee2c92ad125d2a530d49d850a39555`。
- V3 的 4 个 warp 在 CTA 内分别覆盖 `(0,0)`、`(0,8)`、`(16,0)`、
  `(16,8)` 四个 16×8 tile，合计无重叠、无遗漏地覆盖 32×16。
- 冻结 V1 8/8、既有 V2 8/8、V2/V3 新增 9/9 正确性案例全部通过。
  V3/reference 和 V3/V2 的全局最大绝对误差、最大相对误差均为 0。
- V3 PTX 保留 `ld.global.b32` 和 FP8 `mma.sync`；SASS 保留 32-bit `LDG.E`
  与 `QMMA.16832.F32.E4M3.E4M3`。V3 使用 48 registers/thread、0 spill、
  0 barrier；V2 为 40 registers/thread。
- 交替 Event 中位数显示 V3/V2：512³ 为 0.940×，1024³ 为 1.023×，2048³
  为 0.989×。不同 shape 和轮次有正有负，V3 没有形成稳定性能提升。
- 1024³ NCU 中，理论/实测 Occupancy 从 50.00%/48.64% 提高到
  83.33%/78.32%，但 Global Load 请求与 sectors 完全不变。LG Throttle 从
  0.55 增至 39.21，Long Scoreboard 从 79.10 增至 92.36
  cycles/issued instruction；Tensor Pipe Active 只从 14.62% 增至 15.42%。
- V1 当前冻结入口、旧直接编译二进制和独立 Event/墙钟审计互相一致，已排除当前
  计时公式、CMake 和二进制版本造成 5～6 倍差异。历史运行时 GPU 状态没有被完整
  保存，因此不能证明唯一根因；现有证据更符合跨会话功耗/时钟、并发负载或 WSL
  驱动状态差异。该问题不影响本轮同进程交替 V2/V3 结论。

本轮没有加入 Shared Memory、寄存器预取、`cp.async`、Double Buffering 或其他
优化，也没有自动创建 Git commit/tag。

## 2. 文件与 Git 状态

### 2.1 本轮文件

| 文件 | 变更 |
|---|---|
| `src/fp8_gemm.cu` | 用户新增 V3；本轮未修改 V1/V2/V3 核心计算代码 |
| `src/fp8_gemm.cuh` | 增加 V3 声明、shape 和 launch contract |
| `tests/fp8_gemm_v3_compare.cu` | 新增 V2/V3 正确性、交替 benchmark、NCU workload 和 V1 计时审计 |
| `CMakeLists.txt` | 增加独立 target `fp8_gemm_v3_compare` |
| `FP8_V3_VALIDATION_AND_PERFORMANCE.md` | 本报告 |

冻结测试文件没有改动：

```text
5285632a81827fbe333263b53219817e723eb84eff8a4a18c33f34e1e511d17d  tests/fp8_gemm_v1_test.cu
df8cd4675dd9167ca5db8c28afc8117c6806d4d319c8944f464002e7637e654a  tests/fp8_gemm_v2_compare.cu
```

### 2.2 最终 Git 状态

```text
## main...origin/main
 M CMakeLists.txt
 M README.md
 M src/fp8_gemm.cu
 M src/fp8_gemm.cuh
?? FP8_V3_VALIDATION_AND_PERFORMANCE.md
?? tests/fp8_gemm_v3_compare.cu
```

此外仍有多份进入本轮前就存在的未跟踪 FP32/FP16/FP8 报告，未删除或覆盖。当前
HEAD 为：

```text
821d5f7 (HEAD -> main, tag: fp8-v2, origin/main)
feat: optimize FP8 GEMM with 32-bit fragment loads
```

原始 NCU report 位于被 Git 忽略的 `build/ncu/`：

```text
build/ncu/fp8_v2_1024_v3_compare_20261103.ncu-rep  326623 bytes
build/ncu/fp8_v3_1024_compare_20261103.ncu-rep     324535 bytes
```

如需长期保存 profiler 原始数据，需要单独归档。

## 3. 环境与构建

| 项目 | 结果 |
|---|---|
| OS | WSL Ubuntu |
| GPU | NVIDIA GeForce RTX 5070，CC 12.0 / SM120 |
| GPU memory | 12,792,365,056 bytes |
| Driver | 621.01 |
| CUDA Toolkit | 13.2，nvcc 13.2.86 |
| PTX ISA | 9.2 |
| Nsight Compute | 2026.1.1.0 |
| 编译目标 | `sm_120` |

构建命令：

```bash
cmake -S . -B build/cmake-sm120 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-sm120 --target \
  fp8_gemm_v1_test fp8_gemm_v2_compare fp8_gemm_v3_compare -j 4
```

构建成功。ptxas 结果：

| Kernel | Registers/thread | Stack | Spill stores/loads | Barriers |
|---|---:|---:|---:|---:|
| V1 | 39 | 0 | 0 / 0 | 0 |
| V2 | 40 | 0 | 0 / 0 | 0 |
| V3 | 48 | 0 | 0 / 0 | 0 |

V3 相对 V2 增加 8 registers/thread，是本次 4-warp CTA 地址组织的主要资源成本，
但没有造成 register spill。

## 4. Warp/fragment/写回映射审计

### 4.1 CTA 与 Warp tile

V3 launch contract：

```text
block = (128, 1, 1)
grid  = (N / 16, M / 32, 1)
M % 32 == 0, N % 16 == 0, K % 32 == 0
```

线程映射：

```text
warp_id = threadIdx.x / 32        // 0..3
lane    = threadIdx.x % 32        // 0..31
warp_m  = warp_id / 2             // 0,0,1,1
warp_n  = warp_id % 2             // 0,1,0,1
m0 = blockIdx.y * 32 + warp_m * 16
n0 = blockIdx.x * 16 + warp_n * 8
```

| warp_id | warp_m | warp_n | CTA 内 C tile |
|---:|---:|---:|---|
| 0 | 0 | 0 | rows 0..15, cols 0..7 |
| 1 | 0 | 1 | rows 0..15, cols 8..15 |
| 2 | 1 | 0 | rows 16..31, cols 0..7 |
| 3 | 1 | 1 | rows 16..31, cols 8..15 |

四个集合两两不交，并集恰好是 CTA 的 32×16 output tile。相邻 CTA 的行/列起点
分别相差 32/16，因此 CTA 之间也没有重叠或空洞。

### 4.2 Warp 内 C fragment

每个 warp 内仍使用已验证的 V2 映射：

```text
group = lane / 4                 // 0..7
tid   = lane % 4                 // 0..3
row   = group + 8 * (i / 2)      // 0..15
col   = 2 * tid + (i % 2)        // 0..7
```

`group` 唯一确定 8 行组，`i/2` 选择上下 8 行，`tid` 和 `i%2` 唯一确定 8 列；
32 lanes × 4 accumulators 恰好生成 128 个不同坐标，即完整 16×8。

### 4.3 A/B fragment

A/B fragment 的 lane/register 公式与 V2 完全相同，只把 warp tile 的 `m0/n0`
加入 global index：

```text
A: row = group + 8*(r%2), col = 4*tid + 16*(r/2)
B: row = 4*tid + 16*r,    col = group
```

每个寄存器从连续 4 个 E4M3 元素执行一次 `ld.global.b32`。V3 没有 warp 间共享：
横向两个 warp 会独立读取逻辑相同的 A 数据，纵向两个 warp 会独立读取逻辑相同的
B 数据。这对正确性没有影响，但决定了第 8 节中的性能结论。

## 5. 正确性验证

### 5.1 Reference 与检查

Host 先构造 FP32 输入，再转换成 `__nv_fp8_e4m3`，并把实际量化结果转回 FP32。
Reference 按 A row-major、B column-major 执行相同顺序的 FP32 `std::fma`：

```text
C[m,n] = sum_k float(A_e4m3[m,k]) * float(B_e4m3[n,k])
```

判据沿用 V2：

```text
abs_error <= 1e-4 + 1e-4 * abs(reference)
```

每次 launch 检查 CUDA launch/runtime 错误；C 预填 NaN，前后各放 64 个 float
guard。V3 失败路径会报告 C 坐标、reference/V2/V3、warp、lane 和 accumulator。

### 5.2 冻结回归

```bash
./build/cmake-sm120/fp8_gemm_v1_test --correctness
./build/cmake-sm120/fp8_gemm_v2_compare --correctness
```

结果：冻结 V1 8/8 PASS；既有 V2 8/8 PASS。所有误差为 0，guard OK。

### 5.3 V2/V3 同输入结果

```bash
./build/cmake-sm120/fp8_gemm_v3_compare --correctness
```

| Case | Shape | 输入 | V2/ref | V3/ref | V3/V2 | 结果 |
|---|---:|---|---:|---:|---:|---|
| zero_single_cta | 32×16×32 | 全零 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| sparse_single_cta | 32×16×32 | 双点稀疏、正负 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| distinct_single_cta | 32×16×32 | 非均匀位置函数 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| mixed_three_k_tiles | 32×16×96 | 正负混合、3 K tiles | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_multi_cta_a | 64×32×64 | seed 8301 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| rectangular_nonuniform | 96×48×160 | 非方阵、非均匀 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_multi_cta_b | 64×64×128 | seed 8501 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_128 | 128×128×256 | seed 8601 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| random_256 | 256×256×512 | seed 8701 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |

表中每个误差单元为 `max_abs / max_rel`。全部案例同时满足：

```text
over_tol=0, nonfinite=0, V2 guard=OK, V3 guard=OK
```

结果是逐元素 FP32 完全一致，不依赖宽松容差判定。

## 6. PTX/SASS 与指令开销

### 6.1 指令确认

```bash
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_v3_probe/fp8_gemm.ptx
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v3_compare
```

V3 PTX：

```text
ld.global.b32 ...                  # 每 K tile: A × 4, B × 2
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ...
```

V3 SASS：

```text
LDG.E ...
QMMA.16832.F32.E4M3.E4M3 ...
```

没有 `.U8` load、packing 序列、`BAR`、`WARPSYNC` 或 spill。编译器与 V2 一样
把 steady-state K loop 展开 4 倍，每 K tile 平均 6 条 `LDG.E`、1 条 QMMA。

### 6.2 1024³ 动态指令

V2/V3 都处理 262,144 个 warp-level K tiles，因此核心指令数相同：

| 动态指令 | V2 | V3 | 变化 |
|---|---:|---:|---:|
| `LDG.E` | 1,572,864 | 1,572,864 | 0 |
| `QMMA.16832` | 262,144 | 262,144 | 0 |
| `STG.E` | 32,768 | 32,768 | 0 |
| `IADD.64` | 393,216 | 393,216 | 0 |
| `IADD` | 303,104 | 319,488 | +16,384 |
| `IMAD` | 57,344 | 65,536 | +8,192 |
| `LOP3` | 24,576 | 49,152 | +24,576 |
| `S2R` | 16,384 | 24,576 | +8,192 |
| `MOV` | 352,256 | 221,184 | -131,072 |
| 全部 warp instructions | 3,637,248 | 3,538,944 | -98,304（-2.70%） |

V3 的 warp/tile 分解确实增加了少量 `IADD/IMAD/LOP3/S2R`，但 grid block 数从
8192 降到 2048，CTA prologue/epilogue 被摊薄，因此总动态指令反而下降 2.70%。
新增开销主要表现为 40→48 registers/thread，而不是总指令数或同步指令增加。

## 7. V2/V3 交替 CUDA Event Benchmark

命令：

```bash
./build/cmake-sm120/fp8_gemm_v3_compare --benchmark
```

每个 shape 7 轮，奇数轮 V2→V3、偶数轮 V3→V2；每个版本每轮独立执行相同预热
和计时迭代。输入、设备 buffer、layout 和运算量相同；TFLOPS 按
`2*M*N*K/time` 计算。以下全部是 CUDA Event，不是 profiler Duration。

### 7.1 512³（10 warmups / 100 iterations）

| 轮次 | 顺序 | V2 ms | V2 TFLOPS | V3 ms | V3 TFLOPS | V3/V2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V2,V3 | 0.020664 | 12.990689 | 0.019484 | 13.777565 | 1.060572× |
| 2 | V3,V2 | 0.019849 | 13.524124 | 0.019666 | 13.649779 | 1.009291× |
| 3 | V2,V3 | 0.017162 | 15.641633 | 0.019796 | 13.559975 | 0.866916× |
| 4 | V3,V2 | 0.017132 | 15.668220 | 0.019631 | 13.674032 | 0.872724× |
| 5 | V2,V3 | 0.018610 | 14.424322 | 0.019969 | 13.442421 | 0.931927× |
| 6 | V3,V2 | 0.016380 | 16.387519 | 0.022628 | 11.863062 | 0.723908× |
| 7 | V2,V3 | 0.020067 | 13.376827 | 0.020018 | 13.409544 | 1.002446× |
| **中位数** | — | **0.018610** | **14.424322** | **0.019796** | **13.559975** | **0.940077×** |

512³ 时间很短，轮间波动明显；中位数上 V3 慢约 6.0%。

### 7.2 1024³（10 warmups / 50 iterations）

| 轮次 | 顺序 | V2 ms | V2 TFLOPS | V3 ms | V3 TFLOPS | V3/V2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V2,V3 | 0.116004 | 18.512076 | 0.113383 | 18.940080 | 1.023120× |
| 2 | V3,V2 | 0.110274 | 19.474083 | 0.113736 | 18.881247 | 0.969558× |
| 3 | V2,V3 | 0.114374 | 18.776017 | 0.110676 | 19.403252 | 1.033406× |
| 4 | V3,V2 | 0.110732 | 19.393494 | 0.114216 | 18.802004 | 0.969501× |
| 5 | V2,V3 | 0.114330 | 18.783163 | 0.110700 | 19.399102 | 1.032792× |
| 6 | V3,V2 | 0.114325 | 18.784005 | 0.108328 | 19.823843 | 1.055358× |
| 7 | V2,V3 | 0.109926 | 19.535764 | 0.111780 | 19.211616 | 0.983407× |
| **中位数** | — | **0.114325** | **18.784005** | **0.111780** | **19.211616** | **1.022765×** |

1024³ 中位数提高约 2.3%，但 3/7 轮 V3 更慢，幅度不足以称为稳定提升。

### 7.3 2048³（5 warmups / 20 iterations）

| 轮次 | 顺序 | V2 ms | V2 TFLOPS | V3 ms | V3 TFLOPS | V3/V2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V2,V3 | 0.864587 | 19.870603 | 0.885981 | 19.390792 | 0.975853× |
| 2 | V3,V2 | 0.857448 | 20.036048 | 0.874086 | 19.654658 | 0.980965× |
| 3 | V2,V3 | 0.835933 | 20.551734 | 0.845563 | 20.317663 | 0.988611× |
| 4 | V3,V2 | 0.853770 | 20.122372 | 0.879213 | 19.540059 | 0.971061× |
| 5 | V2,V3 | 0.869053 | 19.768499 | 0.863070 | 19.905525 | 1.006932× |
| 6 | V3,V2 | 0.843976 | 20.355874 | 0.863003 | 19.907076 | 0.977952× |
| 7 | V2,V3 | 0.846462 | 20.296080 | 0.840192 | 20.447551 | 1.007463× |
| **中位数** | — | **0.853770** | **20.122372** | **0.863070** | **19.905525** | **0.989224×** |

2048³ 中位数上 V3 慢约 1.1%。三种 shape 合并判断：4-warp CTA 本身没有带来
稳定性能收益。

## 8. 1024³ Nsight Compute 对照

### 8.1 采集方法

V2/V3 各自发射 10 次相同输入的 1024³ workload，跳过 5 次、采 1 次；使用相同
10 个 section 和 `--cache-control none`，各完成 35 replay passes：

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
  --kernel-name-base function --kernel-name fp8_gemm_v2 \
  --launch-skip 5 --launch-count 1 --kill yes \
  --cache-control none --force-overwrite \
  -o build/ncu/fp8_v2_1024_v3_compare_20261103 \
  ./build/cmake-sm120/fp8_gemm_v3_compare --profile-v2

# V3 将 kernel/output/mode 改为：
# fp8_gemm_v3 / fp8_v3_1024_compare_20261103 / --profile-v3
```

NCU 明确警告 cache 不受控制。V3 replay-derived L2 hit rate 为 100.67%，无物理
意义，因此不用于结论；NCU Duration 也不替代 Event benchmark。

### 8.2 Launch、Occupancy 与 Scheduler

| 指标 | V2 | V3 |
|---|---:|---:|
| Grid | 128×64 = 8192 CTAs | 64×32 = 2048 CTAs |
| Block | 32 threads / 1 warp | 128 threads / 4 warps |
| Total threads / warps | 262,144 / 8,192 | 262,144 / 8,192 |
| Registers/thread | 40 | 48 |
| Register block limit / SM | 48 | 10 |
| Theoretical active warps / SM | 24 | 40 |
| Theoretical Occupancy | 50.00% | 83.33% |
| Achieved active warps / SM | 23.35 | 37.59 |
| Achieved Occupancy | 48.64% | 78.32% |
| Active warps / scheduler | 5.85 | 9.39 |
| Eligible warps / scheduler | 0.07 | 0.16 |
| One or more eligible | 6.98% | 6.80% |
| No eligible | 93.02% | 93.20% |

V3 成功突破 V2 因 SM block-count 上限造成的 50% theoretical occupancy。V3 被
48 registers/thread 限制为 10 blocks/SM，即 40 warps/SM；尽管不是满 48 warps，
仍显著高于 V2 的 24 warps。

但 eligible warps 的绝对数虽然从 0.07 增至 0.16，scheduler 至少有一个 eligible
warp 的周期比例反而从 6.98% 略降到 6.80%。这说明新增 resident warps 大部分也在
等待相同的数据依赖，并没有转化为可持续 issue。

### 8.3 Memory、stall 与 Tensor pipe

| 指标 | V2 | V3 | 变化 |
|---|---:|---:|---:|
| Global Load requests/instructions | 1,572,864 | 1,572,864 | 0 |
| Global Load sectors | 12,582,912 | 12,582,912 | 0 |
| Sectors / load request | 8 | 8 | 0 |
| Useful bytes / 32-byte sector | 16 | 16 | 0 |
| L1/TEX hit rate | 75.68% | 81.43% | +5.75 pp，cache replay 需谨慎 |
| L1/TEX throughput / active | 98.69% | 99.03% | +0.34 pp |
| Memory throughput / elapsed | 88.84% | 94.25% | +5.41 pp |
| DRAM throughput | 6.47% | 6.47% | 不变 |
| DRAM bytes/s | 43.41 GB/s | 43.44 GB/s | 基本不变 |
| LG Throttle | 0.55 | 39.21 | +38.66 cycles/issue |
| Long Scoreboard | 79.10 | 92.36 | +13.26 cycles/issue |
| Warp cycles / issued instruction | 83.75 | 138.19 | +65.0% |
| Tensor Pipe Active / elapsed | 14.62% | 15.42% | +0.80 pp |
| NCU Duration | 119.52 us | 112.38 us | profiler-only |

Global request 和 sector 总数完全相同，证明 2×2 warp grouping 没有自动产生 A/B
数据复用。V3 更高 Occupancy 同时提高了 L1/TEX 上的并发请求压力：L1 active
throughput 已接近 99%，LG Throttle 大幅增加，Long Scoreboard 也更高。DRAM 只有
6.47%，因此不能把等待解释为 DRAM 带宽饱和。

V3 的 Tensor Pipe Active 只提高 0.80 个百分点，远小于 Occupancy 增幅。更多
resident warps 主要堆积在 L1TEX 请求队列和 load dependency 上，而不是持续向
Tensor Core 供数。这解释了为何 Event 性能只在 1024³ 中位数上略升、其余 shape
持平或下降。

另一个次要因素是 1024³ 的 V3 grid 只有 4.27 waves/SM；NCU 报告最后一个 partial
wave 可能带来最高约 20% tail effect。V2 有 7.11 waves/SM。该估计不是实测加速值，
但说明更大 CTA 会降低 block scheduling 粒度，可能抵消部分 Occupancy 收益。

## 9. V1 跨会话计时差异核查

### 9.1 计时逻辑审计

冻结 V1、V2 harness 和本轮 V3 harness 的 Event 路径完全一致：

1. 执行指定 warmup；
2. `cudaGetLastError()` + `cudaDeviceSynchronize()`；
3. 记录 start event；
4. 连续发射 `iterations` 次 Kernel；
5. 记录并同步 stop event；
6. `cudaEventElapsedTime / iterations`。

三者也都使用 512³: 10/100、1024³: 10/50、2048³: 5/20 的
warmup/iteration 协议。没有发现少除一次迭代、把总时间当平均值或把 profiler
replay 时间混入 Event 的逻辑错误。

### 9.2 当前独立实测

冻结 V1 原测试入口：

| Shape | 当前 CMake binary | 历史冻结值 | 历史/当前 |
|---:|---:|---:|---:|
| 512³ | 0.051440 ms | 0.243956 ms | 4.743× |
| 1024³ | 0.399734 ms | 2.296136 ms | 5.744× |
| 2048³ | 3.080645 ms | 18.061241 ms | 5.863× |

保留于 `build/fp8_gemm_v1_test`、在 11:22 直接编译的旧二进制当前结果：

```text
512³   0.051608 ms
1024³  0.395853 ms
2048³  3.068517 ms
```

旧 binary 与当前 CMake binary 差异只有约 -0.3%、+1.0%、+0.4%，排除了编译入口
导致 5～6 倍差异。

独立 1024³ 审计在相同 50 次 launch 区间同时测量 Event 和 host steady clock：

| Repeat | Event ms/launch | Wall ms/launch |
|---:|---:|---:|
| 1 | 0.391997 | 0.397903 |
| 2 | 0.408470 | 0.409362 |
| 3 | 0.391117 | 0.392066 |
| 4 | 0.395988 | 0.396559 |
| 5 | 0.391603 | 0.392954 |
| **中位数** | **0.391997** | **0.396559** |

Event 与墙钟中位数只差约 1.16%，当前计时可信。

### 9.3 能确认与不能确认的结论

实测能确认：当前计时公式正确；当前旧/新 binary 一致；当前同会话 Event 和墙钟
一致。当前 `nvidia-smi` 可见 P0，但空闲 SM clock 采样在 285–1455 MHz 间变化；
NCU workload 中实际 SM frequency 约为 2.53–2.54 GHz。

历史冻结运行没有保存每轮 active clock、power、温度、并发 GPU process 和完整
WSL/driver 状态，因此不能事后证明唯一根因。结合相同源码/旧 binary 当前都恢复到
约 0.4 ms，更合理的推断是历史会话存在 GPU power/clock 状态、并发负载或 WSL
runtime 状态差异；这是推断，不是已证明事实。

本轮 V2/V3 在同一进程、同一 buffer、交替顺序下测量，因此该跨会话问题不会污染
V3/V2 比值。V2 当前绝对时间也不应用来覆盖此前报告中的冻结历史数据。

## 10. 实测与推断边界

### 已实测

- V1/V2 源码冻结哈希；V3 映射覆盖与 9 组逐元素结果。
- PTX/SASS 指令、register/spill/barrier 和动态 opcode 数。
- 7 轮交替 CUDA Event 的全部延迟与 TFLOPS。
- 两份 35-pass NCU 中的 Occupancy、scheduler、stall、load 和 sector 指标。
- 当前 V1 原入口、旧 binary、Event/墙钟一致性。

### 推断

- V3 无稳定提升的主因是新增 resident warps 没有共享数据，反而加重 L1TEX 请求
  队列与 load dependency；该推断由不变的 load/sectors、显著增加的 LG/Long
  Scoreboard 和很小的 Tensor Active 增幅共同支持。
- V3 partial-wave tail 和 8 个额外 registers 是次要不利因素；尚未通过独立消融
  实验量化各自贡献。
- V1 历史差异可能来自跨会话 GPU/WSL 状态；因历史 telemetry 缺失，不能声称已
  精确定位。

## 11. 当前 V3 局限

- 仅支持紧密 A row-major、B column-major、C row-major。
- M/N/K 必须分别为 32/16/32 的倍数，无边界 tile。
- 输入需预先量化为 E4M3，无 scaling、batch、任意 leading dimension 或 epilogue。
- 4 个 warp 只在 CTA 组织上聚合，没有 shared-memory 或 warp 间 A/B 广播，因此
  没有减少 global load requests/sectors。
- 更大 CTA 提高 Occupancy，但降低 block 调度粒度，并增加 registers/thread。
- `--cache-control none` 的 NCU replay 无法提供可信的 V3 L2 hit-rate 百分比。

下一阶段若要验证 4-warp CTA 的真正价值，需要显式复用 A/B 数据；这通常涉及
Shared Memory 或其他 staging。但这超出本轮范围，本次没有实施。

## 12. 可复现命令

```bash
# 构建
cmake --build build/cmake-sm120 --target \
  fp8_gemm_v1_test fp8_gemm_v2_compare fp8_gemm_v3_compare -j 4

# 正确性
./build/cmake-sm120/fp8_gemm_v1_test --correctness
./build/cmake-sm120/fp8_gemm_v2_compare --correctness
./build/cmake-sm120/fp8_gemm_v3_compare --correctness

# V2/V3 七轮交替 Event
./build/cmake-sm120/fp8_gemm_v3_compare --benchmark

# V1 计时审计
./build/cmake-sm120/fp8_gemm_v3_compare --v1-timing-audit
./build/cmake-sm120/fp8_gemm_v1_test --benchmark
./build/fp8_gemm_v1_test --benchmark

# PTX / SASS
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_v3_probe/fp8_gemm.ptx
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v3_compare

# NCU report
ncu --import build/ncu/fp8_v2_1024_v3_compare_20261103.ncu-rep
ncu --import build/ncu/fp8_v3_1024_compare_20261103.ncu-rep
```
