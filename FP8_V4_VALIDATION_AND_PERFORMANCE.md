# FP8 Tensor Core GEMM V4 集成、正确性与性能报告

验证日期：2026-11-03（Asia/Shanghai）

## 1. 结论

用户实现的 `fp8_gemm_v4()` 已完成工程集成、Shared Memory/同步审计、严格正确性
验证、PTX/SASS 检查、7 轮交替 CUDA Event benchmark 和 1024³ Nsight Compute
对照。

- V1/V2/V3 的 canonical Kernel 均未修改。当前 `src/fp8_gemm.cu` 前 436 行与
  `fp8-v3` tag 前 436 行完全一致，SHA-256 均为
  `840b325058c319b83b8080abb0acca50dff95227b3fc26c341afa38841bd4169`。
- V4 的 1024-byte A tile 与 512-byte B tile 合计 1536 bytes。Global→Shared
  写入坐标一一唯一，没有写写竞争；两次 CTA barrier 分别保护“写后读”和下一
  K tile 的“读后写”依赖。
- Global 32-bit load 地址均满足 4-byte 对齐。A/B Shared 坐标完整覆盖并正确恢复
  V3 fragment 映射。
- V3 基线与 V3/V4 新增 9 组正确性案例全部通过。V4/reference、V4/V3 的全局
  最大绝对误差和最大相对误差均为 0，所有输出被覆盖且 guard 完整。
- PTX/SASS 确认实际生成 `ld.global.b32`/`LDG.E`、`st.shared.b32`/`STS`、
  `ld.shared.b32`/`LDS`、`bar.sync`/`BAR.SYNC.DEFER_BLOCKING` 和
  `QMMA.16832.F32.E4M3.E4M3`。
- V4 使用 44 registers/thread、1536 bytes static shared memory、1 个 barrier
  slot、0 spill；V3 为 48 registers/thread、0 shared memory、0 barrier。
- V4 在全部 21 个 Event 轮次都快于 V3。512³、1024³、2048³ 的中位数加速
  分别为 1.531×、1.443×、1.453×。
- 1024³ NCU 中，Global Load requests 减少 50%，Global Load sectors 减少 75%；
  LG Throttle 从 39.58 降至 0.30，Long Scoreboard 从 91.72 降至 15.51，Tensor
  Pipe Active 从 15.27% 升至 23.00%。
- V4 的代价是 1,572,864 次 Shared Load 平均发生 2-way bank conflict；MIO
  Throttle 增至 22.13，Barrier Stall 增至 9.04 cycles/issued instruction。
  V4 已显著提速，但瓶颈已从 global-load latency 转移到 Shared/MIO 与同步。

本轮没有加入 Padding、Swizzling、`cp.async`、Double Buffering 或其他额外优化，
没有修改 V4 核心实现，也没有创建 Git commit/tag。

## 2. 文件与 Git 状态

### 2.1 本轮新增/修改

| 文件 | 变更 |
|---|---|
| `src/fp8_gemm.cu` | 用户新增 V4；本轮未改 Kernel 核心 |
| `src/fp8_gemm.cuh` | 增加 V4 声明、shape/launch/shared-memory contract |
| `tests/fp8_gemm_v4_compare.cu` | 新增 V3/V4 正确性、交替 Event 和 NCU workload |
| `CMakeLists.txt` | 增加独立 target `fp8_gemm_v4_compare` |
| `FP8_V4_VALIDATION_AND_PERFORMANCE.md` | 本报告 |

冻结测试文件哈希保持不变：

```text
5285632a81827fbe333263b53219817e723eb84eff8a4a18c33f34e1e511d17d  tests/fp8_gemm_v1_test.cu
df8cd4675dd9167ca5db8c28afc8117c6806d4d319c8944f464002e7637e654a  tests/fp8_gemm_v2_compare.cu
cb491b35a6eac0c587507f9bbf7467a78b76bd39e2bb0aba4666c2c28764cbbc  tests/fp8_gemm_v3_compare.cu
```

### 2.2 Git 说明

当前 HEAD：

```text
2cceb8b (HEAD -> main, tag: fp8-v3, origin/main)
feat: add 4-warp CTA FP8 GEMM V3
```

与本轮直接相关的工作树状态：

```text
 M CMakeLists.txt
 M src/fp8_gemm.cu
 M src/fp8_gemm.cuh
?? FP8_V4_VALIDATION_AND_PERFORMANCE.md
?? tests/fp8_gemm_v4_compare.cu
```

`README.md` 和多份未跟踪实验报告是进入本轮前已有的用户工作，未覆盖或删除。

需要说明，`fp8-v3` tag 的 `src/fp8_gemm.cu` 在 canonical V3 后还包含第二份重复的
`fp8_gemm_v3` 定义；用户当前 V4 正好替换了这份冗余副本。前 436 行中实际使用的
V1/V2/V3 canonical 实现逐字节未变，本轮没有恢复或编辑重复副本。

原始 NCU report 位于被 Git 忽略的 `build/ncu/`：

```text
build/ncu/fp8_v3_1024_v4_compare_20261103.ncu-rep  347101 bytes
build/ncu/fp8_v4_1024_compare_20261103.ncu-rep     377131 bytes
```

## 3. 环境与构建

| 项目 | 结果 |
|---|---|
| 环境 | WSL Ubuntu |
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
  fp8_gemm_v1_test fp8_gemm_v2_compare \
  fp8_gemm_v3_compare fp8_gemm_v4_compare -j 4
```

构建成功。ptxas 资源结果：

| Kernel | Registers/thread | Static smem | Barrier slots | Spill |
|---|---:|---:|---:|---:|
| V1 | 39 | 0 | 0 | 0 |
| V2 | 40 | 0 | 0 | 0 |
| V3 | 48 | 0 | 0 | 0 |
| V4 | 44 | 1536 B | 1 | 0 |

源码有两处 `__syncthreads()`，ptxas 的 `used 1 barriers` 表示二者复用 barrier 0，
不是其中一处被删除。SASS 中每个 K tile 可见两处 `BAR.SYNC`。

## 4. Shared Memory、竞争、同步与对齐审计

### 4.1 Shared 布局

```text
As_words[32][8] : 32 × 8 × 4 B = 1024 B
Bs_words[16][8] : 16 × 8 × 4 B =  512 B
合计                                      1536 B
```

A tile 表示 32×32 个 FP8；B tile 表示 column-major 的 32×16 个 FP8。第二维
都是 8 个 word，即每行/每列 K tile 的 32 个 FP8。

### 4.2 Global→Shared 唯一写入

A：

```text
word = threadIdx.x + i * 128, i=0,1
word 范围 0..255，且每个值只由一个 thread/i 组合产生
row = word / 8       -> 0..31
word_k = word % 8    -> 0..7
```

因此 256 个 A words 恰好各写一次 `As_words[row][word_k]`。

B：

```text
b_word = threadIdx.x -> 0..127
b_col = b_word / 8   -> 0..15
b_word_k = b_word%8  -> 0..7
```

因此 128 个 B words 也恰好各写一次。Global→Shared 阶段没有两个线程写同一
Shared word，不存在写写竞争。

### 4.3 Shared→fragment 坐标

A fragment：

```text
shared_row = warp_m * 16 + row   -> 0..31
shared_word_k = col / 4          -> 0..7
```

横向两个 warp 读取同一 16×32 A subtile，实现 A 复用。

B fragment：

```text
shared_col = warp_n * 8 + col    -> 0..15
shared_word_k = row / 4          -> 0..7
```

纵向两个 warp 读取同一 32×8 B subtile，实现 B 复用。读出的 word 内 4 个 FP8
顺序与 V3 `ld.global.b32` fragment 完全相同。

### 4.4 两次 barrier

1. 第一处 barrier 位于全部 `STS` 之后、任何 `LDS` 之前，保证所有 warp 看到完整
   A/B tile，解决 Shared 写后读依赖。
2. 第二处 barrier 位于当前 tile fragment 读取完成后、下一轮 Shared 覆盖之前，
   防止较快 warp 的下一轮 `STS` 覆盖较慢 warp 尚未完成的 `LDS`，解决读后写依赖。

`K` 和循环条件对 CTA 全部线程相同，两处 barrier 均为无条件 CTA 收敛点，不存在
部分线程跳过 barrier 的死锁风险。

PTX 保持源码顺序。SASS 在部分 unrolled 路径中把第二处 barrier 调度到 QMMA 之前，
但它仍严格位于当前 `LDS` 之后和下一轮 `STS` 之前；QMMA 只使用已进入寄存器的
fragment，不再读取 Shared，因此该重排不破坏依赖。

### 4.5 32-bit Global Load 对齐

CUDA allocation 基址满足高于 4-byte 的对齐。所有加载元素偏移均为：

```text
A: global_row*K + k0 + word_k*4
B: global_col*K + k0 + word_k*4
```

`K`、`k0` 是 32 的倍数，`word_k*4` 是 4 的倍数，E4M3 元素为 1 byte，因此每个
`ld.global.b32` 地址都是 4-byte aligned。

## 5. 正确性验证

测试沿用 V3 完全相同的输入公式、seed、E4M3 量化、A row-major、B column-major、
C row-major 和 sequential FP32 `std::fma` reference。容差仍为：

```text
abs_error <= 1e-4 + 1e-4 * abs(reference)
```

每次 launch 后检查 CUDA launch/runtime error；C 预填 NaN，前后各放 64 个
float guard。

先运行 V3 基线：

```bash
./build/cmake-sm120/fp8_gemm_v3_compare --correctness
```

结果：9/9 PASS，V2/V3/reference 全部误差为 0。

V3/V4 对照：

```bash
./build/cmake-sm120/fp8_gemm_v4_compare --correctness
```

| Case | Shape | 输入 | V3/ref | V4/ref | V4/V3 | 结果 |
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

表中误差为 `max_abs / max_rel`。全部案例同时满足 `over_tol=0`、
`nonfinite=0`、V3/V4 guard OK，结果逐元素完全一致。

## 6. PTX/SASS 与每 K tile 指令

### 6.1 指令确认

```bash
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_v4_probe/fp8_gemm.ptx
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v4_compare
```

关键 PTX/SASS 对应：

| 功能 | PTX | SASS |
|---|---|---|
| 32-bit Global Load | `ld.global.b32` | `LDG.E` |
| Shared Store | `st.shared.b32` | `STS` |
| Shared Load | `ld.shared.b32` | `LDS` |
| CTA barrier | `bar.sync 0` | `BAR.SYNC.DEFER_BLOCKING 0x0` |
| FP8 Tensor Core | `mma.sync...e4m3.e4m3` | `QMMA.16832.F32.E4M3.E4M3` |

### 6.2 每 CTA K tile

1024³ 有 2048 CTAs，每 CTA 有 32 个 K tiles，共 65,536 个 CTA-level K tiles。
NCU 动态计数与源码/SASS 静态计数一致：

| 每 CTA K tile | V3 | V4 | 变化 |
|---|---:|---:|---:|
| Global Load warp instructions | 24 | 12 | -50% |
| Global Load sectors | 192 | 48 | -75% |
| Shared Store warp instructions | 0 | 12 | +12 |
| Shared Load warp instructions | 0 | 24 | +24 |
| Source CTA barrier points | 0 | 2 | +2 |
| Dynamic warp-level `BAR.SYNC` | 0 | 8 | +8（2×4 warps） |
| QMMA | 4 | 4 | 不变 |

V3 每个 warp 独立发出 6 次 Global Load，4 warps 共 24 次。V4 每个 thread 读取
2 A words + 1 B word，即每 warp 3 次、4 warps 共 12 次，实际 global bytes 从
3072 B 降为唯一 A/B tile 的 1536 B。

V3 每条 warp load 平均产生 8 sectors、每个 sector 只使用 16/32 bytes；V4 的
cooperative word 编号使每条 load 只产生 4 sectors，并使用完整 32/32 bytes。
因此 requests 减半的同时，sectors 下降到四分之一。

### 6.3 1024³ 动态总量

| 动态指令 | V3 | V4 | 变化 |
|---|---:|---:|---:|
| Global `LDG.E` | 1,572,864 | 786,432 | -50% |
| Shared `STS` | 0 | 786,432 | 新增 |
| Shared `LDS` | 0 | 1,572,864 | 新增 |
| `BAR.SYNC` | 0 | 524,288 | 新增 |
| QMMA | 262,144 | 262,144 | 不变 |
| 全部 warp instructions | 3,538,944 | 6,078,464 | +71.76% |

V4 总指令增加 71.8%，但减少了高延迟、低效率的 Global Load 与 sectors；新增的大量
Shared/barrier 指令代价低于被移除的 global-load 等待，因此实际性能仍提升。

## 7. V3/V4 交替 CUDA Event Benchmark

```bash
./build/cmake-sm120/fp8_gemm_v4_compare --benchmark
```

每个 shape 7 轮，奇数轮 V3→V4、偶数轮 V4→V3；每个版本每轮独立执行相同预热
和计时迭代。输入、buffer、layout、grid/block 和运算量相同。以下均为 CUDA Event。

### 7.1 512³（10 warmups / 100 iterations）

| 轮次 | 顺序 | V3 ms | V3 TFLOPS | V4 ms | V4 TFLOPS | V4/V3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V3,V4 | 0.020000 | 13.421773 | 0.012736 | 21.076904 | 1.570352× |
| 2 | V4,V3 | 0.019572 | 13.715393 | 0.014517 | 18.491366 | 1.348220× |
| 3 | V3,V4 | 0.019353 | 13.870742 | 0.013305 | 20.176079 | 1.454578× |
| 4 | V4,V3 | 0.023057 | 11.642113 | 0.013066 | 20.544200 | 1.764645× |
| 5 | V3,V4 | 0.019646 | 13.663340 | 0.012994 | 20.658033 | 1.511931× |
| 6 | V4,V3 | 0.020228 | 13.270175 | 0.013165 | 20.389402 | 1.536483× |
| 7 | V3,V4 | 0.020204 | 13.286148 | 0.012853 | 20.884328 | 1.571887× |
| **中位数** | — | **0.020000** | **13.421773** | **0.013066** | **20.544200** | **1.530662×** |

### 7.2 1024³（10 warmups / 50 iterations）

| 轮次 | 顺序 | V3 ms | V3 TFLOPS | V4 ms | V4 TFLOPS | V4/V3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V3,V4 | 0.101266 | 21.206382 | 0.080454 | 26.691937 | 1.258675× |
| 2 | V4,V3 | 0.101441 | 21.169722 | 0.075246 | 28.539475 | 1.348127× |
| 3 | V3,V4 | 0.106365 | 20.189796 | 0.070090 | 30.639119 | 1.517555× |
| 4 | V4,V3 | 0.111626 | 19.238161 | 0.069847 | 30.745521 | 1.598153× |
| 5 | V3,V4 | 0.110378 | 19.455793 | 0.073713 | 29.133181 | 1.497404× |
| 6 | V4,V3 | 0.101276 | 21.204237 | 0.070056 | 30.653675 | 1.445639× |
| 7 | V3,V4 | 0.107422 | 19.991083 | 0.073908 | 29.056237 | 1.453460× |
| **中位数** | — | **0.106365** | **20.189796** | **0.073713** | **29.133181** | **1.442966×** |

### 7.3 2048³（5 warmups / 20 iterations）

| 轮次 | 顺序 | V3 ms | V3 TFLOPS | V4 ms | V4 TFLOPS | V4/V3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | V3,V4 | 0.791435 | 21.707235 | 0.559592 | 30.700705 | 1.414308× |
| 2 | V4,V3 | 0.762776 | 22.522823 | 0.557125 | 30.836662 | 1.369130× |
| 3 | V3,V4 | 0.831128 | 20.670546 | 0.530618 | 32.377119 | 1.566341× |
| 4 | V4,V3 | 0.793755 | 21.643789 | 0.544653 | 31.542789 | 1.457360× |
| 5 | V3,V4 | 0.812395 | 21.147182 | 0.534835 | 32.121799 | 1.518964× |
| 6 | V4,V3 | 0.769784 | 22.317778 | 0.577675 | 29.739668 | 1.332555× |
| 7 | V3,V4 | 0.785698 | 21.865753 | 0.537626 | 31.955080 | 1.461421× |
| **中位数** | — | **0.791435** | **21.707235** | **0.544653** | **31.542789** | **1.453100×** |

V4 在全部 21 轮都快于 V3，三个 shape 的中位数加速均在 1.44×–1.53×，可判断为
稳定性能提升。跨会话绝对值仍可能随 GPU 状态改变，结论只采用本轮同进程交替比值。

## 8. 1024³ Nsight Compute 对照

### 8.1 采集方法

V3/V4 各发射 10 次相同输入的 1024³ workload，跳过前 5 次、采 1 次；使用与
V3 报告相同的 10 个 section、`--cache-control none`，各完成 35 replay passes：

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
  --kernel-name-base function --kernel-name fp8_gemm_v3 \
  --launch-skip 5 --launch-count 1 --kill yes \
  --cache-control none --force-overwrite \
  -o build/ncu/fp8_v3_1024_v4_compare_20261103 \
  ./build/cmake-sm120/fp8_gemm_v4_compare --profile-v3

# V4 将 kernel/output/mode 改为：
# fp8_gemm_v4 / fp8_v4_1024_compare_20261103 / --profile-v4
```

`--cache-control none` 会使跨 replay cache 状态不受控制，因此不使用异常的 L2
分项 hit rate，也不把 NCU Duration 当作 Event 时间。

### 8.2 主要指标

| 指标 | V3 | V4 | 变化/含义 |
|---|---:|---:|---|
| NCU Duration | 113.15 us | 79.04 us | profiler-only，约 1.43× |
| Registers/thread | 48 | 44 | -4 |
| Static shared memory | 0 | 1536 B | 新增 staging |
| Theoretical Occupancy | 83.33% | 83.33% | 不变 |
| Achieved Occupancy | 78.30% | 76.13% | -2.17 pp |
| Active warps/scheduler | 9.41 | 9.12 | 略降 |
| Eligible warps/scheduler | 0.16 | 0.55 | 3.4× |
| No eligible | 93.20% | 83.31% | 改善 9.89 pp |
| Global Load requests | 1,572,864 | 786,432 | -50% |
| Global Load sectors | 12,582,912 | 3,145,728 | -75% |
| Shared Load | 0 | 1,572,864 | 新增 |
| Shared Store | 0 | 786,432 | 新增 |
| Shared Load bank conflicts | 0 | 1,575,823 | 平均 2-way |
| Excessive Shared wavefronts | 0 | 1,572,864 | 40% total shared wavefronts |
| Barrier Stall | 0.00 | 9.04 | 新增同步等待 |
| LG Throttle | 39.58 | 0.30 | 基本消除 |
| Long Scoreboard | 91.72 | 15.51 | -83.1% |
| MIO Throttle | 0.94 | 22.13 | 新 Shared 瓶颈 |
| Warp cycles/issued inst | 138.29 | 54.67 | -60.5% |
| Tensor Pipe Active | 15.27% | 23.00% | +7.73 pp |
| DRAM throughput | 4.06% | 4.61% | 均远未饱和 |
| 全部动态指令 | 3,538,944 | 6,078,464 | +71.8% |

V4 使用 44 registers 后，register block limit 仍为 10 blocks/SM；1536 B static
smem 对应 Shared limit 25 blocks/SM，因此 theoretical occupancy 仍由 registers
限制在 40 warps/SM，即 83.33%。性能提升不是 Occupancy 提升造成的。

### 8.3 Shared bank conflict

NCU 明确报告：

```text
all 1,572,864 shared load requests average a 2.0-way bank conflict
1,575,823 load bank conflicts, 50.05% of 3,148,690 load wavefronts
```

静态地址也能解释该结果。A/B 的 word 地址都可写成：

```text
word_address = major_index * 8 + word_k
bank = word_address % 32
```

对一个 warp 的固定 fragment register，`major_index` 覆盖 8 个值；相差 4 的两行
或两列落到同一 bank，但地址不同，因此形成 2-way conflict。Shared Stores 由连续
thread word 编号产生，只有约 30 个 replay 采样冲突，基本可视为 conflict-free。

由于用户明确要求本轮不增加 Padding/Swizzling，本次只记录该瓶颈，没有修复。

### 8.4 性能为何提升、当前瓶颈为何改变

V3 的核心限制是大量直接 Global Load：同一 A tile 被横向两个 warp 重读，同一 B
tile 被纵向两个 warp 重读，同时 fragment lane layout 只使用每个 sector 的一半。

V4 同时获得两类收益：

1. A/B CTA tile 只从 global 读取一次，Global Load requests 减半；
2. cooperative word 编号使 Global Load 从 8 sectors/request 改为 4，sector 利用率
   从 16/32 bytes 提升到 32/32 bytes。

因此 sectors 下降 75%，LG Throttle 和 Long Scoreboard 大幅减少，可发射 warp 数和
Tensor Pipe Active 上升。即使 V4 动态指令增加 71.8%、Occupancy 略降，减少的
global latency 仍产生约 1.45× Event 提升。

V4 当前主要损失转为：

- Shared Load 2-way bank conflicts；
- 大量 LDS/STS 导致 MIO Throttle 22.13 cycles/issue；
- 每 K tile 两次 CTA 同步导致 Barrier Stall 9.04 cycles/issue；
- Shared staging 与 barrier 使总动态指令增加。

这些指标说明 V4 仍有后续优化空间，但本轮没有按结果直接修改 Kernel。

## 9. 实测与推断边界

### 已实测

- Canonical V1/V2/V3 前缀哈希未变。
- V4 9 组 V3/reference 逐元素比较全部误差为 0。
- PTX/SASS 中的 LDG/STS/LDS/BAR/QMMA 和 ptxas 资源结果。
- 21 轮交替 Event 全部正向提升。
- NCU Global/Shared 请求、sectors、bank conflicts、stall、Tensor Active 和
  Occupancy 指标。

### 推断

- barrier 放置正确由源码 happens-before、SASS 顺序和多 K tile 正确性共同支持；
  没有使用 racecheck 工具证明所有可能执行交错。
- V4 性能提升主要来自 global requests/sectors 减少；该推断由动态计数与 stall
  转移直接支持。
- 下一瓶颈是 Shared bank conflict/MIO/barrier；各因素的独立贡献尚未通过 Padding、
  Swizzle 或 barrier 消融实验分别量化。

## 10. 当前 V4 局限

- 仅支持紧密 A row-major、B column-major、C row-major。
- M/N/K 必须分别为 32/16/32 的倍数，无边界 tile。
- 输入需预先量化为 E4M3，无 scaling、batch、任意 leading dimension 或 epilogue。
- Shared `[major][8]` 布局产生稳定 2-way bank conflict。
- 每 K tile 有两次 CTA barrier，没有异步拷贝或计算/访存双缓冲。
- 仍只有 32×16 CTA output tile，未进行更大 tile 或更多 warp 的实验。

本轮没有实现这些后续优化。

## 11. 可复现命令

```bash
# 构建
cmake --build build/cmake-sm120 --target \
  fp8_gemm_v3_compare fp8_gemm_v4_compare -j 4

# 正确性
./build/cmake-sm120/fp8_gemm_v3_compare --correctness
./build/cmake-sm120/fp8_gemm_v4_compare --correctness

# 七轮交替 Event
./build/cmake-sm120/fp8_gemm_v4_compare --benchmark

# PTX/SASS
nvcc -std=c++17 -O3 -arch=compute_120 -code=compute_120 \
  -ptx src/fp8_gemm.cu -o build/fp8_v4_probe/fp8_gemm.ptx
cuobjdump --dump-sass build/cmake-sm120/fp8_gemm_v4_compare

# NCU reports
ncu --import build/ncu/fp8_v3_1024_v4_compare_20261103.ncu-rep
ncu --import build/ncu/fp8_v4_1024_compare_20261103.ncu-rep
```
