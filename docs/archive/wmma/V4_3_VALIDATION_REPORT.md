# Tensor Core WMMA GEMM V4_3 集成、验证与性能分析

日期：2026-11-02（Asia/Shanghai）

## 1. 结论摘要

本轮开始时，用户手写的 `wmmaGemmV4_3()` 已存在于工作树。静态审查未发现功能 Bug，因此没有改写 kernel；只清理了一行尾随空格，并完成 host 侧工程集成、正确性、SASS/NCU 和 CUDA Event 测试。冻结的 V3、V4_1、V4_2 kernel 均未修改。

主要结论：

- K=16、32、48 的 prologue/steady/epilogue 和 buffer ownership 正确；每个边界各重复 3 次均通过。
- 所有要求尺寸均通过 FP16 量化输入 CPU reference；V4_3 与 V3、V4_2 逐元素完全一致。
- V4_3 为 72 registers/thread、8704 B kernel static shared，无 spill；相对 V4_2 少 6 registers/thread，理论 occupancy 从 50.00% 提高到 58.33%。
- V4_3 executed warp instructions 为 15,638,528，比 V4_2 的 24,944,640 少 9,306,112，下降 37.31%。
- branch/predicate 下降 83.70%，integer/address/move 下降 61.17%，system/constant reads 下降 72.73%；本轮核心目标已明确达成。
- 编译器仍采用 4-tile unroll，但从 V4_2 的 32 个带条件展开块，变为 V4_3 的 31 个完整 4-tile 块、3 次 remainder 和 1 次显式 epilogue。
- V4_2 热循环内重复的 `S2R LANEID/TID/CTAID` 在 V4_3 steady-state fast path 中消失；global A 地址也变为 pointer induction，原 10-instruction A/B 地址链不再逐 tile 出现。
- 两批 1024³ 中，V4_3 均比 V4_2 快，分别为 1.0569x 和 1.0172x；相对 V3 的方向因短任务波动而不一致。
- 两批 2048³ 中，V4_3/V4_2 分别为 1.0202x 和 0.9981x；NCU replay duration 为 1.0347x。指令减少产生了可测性能潜力，但尚不能宣称稳定 Event 加速。
- 循环尾拆分值得保留，作为更干净的双缓冲基线；但不能仅凭一批 Event 数据把 V4_3 宣布为稳定优于 V3/V4_2 的最终版本。

## 2. 修改范围

修改文件：

- `src/tensor_core_gemm.cu`
  - 确认已有 `wmmaGemmV4_3()`，功能代码未改写。
  - `KernelVersion` 增加 `V4_3`。
  - `kernelName()` 增加 V4_3 名称。
  - `launchKernel()` 使用 V3/V4_2 相同的 grid 和 128-thread block 启动 V4_3。
  - `runCase()` 增加 V4_3 correctness、V3/V4.3 与 V4_2/V4.3 精确输出对照、V4.3/cuBLAS 对照。
  - 将 V4_3 加入 rotating-order interleaved benchmark，并输出 V4.3/V3、V4.3/V4.2、V4.3/cuBLAS 比例。
- `V4_3_VALIDATION_REPORT.md`
  - 本报告。

没有修改 tile、Shared Layout、WMMA 计算、C 写回或 Global Pointer 表达式，没有实现其他优化，也没有 Git commit/push。

## 3. 静态正确性与同步审查

### 3.1 Prologue

V4_3 与 V4_2 相同：向 `current=0` 的 A/B plane 复制 tile 0，执行 commit、`wait_prior(0)` 和 CTA barrier。首次 WMMA 前 tile 0 已对所有 warp 可见。

### 3.2 Steady State

循环条件为：

```cpp
for (int tile = 0; tile < num_k_tiles - 1; tile++)
```

因此恰好执行 `num_k_tiles-1` 次。每轮无条件执行：

```text
prefetch(tile+1 -> next) + commit
compute(tile <- current)
wait_prior(0) + CTA barrier
swap(current, next)
```

wait 保证本线程提交的 copy 完成，随后 CTA barrier 保证所有线程的 next plane 都可见，并保证所有 warp 已完成对旧 current plane 的 LDSM 读取。之后才交换 plane，因此没有提前读取或提前覆盖。

### 3.3 Epilogue 和边界 K

- K=16：`num_k_tiles=1`，steady loop 执行 0 次；epilogue 读取 prologue 已准备的 buffer 0，计算 tile 0。
- K=32：steady loop 执行 1 次，预取 tile 1、计算 tile 0、等待并交换；epilogue 从 buffer 1 计算 tile 1。
- K=48：循环两次。第二轮在 buffer 1 计算 tile 1，同时将 tile 2 写回已安全释放的 buffer 0；epilogue 从 buffer 0 计算 tile 2。
- 更大 K 按相同 ping/pong 归纳成立。

最后一 tile 不再预取，不需要额外 pipeline wait 或 barrier。Epilogue 没有后续 shared plane 复用，因此结尾无 CTA barrier 是安全的。

### 3.4 地址、fragment 和写回

V4_3 保持 V4_2 的双 buffer `As[2][64][16]` 和 `Bs[2][16][72]`，每个 A/B plane 分别为 2048 B/2304 B，均为 16 B 的倍数。Global row stride、每线程列偏移、shared row stride 和 plane stride 均继续满足 16-byte copy 对齐条件。

Steady 与 epilogue 的 A0/A1/B0/B1 都使用当前 `current` plane；WMMA fragment 类型、load stride、MMA 顺序和四块 C 写回与冻结版本相同。

## 4. 构建、资源和正确性

### 4.1 构建

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu \
  -o build/tensor_core_gemm_v4_3 -lcublas
```

CUDA 13.2 / `sm_120` 编译成功。唯一 warning 仍是既有 V0 的未使用 `lane_id`。

| Kernel | Registers/thread | Kernel static shared | Stack | Spill store/load |
|---|---:|---:|---:|---:|
| V3 | 78 | 4352 B | 0 | 0 / 0 B |
| V4_2 | 78 | 8704 B | 0 | 0 / 0 B |
| V4_3 | 72 | 8704 B | 0 | 0 / 0 B |

`cuobjdump` 显示 V4_3 total shared 为 9728 B，其中包含 1024 B driver shared；kernel 自身 static shared 为预期的 8704 B。

### 4.2 正确性结果

判定标准保持原工程不变：

```text
abs_error <= 1e-2 + 1e-2 * abs(FP16-quantized-input CPU reference)
```

| Shape | V4_3 quantized-ref max abs | max rel | V4_3 | V3/V4_2 vs V4_3 |
|---|---:|---:|---|---|
| 64×64×16 | 4.768372e-07 | 3.003707e-06 | PASS | max abs=0、max rel=0 |
| 64×64×32 | 5.960464e-07 | 1.157327e-05 | PASS | max abs=0、max rel=0 |
| 64×64×48 | 5.364418e-07 | 2.008032e-03 | PASS | max abs=0、max rel=0 |
| 128×192×64 | 8.344650e-07 | 8.325396e-07 | PASS | max abs=0、max rel=0 |
| 1024³ | 6.079674e-06 | 1.020690e-05 | PASS | max abs=0、max rel=0 |
| 2048³ | 1.084805e-05 | 3.015384e-04 | PASS | max abs=0、max rel=0 |

K=16/32/48 各运行 3 个独立进程，9/9 均通过 V4_3 reference 和 V3/V4_2 exact comparison。所有正式运行执行 `cudaGetLastError()`、device synchronization 和 D2H copy，exit code 均为 0，没有观察到 CUDA Runtime error。

### 4.3 Compute Sanitizer 限制

已对 64×64×48 的 V4_3 尝试 memcheck。工具在插桩前报告：

```text
Failed to initialize WDDM debugger interface
Please run EnableDebuggerInterface.bat as an administrator
Device not supported
```

程序随后正常运行，但 sanitizer 汇总的两个错误是上述初始化错误，并非具体 device address 或 thread hazard。racecheck 依赖相同的 WSL/WDDM debugger interface，因此本轮不能宣称 memcheck/racecheck clean。

## 5. 动态 SASS 指令分析

### 5.1 方法

沿用 `V4_2_INSTRUCTION_ANALYSIS.md`：从同配置 2048³ NCU Source page 读取每条 SASS 的 `Instructions Executed`，按 mnemonic 和功能类别汇总。各类别之和严格等于 NCU Details 中的总指令数。

报告文件：

- V3：`build/ncu/v4_1_v3_2048.ncu-rep`
- V4_2：`build/ncu/v4_2_v4_2_2048.ncu-rep`
- V4_3：`build/ncu/v4_3_v4_3_2048.ncu-rep`

### 5.2 完整动态分类

| 类别 | V3 | V4_2 | V4_3 | V4_3 - V4_2 |
|---|---:|---:|---:|---:|
| Integer / address / move | 4,288,512 | 11,812,864 | 4,587,520 | -7,225,344 |
| Branch / predicate / exit | 286,720 | 1,859,584 | 303,104 | -1,556,480 |
| System / constant reads | 192,512 | 720,896 | 196,608 | -524,288 |
| Global→shared staging | 2,097,152 | 1,048,576 | 1,048,576 | 0 |
| Pipeline sync | 0 | 1,048,576 | 1,048,576 | 0 |
| CTA barrier | 1,048,576 | 524,288 | 524,288 | 0 |
| LDS + LDSM | 2,097,152 | 3,670,016 | 3,670,016 | 0 |
| HMMA | 4,194,304 | 4,194,304 | 4,194,304 | 0 |
| C global stores | 65,536 | 65,536 | 65,536 | 0 |
| **Total** | **14,270,464** | **24,944,640** | **15,638,528** | **-9,306,112** |

相对 V4_2：

- 总指令下降 37.3071%。
- Integer/address/move 下降 61.1650%。
- Branch/predicate 下降 83.7004%。
- System/constant reads 下降 72.7273%。

V4_3 总指令仍比 V3 多 1,368,064。主要剩余差异来自 async lowering 中 1,572,864 条 predicated-off `LDS RZ,[RZ]`；这组指令在 V4_2/V4_3 中完全相同。

### 5.3 循环展开策略

NCU 动态频率分组：

| Version | 动态频率 | 静态槽位数 | 含义 |
|---|---:|---:|---|
| V4_2 | 131,072 | 168 | 32 个完整 4-tile blocks |
| V4_2 | 126,976 | 18 | 最后一块跳过 final prefetch |
| V4_3 | 126,976 | 111 | 31 个完整 4-tile steady blocks |
| V4_3 | 12,288 | 45 | 3 个 remainder steady iterations |
| V4_3 | 4,096 | 242 | prologue、显式 epilogue、C 写回和一次性 setup |

其中 4096 是 grid 中的 warp 数。V4_3 仍然是 4-way unroll；改变的是控制流形状，而不是 unroll factor：

```text
V4_2: 128 = 32 * 4，展开块内保留条件路径
V4_3: steady 127 = 31 * 4 + 3，之后显式计算最后 1 tile
```

V4_3 完整 steady fast path 从 V4_2 的 186 个条件相关热槽位降为 111 个无条件热槽位。Epilogue 增加了一次性静态路径，但没有增加 LDSM/HMMA 总执行次数。

### 5.4 Branch/predicate

V4_2 的两个 `tile+1<num_k_tiles` 条件在四路展开后生成多组 `ISETP/BRA`。V4_3 steady fast path 只保留 uniform loop back-edge：

```text
UIADD3 ... loop induction ...
UISETP.NE ... UP0 ...
...
BRA.U UP0, <steady-loop-head>
```

动态 branch/predicate 从 1,859,584 降到 303,104，接近 V3 的 286,720。Branch Efficiency 仍为 100%，Avg. Divergent Branches 为 0；收益来自减少 uniform control instructions，而非修复 divergence。

### 5.5 `S2R` 和地址计算

| Metric | V4_2 | V4_3 |
|---|---:|---:|
| Total `S2R` | 417,792 | 24,576 |
| Hot steady-loop `S2R` | 393,216 | 0 |

V4_2 每个四 tile 展开块重新读取 `LANEID/TID/CTAID`；V4_3 的完整 steady fast path 不再包含 `S2R`。剩余 24,576 次只出现在一次性或 remainder/epilogue 路径。

V4_2 每次 prefetch 的 A/B global 64-bit 地址需要约 10 条连续构造指令。V4_3 的完整 fast path 改为：

```text
LDGSTS ... desc[...][R66.64 + static_offset], [A-shared-plane]

SHF.R.S32.HI ... B-index
MOV            ...
IADD.64        ...
LEA / LEA.HI.X ... B-global-address
LDGSTS ... B ...
```

A 使用 `R66` pointer induction 和静态偏移；B 保留约 5 条地址构造。四 tile 块尾部统一执行：

```text
IADD.64 R66, R66, 0x80
IMAD R61, ..., 0x40, R61
IMAD R60, ..., 0x40, R60
IMAD R59, ..., 0x40, R59
IMAD R58, ..., 0x40, R58
```

因此，虽然 V4_3 源码没有手写 Global Pointer 递推，删除内层条件后编译器已自动把大部分地址计算转化为 induction。完整 steady loop 的 integer/address/move 为 3,964,928，折合约 7.62 条/tile/warp，已经接近 V3/V4_1 的 7.5。

### 5.6 Epilogue 开销

V4_3 的显式 epilogue 以每 warp 一次的 4096 动态频率执行，包含 4 条 LDSM、8 条 HMMA 及一次性 shared/output 地址准备。与 V4_2 比较：

- HMMA 总数仍为 4,194,304。
- LDSM 总数仍为 2,097,152。
- LDGSTS/LDGDEPBAR/DEPBAR/barrier 总数也完全相同。

因此 epilogue 没有重复计算最后一 tile，只把最后一 tile 的工作从 guarded loop 移到一次性路径。V4_3 的 4096-frequency group 比 V4_2 多 352,256 条动态指令，但热循环减少约 9.66M，净减少仍为 9.306M。

### 5.7 重叠关系

V4_3 完整 steady SASS 仍保持：

```text
LDGSTS next A/B
LDSM current
LDGDEPBAR
HMMA current
...
DEPBAR.LE SB0, 0x0
BAR.SYNC
```

部分 HMMA 还被调度到下一组地址/copy 指令附近。静态顺序继续提供 copy/compute overlap 机会；没有产生新的额外 wait 或 barrier。

## 6. CUDA Event Benchmark

协议：同一进程、同输入、默认 stream、每轮 10 次 warmup、100 次 kernel 的平均延迟、5 个 rotating-order rounds，最终取各批 5 轮中位数。所有 cuBLAS 数值均来自对应当前批次，没有使用历史数据。鉴于波动，1024³ 和 2048³ 都执行了两批独立测量，不跨批混合中位数。

### 6.1 1024³，批次 1

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.045271, 0.055456, 0.045204, 0.045389, 0.046939 | 0.045389 | 47.312411 |
| V4_1 | 0.049380, 0.047580, 0.049694, 0.059434, 0.050936 | 0.049694 | 43.214353 |
| V4_2 | 0.057468, 0.055797, 0.050579, 0.050015, 0.050002 | 0.050579 | 42.458109 |
| V4_3 | 0.047730, 0.047671, 0.058655, 0.058944, 0.047857 | 0.047857 | 44.872663 |
| cuBLAS | 0.044956, 0.040025, 0.040432, 0.040108, 0.042333 | 0.040432 | 53.113047 |

- V4_3/V4_2：1.0569x
- V4_3/V3：0.9484x
- V4_3/cuBLAS：0.8449

### 6.2 1024³，批次 2

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.052324, 0.047376, 0.057191, 0.047258, 0.055477 | 0.052324 | 41.042164 |
| V4_1 | 0.050019, 0.056351, 0.056451, 0.049895, 0.047856 | 0.050019 | 42.933736 |
| V4_2 | 0.046706, 0.048386, 0.046758, 0.050093, 0.048974 | 0.048386 | 44.382406 |
| V4_3 | 0.057675, 0.048782, 0.047249, 0.047240, 0.047570 | 0.047570 | 45.143731 |
| cuBLAS | 0.040532, 0.043065, 0.043490, 0.043772, 0.042026 | 0.043065 | 49.866144 |

- V4_3/V4_2：1.0172x
- V4_3/V3：1.0999x
- V4_3/cuBLAS：0.9053

V4_3 两批 median 只相差约 0.60%，但各批仍有约 0.058 ms 高值；V3 两批 median 方向差异更大。因此只能确认 V4_3 两批均优于 V4_2，不能确认其稳定优于 V3。

### 6.3 2048³，批次 1

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.311554, 0.298867, 0.312167, 0.320599, 0.315625 | 0.312167 | 55.034275 |
| V4_1 | 0.321832, 0.334710, 0.305092, 0.325969, 0.328505 | 0.325969 | 52.704058 |
| V4_2 | 0.305103, 0.298698, 0.309469, 0.316193, 0.307365 | 0.307365 | 55.893951 |
| V4_3 | 0.301270, 0.311078, 0.296953, 0.312676, 0.300475 | 0.301270 | 57.024869 |
| cuBLAS | 0.308157, 0.295153, 0.304870, 0.294920, 0.311472 | 0.304870 | 56.351385 |

- V4_3/V4_2：1.0202x
- V4_3/V3：1.0362x
- V4_3/cuBLAS：1.0120

### 6.4 2048³，批次 2

| Implementation | 5 轮原始平均延迟（ms） | Median ms | Median TFLOPS |
|---|---|---:|---:|
| V3 | 0.301386, 0.304583, 0.307359, 0.308646, 0.306640 | 0.306640 | 56.026185 |
| V4_1 | 0.320036, 0.323003, 0.322197, 0.321082, 0.320070 | 0.321082 | 53.506244 |
| V4_2 | 0.305470, 0.308478, 0.305306, 0.309995, 0.309492 | 0.308478 | 55.692295 |
| V4_3 | 0.309066, 0.310756, 0.310664, 0.307903, 0.303809 | 0.309066 | 55.586367 |
| cuBLAS | 0.294564, 0.295471, 0.289214, 0.293117, 0.290555 | 0.293117 | 58.611000 |

- V4_3/V4_2：0.9981x
- V4_3/V3：0.9921x
- V4_3/cuBLAS：0.9484

两批 2048³ 对 V4_2/V3 的方向不一致，差值都在约 0–3.6% 区间，且批次 1 的 cuBLAS 也有明显离散。不能把批次 1 的超越 cuBLAS 视为稳定结论。

Benchmark 后的空闲状态快照为 42.9°C、SM 847 MHz、显存 14001 MHz、33.03/250 W、GPU utilization 6%。它不是计时期间连续遥测，只用于说明采样后环境，不能解释具体轮次。

## 7. NCU 性能指标

NCU replay duration 只作 profiler 内方向比较，不代替 CUDA Event。Stall 指标单位为 cycles/issued，不是时间百分比。

| Metric | V3 | V4_2 | V4_3 |
|---|---:|---:|---:|
| NCU replay duration | 325.60 us | 328.42 us | 317.41 us |
| Registers/thread | 78 | 78 | 72 |
| Theoretical occupancy | 50.00% | 50.00% | 58.33% |
| Achieved occupancy | 46.00% | 46.29% | 52.32% |
| Active warps/scheduler | 5.53 | 5.58 | 6.30 |
| Eligible warps/scheduler | 0.15 | 0.23 | 0.18 |
| Issue Slots Busy | 9.20% | 15.96% | 10.39% |
| Warp cycles/issued | 58.43 | 33.71 | 59.67 |
| Long Scoreboard | 18.847 | 10.138 | 20.463 |
| Short Scoreboard | 3.637 | 2.132 | 2.111 |
| Barrier Stall | 7.399 | 4.133 | 8.334 |
| Tensor Pipe Active | 87.316% | 88.065% | 89.697% |
| HMMA subpipe active | 87.320% | 88.032% | 89.697% |
| DRAM throughput | 14.93% | 15.06% | 15.04% |
| L1/TEX throughput | 75.00% | 64.24% | 66.86% |
| L2 throughput | 62.75% | 82.36% | 85.49% |

V4_3 的 Long Scoreboard、Barrier Stall 和 warp cycles/issued 比 V4_2 高，但动态 barrier、LDGSTS 和 pipeline sync 数量完全相同，NCU duration 反而更短。原因是 V4_3 删除了大量可 issue 的整数/控制指令，`cycles/issued` 的分母显著变小；不能把这些数当成 wall-time 百分比。Short Scoreboard 基本不变，Tensor/HMMA active 提高约 1.63 pp。

### 7.1 Global 与 LDSM

| Metric | V4_2 | V4_3 |
|---|---:|---:|
| Global load requests | 1,048,576 | 1,048,576 |
| Actual global load sectors | 17,825,792 | 17,825,792 |
| Source theoretical/ideal/excessive sectors | 20,971,520 / 16,777,216 / 4,194,304 | 相同 |
| LDSM instructions | 2,097,152 | 2,097,152 |
| LDSM actual/ideal/excessive wavefronts | 12,582,912 / 8,388,608 / 4,194,304 | 相同 |

V4_3 没有改变 global coalescing、V4_2 的 B-side sector 增量或 LDSM bank-conflict 属性。这符合本轮只改变循环尾结构的约束。

## 8. 事实、解释与下一步

### 8.1 已实测确认

- 循环尾拆分把 V4_2 动态总指令减少 37.31%。
- hot-loop branch 和重复 `S2R` 基本消除。
- 编译器在 V4_3 fast path 自动采用更紧凑的 pointer induction。
- V4_3 少 6 registers/thread、occupancy 提高、无 spill。
- copy、sync、barrier、LDSM、HMMA 和 memory transaction 工作量没有变化。
- NCU duration 改善 3.47%，但两批 CUDA Event 对 V4_2/V3 的方向并不完全一致。

### 8.2 分析解释

V4_2 的条件控制阻碍了 backend 对 unrolled loop 的 induction 和 invariant hoisting。V4_3 删除热循环条件后，不只是减少 branch，还触发了更紧凑的 global/shared address codegen；这解释了 integer/address 类别同时大幅下降。

V4_3 更少的 issueable filler instructions 使 cycles/issued 类 stall 数值上升，但 Tensor active 和 NCU duration 改善。该现象说明 stall ratio 与总时间不能线性对应。

### 8.3 尚不能确认

- 现有 Event 数据不足以证明 V4_3 稳定快于 V4_2 或 V3。
- 无法把 37% 指令减少线性换算为性能收益；Tensor pipeline 已接近 90% active，减少非 Tensor 指令的边际收益有限。
- 缺少计时期间连续频率/温度/系统负载遥测，不能为批次差异指定唯一原因。

## 9. 最终建议

V4_3 的循环尾拆分值得保留。理由不是“已稳定加速”，而是它在保持完全相同结果与内存/WMMA 工作量的前提下，显著减少动态控制和地址指令、降低寄存器，并让 SASS 更接近 V3/V4_1 的紧凑循环。它是后续单变量实验更合理的双缓冲基线。

下一步若只选择一个变量，可以测试显式 Global Pointer 递推，但预期收益应低于 `V4_2_INSTRUCTION_ANALYSIS.md` 中的初始判断：V4_3 编译器已经自动对 A 和部分循环状态做了 pointer induction，完整 steady loop 的 integer/address/move 已从 V4_2 的约 22.09 条/tile 降到约 7.62，接近 V3 的 7.5。下一实验应只针对仍保留的 B-side 64-bit `SHF/MOV/IADD64/LEA` 链，并以 SASS/NCU 是否进一步下降为成功标准；不要同时修改 shared pointers、tile 或 unroll policy。
