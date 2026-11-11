# Tensor Core WMMA GEMM V4_2 动态指令开销专项诊断

日期：2026-11-02（Asia/Shanghai）

## 1. 范围与结论

本轮仅诊断现有 V3、V4_1、V4_2，没有修改任何 kernel 源码，没有实现 V4_3，也没有执行 Git commit/push。

核心结论：V4_2 相比 V4_1 增加的 9,105,408 条 executed warp instructions，几乎全部来自稳态 K 循环中的整数/地址重建、条件控制和少量系统寄存器/常量读取，而不是 `LDGSTS`、pipeline sync、LDSM 或 HMMA 工作量增加。

2048³ 下，每个 CTA 有 128 个 K tile，grid 为 1024 CTA，每 CTA 4 warp，因此共有：

```text
1024 CTA * 4 warp/CTA * 128 K tiles = 524,288 warp-tile iterations
```

按该数量归一化，热循环每个 tile/warp 的动态指令为：

| Version | 热循环指令/tile/warp |
|---|---:|
| V3 | 26.0000 |
| V4_1 | 29.0000 |
| V4_2 | 46.3594 |

V4_2 比 V4_1 每 tile/warp 多约 17.36 条热循环指令。其中：

- 整数、地址和 move：增加约 14.42 条。
- 分支和 predicate：增加 3.00 条。
- system/constant 读取：增加 1.00 条。
- CTA barrier：减少约 1.01 条。
- staging、pipeline、LDSM、HMMA 的有效工作量基本不变。

因此，V4_2 的 double buffering 确实降低了等待和 barrier 开销，但编译后的地址/控制开销消耗了大量 scheduler issue slots，足以解释它为何只追回 V4_1 的性能退化，而未稳定超过 V3。

## 2. 数据来源与归因方法

使用 V4_2 验证阶段生成的同口径 2048³ NCU 报告：

- `build/ncu/v4_1_v3_2048.ncu-rep`
- `build/ncu/v4_1_v4_1_2048.ncu-rep`
- `build/ncu/v4_2_v4_2_2048.ncu-rep`

三者使用相同的 9 个 NCU section、`--launch-count 1` 和 35-pass 配置。SM frequency 分别为 2.48、2.49、2.48 GHz。

归因步骤：

1. 从 NCU Source page 读取每条 SASS 的 `Instructions Executed`。
2. 按 SASS mnemonic 汇总动态执行次数。
3. 再按功能归入整数/地址、控制、staging、pipeline、barrier、shared load、Tensor 等类别。
4. 所有类别之和分别严格等于 NCU Details page 的 14,270,464、15,839,232、24,944,640，说明没有遗漏动态指令类别。
5. 静态 SASS 只用于识别展开结构和指令用途；所有执行次数结论均来自 Source Counters，而不是静态行数乘以主观循环次数。

这里的 `Instructions Executed` 是 warp instruction 口径，不是逐线程 scalar instruction 数。NCU 会把 predicated-off 的指令槽也计入 executed warp instructions，例如 `@!PT LDS RZ, [RZ]`；这类指令没有产生对应 shared-memory request，但仍占动态指令数。

## 3. 全 kernel 动态指令拆分

### 3.1 功能类别

| 类别 | V3 | V4_1 | V4_2 | V4_2 - V4_1 |
|---|---:|---:|---:|---:|
| Integer / address / move | 4,288,512 | 4,284,416 | 11,812,864 | +7,528,448 |
| Branch / predicate / exit | 286,720 | 290,816 | 1,859,584 | +1,568,768 |
| System / constant reads | 192,512 | 188,416 | 720,896 | +532,480 |
| Global→shared staging | 2,097,152 | 1,048,576 | 1,048,576 | 0 |
| Pipeline sync | 0 | 1,048,576 | 1,048,576 | 0 |
| CTA barrier | 1,048,576 | 1,048,576 | 524,288 | -524,288 |
| Shared/fragment loads | 2,097,152 | 3,670,016 | 3,670,016 | 0 |
| HMMA | 4,194,304 | 4,194,304 | 4,194,304 | 0 |
| C global stores | 65,536 | 65,536 | 65,536 | 0 |
| **Total** | **14,270,464** | **15,839,232** | **24,944,640** | **+9,105,408** |

V4_2 相比 V4_1 的净增量中：

- integer/address/move 占净增量约 82.7%；
- branch/predicate 占约 17.2%；
- system/constant 占约 5.8%；
- barrier 减少抵消约 5.8%。

比例之和超过 100% 是因为 barrier 是负增量。

### 3.2 搬运与计算指令没有增加

| SASS 类别 | V3 | V4_1 | V4_2 |
|---|---:|---:|---:|
| `LDG.E.128` | 1,048,576 | 0 | 0 |
| `STS.128` | 1,048,576 | 0 | 0 |
| `LDGSTS` | 0 | 1,048,576 | 1,048,576 |
| `LDGDEPBAR` | 0 | 524,288 | 524,288 |
| `DEPBAR` | 0 | 524,288 | 524,288 |
| `LDSM` | 2,097,152 | 2,097,152 | 2,097,152 |
| `HMMA` | 4,194,304 | 4,194,304 | 4,194,304 |

V4_1 和 V4_2 的 async copy、commit/wait lowering、WMMA shared loads 和 Tensor 指令次数完全一致。动态增长不能归因于“double buffer 复制了更多数据”或“做了更多 MMA”。

V4_1/V4_2 还各有 1,572,864 条 `LDS`：

```text
@!PT LDS RZ, [RZ]
@!PT LDS RZ, [RZ]
@!PT LDS RZ, [RZ]
LDGSTS.E.BYPASS.128 ...
```

它们是 predicated-off 的 SASS instruction slots，没有 Source Counter shared request。二者数量相同，所以能解释 V4_1 相比 V3 的部分指令增量，但不能解释 V4_2 相比 V4_1 的增量。

## 4. 增量位于稳态 K 循环

按每条静态 SASS 的动态执行次数分组：

| Version | setup/epilogue 动态指令 | 热循环动态指令 | 热循环静态槽位 |
|---|---:|---:|---:|
| V3 | 638,976 | 13,631,488 | 104 |
| V4_1 | 634,880 | 15,204,352 | 116 |
| V4_2 | 638,976 | 24,305,664 | 186 |

V4_2 与 V4_1 的 setup/epilogue 只差 4,096 条；9.105M 增量中的 9.101M 来自热循环。

NCU Source Counters 还揭示了编译器的展开结构：

- V3/V4_1 热循环槽位各执行 131,072 次。
- V4_2 有 168 个槽位执行 131,072 次，另有 18 个 final-prefetch 路径槽位执行 126,976 次。
- 2048³ 一共有 4,096 个 warp；`131,072 / 4,096 = 32`，而 K 有 128 tile，因此主循环被按 4 tile 展开。
- `126,976 / 4,096 = 31`，对应只有前 31 个四 tile 展开块执行最后一组 prefetch；最后一个展开块正确跳过越界预取。

这既证明了 4-way unroll，也证明了末 tile 条件在动态计数中确实生效。

## 5. 热循环的每 tile 归一化

| 类别（每 tile/warp） | V3 | V4_1 | V4_2 |
|---|---:|---:|---:|
| Integer / address / move | 7.5000 | 7.5000 | 21.9219 |
| Branch / predicate | 0.5000 | 0.5000 | 3.5000 |
| System / constant | 0 | 0 | 1.0000 |
| Global→shared staging | 4.0000 | 2.0000 | 1.9844 |
| Pipeline sync | 0 | 2.0000 | 1.9844 |
| CTA barrier | 2.0000 | 2.0000 | 0.9922 |
| LDS + LDSM | 4.0000 | 7.0000 | 6.9766 |
| HMMA | 8.0000 | 8.0000 | 8.0000 |
| **Total** | **26.0000** | **29.0000** | **46.3594** |

V4_2 的 staging、pipeline、barrier 和 predicated-off LDS 略低于整数，是因为最后一个 tile 只计算、不预取，相关次数为 `127/128`。这不是漏计。

### 5.1 Integer/address/move 明细

| Mnemonic（热循环，每 tile/warp） | V4_1 | V4_2 | 增量 |
|---|---:|---:|---:|
| `IMAD*` | 1.2500 | 5.4922 | +4.2422 |
| `LEA*` | 2.0000 | 4.4688 | +2.4688 |
| `IADD*` | 1.7500 | 3.7344 | +1.9844 |
| `MOV` | 1.0000 | 2.4844 | +1.4844 |
| `UIADD3` | 0.5000 | 2.2500 | +1.7500 |
| `SHF*` | 1.0000 | 1.4922 | +0.4922 |
| `LOP3` | 0 | 1.0000 | +1.0000 |
| `IADD3` | 0 | 1.0000 | +1.0000 |
| **Total** | **7.5000** | **21.9219** | **+14.4219** |

绝大部分增量是 wide/global pointer 构造和 shared/fragment 地址重建，而不是一个单独的 `current ^= 1`。

### 5.2 Branch/predicate 明细

V4_1 热循环每 4-tile 展开块只有一个 `UISETP` 和一个 back-edge `BRA`，折合 0.5 条/tile。

V4_2 热循环为：

| Mnemonic | 每 tile/warp |
|---|---:|
| `BRA` | 2.2500 |
| `ISETP` | 1.0000 |
| `UISETP` | 0.2500 |
| **Total** | **3.5000** |

源码在每个 tile 对 `tile + 1 < num_k_tiles` 检查两次：一次包围 prefetch/commit，一次包围 wait/barrier。SASS 保留了独立 predicate 和 branch，因为两段中间隔着整个 WMMA 计算。分支效率仍为 100%，平均 divergent branches 为 0；它们是 CTA/warp-uniform 控制，因此没有 divergence，但仍需 issue 和执行。

## 6. 关键 PTX/SASS 证据

### 6.1 V4_1 的紧凑 pointer induction

V4_1 的 4-tile 展开块使用寄存器递增维护 global pointer，并在循环尾统一推进：

```text
UIADD3 UR4, ..., UR4, 0x4, ...
IADD    R59, R59, 0x40
IADD    R60, R60, 0x40
...
LDGSTS.E.BYPASS.128 ...
...
UISETP.NE... UP0, UR4, URZ
...
BRA.U UP0, <loop-head>
```

四个 tile 共 116 个热路径 instruction slots，即 29 条/tile。

### 6.2 V4_2 重复读取线程映射状态

V4_2 每个 4-tile 展开块开头重新出现：

```text
S2R        R30, SR_LANEID
S2R        R2,  SR_TID.X
S2R        R65, SR_CgaCtaId
SHF.R.U32  ...
LOP3.LUT   ...
IMAD.SHL   ...
...
IMAD.U32   R71, R64, 0x2, R55
IMAD.U32   R69, R74, 0x2, R53
```

这些值在 CTA 生命周期内不变，但 backend 在热循环中 rematerialize 了 lane/thread/CTA 映射及其派生地址。三条 `S2R` 加一条循环内 `LDC` 正好形成每 tile 1.0 条 system/constant 指令，即 524,288 条动态热循环指令。

源码已经在循环外定义 `tid`、`warp_row`、`warp_col`，因此不能简单断言“把 C++ 局部变量移出去”必然解决问题；实测说明的是最终 backend 没有把所有派生地址保持为循环不变量。

### 6.3 每次 prefetch 的 global 地址链

在两条 `LDGSTS` 前，V4_2 为 A/B 同时构造 64-bit global 地址：

```text
IADD       R38, R58, <tile-offset>
MOV        R39, RZ
SHF.R.S32  R29, RZ, 0x1f, <B-index>
MOV        R28, <B-index>
IMAD.WIDE  R38, R62, UR6, R38
IADD.64    R36, R72, R28
LEA        R28, ..., R38, UR8,  0x1
LEA        R30, ..., R36, UR12, 0x1
LEA.HI.X   R29, ..., R38, UR9,  ...
LEA.HI.X   R31, ..., R36, UR13, ...
LDGSTS.E.BYPASS.128 ... A ...
LDGSTS.E.BYPASS.128 ... B ...
LDGDEPBAR
```

这个 10-instruction 地址链在 127 次 prefetch 上执行，动态计数为：

```text
10 * 127 tiles * 4,096 warps = 5,201,920 instructions
```

这是 Source Counter 与具体 SASS 序列共同支持的实际计数。它本身已超过 V4_1 整个热循环 integer/address/move 类别的 3,932,160 条。并非这 5.20M 都可消除，但它明确标出了最大单一可研究区域。

### 6.4 Shared buffer/fragment 地址也被重建

在 `LDGDEPBAR` 后、LDSM/HMMA 前，V4_2 还执行了多组：

```text
LEA / IMAD.SHL / IADD / LOP3
IADD3 / IMAD.U32
LDSM.16.M88.4 / LDSM.16.MT88.4
```

它们将 warp/lane 映射与当前 shared plane 偏移组合成 A0/A1/B0/B1 的 LDSM 地址。V4_1 的单 buffer LDSM 基址可以跨循环复用；V4_2 的 `current` plane 交替使 live pointer 关系更复杂，编译器选择在展开块内重建部分地址。

值得注意的是，V4_2 快路径中没有显式 `XOR current` 成为热点。编译器已经把 ping/pong 展开成固定 shared 偏移：

```text
LDGSTS ... [R61]
LDGSTS ... [R57]
...
LDGSTS ... [R61+0x800]
LDGSTS ... [R57+0x900]
```

`0x800` 是一个 A plane，`0x900` 是一个 B plane。因此只优化源码中的两个 `^= 1`，预期不会直接解决主要开销；真正问题是两套地址的 rematerialization 和 guarded unrolled control flow。

### 6.5 重叠调度仍然成立

V4_2 稳态 SASS 保持：

```text
LDGSTS(A next)
LDGSTS(B next)
LDGDEPBAR
LDSM(current)
HMMA(current)
...
DEPBAR.LE SB0, 0x0
BAR.SYNC
```

因此动态指令增长并不否定双缓冲重叠机会。它说明的是：等待减少与额外 issue 工作同时发生。

## 7. 寄存器和调度差异

| Metric | V3 | V4_1 | V4_2 |
|---|---:|---:|---:|
| Registers/thread | 78 | 72 | 78 |
| Theoretical occupancy | 50.00% | 58.33% | 50.00% |
| Achieved occupancy | 46.00% | 52.47% | 46.29% |
| Active warps/scheduler | 5.53 | 6.38 | 5.58 |
| Eligible warps/scheduler | 0.15 | 0.15 | 0.23 |
| Issue Slots Busy | 9.20% | 9.73% | 15.96% |
| Warp cycles/issued instruction | 58.43 | 63.14 | 33.71 |
| NCU replay duration | 325.60 us | 340.13 us | 328.42 us |

实测事实：V4_2 回到 78 registers/thread，因而失去 V4_1 的 occupancy 优势；但 eligible warps 和 issue-slot utilization 反而改善，warp cycles/issued 明显下降。

分析解释：V4_2 同时保持当前 WMMA accumulator/fragments、下一 tile global/shared addresses、buffer parity、多个 unrolled predicate 和地址递推状态，live range 比 V4_1 复杂。这与物理寄存器从 72 回升到 78 相符，但 NCU/ptxas 不提供“每个源码变量占多少寄存器”的直接归因，所以这部分是基于 SASS live state 的解释，不是逐变量实测。

V4_2 的 15.96% Issue Slots Busy 一部分来自更好的 copy/compute 安排，另一部分也用于执行额外整数和控制指令。Tensor/HMMA 指令数完全相同，所以更高的 instruction issue rate 不能单独视为有效计算吞吐提升。

## 8. 明确存在和不存在的冗余机会

### 已由动态证据确认

- 两个 `tile + 1 < num_k_tiles` 源码条件在 SASS 中形成多组独立 predicate/branch。
- 每 4-tile 展开块重新读取 lane id、thread id、CTA id，并重建其派生地址。
- 每次 prefetch 都有一条 10-instruction A/B global address chain。
- shared plane 交替导致额外的 shared/LDSM address materialization。
- 最后一个 tile 的 copy/wait/barrier 被正确跳过，没有越界 copy 或无用最后一次 pipeline wait。

### 未发现或不能据此认定

- `LDGSTS`、LDSM、HMMA 没有重复执行。
- 没有 register spill。
- `current ^= 1`/`next ^= 1` 没有直接变成每 tile 两条显式 XOR 热点。
- 静态 SASS 和 Source Counters不能证明某条整数指令可以在不改变寄存器压力或调度的情况下完全删除。

## 9. 候选优化方案（仅建议，未实施）

最多保留以下三个候选，建议按顺序独立实验，每项都重新检查 correctness、SASS、registers 和 Event benchmark。

### 1. 将最后一 tile 从 steady-state loop 中拆出

思路：steady loop 只处理 `0 .. num_k_tiles-2`，其中 prefetch、wait 和 barrier 无条件执行；最后一 tile 在 loop 外只计算。保留现有 prologue 和双 buffer 语义。

- 直接目标：把 branch/predicate 从 3.5 条/tile 向普通 loop 的约 0.5 条/tile靠近；理论直接上界约 1.57M 条动态控制指令。
- 间接机会：消除 4-way unroll 中多套 guarded basic blocks，可能让地址递推和 pointer hoisting 更容易。
- 预期收益：中等，证据置信度高。
- 正确性风险：低到中；必须特别覆盖 K=16、32、48 和最后一次 buffer ownership。
- 实现复杂度：低到中。

### 2. 用递推 pointer 代替每 tile 的 A/B 64-bit 地址重建

思路：在 prologue/loop setup 中建立 next A/B global pointers，steady state 只按固定字节步长推进；避免每 tile 从 `(tile + 1) * BLOCK_K` 和 row index 重建两条 64-bit 地址。

- 直接目标：当前 5,201,920 条 prefetch address-chain 指令；这是研究上界，不代表它们能全部消失。
- 预期收益：中到高，若 ptxas 能保留 pointer induction 而不增加 spill。
- 正确性风险：中；A 的推进步长与 B 的推进步长不同，且必须保持最后 tile 不预取。
- 实现复杂度：中。

### 3. 显式预计算 ping/pong shared pointer 集，并匹配 2-tile 周期

思路：在循环外建立 buffer 0/1 的 A0/A1/B0/B1 fragment pointers 和 copy destinations；用固定 parity 的两阶段 loop body 表达 ping/pong，减少每个展开块的 `S2R`、LOP3/IMAD shared 地址重建。随后分别测试自然编译、2-way unroll 和禁止额外 4-way unroll。

- 直接目标：循环内 524,288 条 system/constant 指令以及 11.49M integer/address/move 热点中的 shared-address 部分。
- 预期收益：潜力中到高，但编译器是否保留两套指针需以 SASS 验证。
- 正确性风险：中；容易在 phase 切换、单 tile 和尾 tile 上产生 buffer ownership 错误。
- 实现复杂度：中到高。

不建议仅加入 `#pragma unroll 1` 后直接宣布优化：它可能缩小静态代码，却增加 loop-control 指令或削弱 copy/HMMA 调度。unroll policy 应作为候选 3 的受控变量，最终只按动态 Source Counters 和 Event median 判断。

## 10. 证据局限与最终判断

- 三份 NCU 报告是相同配置但分开的 profiler process；动态指令计数对固定 shape 是确定性的，duration 仍可能受 replay 和运行环境影响。
- mnemonic 分类无法完全区分同一 `IMAD/LEA/MOV` 是 global pointer、shared pointer还是 WMMA backend address preparation；只有第 6.3 节的连续 10-instruction prefetch chain可以做严格的语义计数。
- NCU replay duration 不等于 CUDA Event latency；性能结论继续以 `V4_2_VALIDATION_REPORT.md` 的 Event 中位数为准。
- 候选优化可能改变寄存器分配、unroll 和调度，不能把“预计删除的指令”线性换算为预计加速比。

最终判断：V4_2 的主要未解决问题不是 async copy 或 Tensor 工作量，而是 double-buffer 控制结构在当前编译结果中生成了大量 wide address、shared pointer rematerialization 和 uniform branch 指令。最稳妥的下一步是先拆出最后一 tile，得到无条件 steady loop；随后分别验证 global pointer induction 和两套 shared pointer hoisting。只有新 SASS/NCU 证明动态指令下降且 Event median 稳定超过 V3，才应把它视为后续版本的有效优化方向。
