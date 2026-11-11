# WMMA V4 Async Pipeline 实验总结

本文整理 FP16 WMMA GEMM 从 V3 的同步 128-bit staging，演进到 V4.1–V4.3 async-copy / double-buffering 的实验结果。完整的逐轮时间、SASS 片段和 Nsight Compute 表格保存在 [`archive/wmma/`](archive/wmma/README.md)。

## 结论

- V4.1、V4.2、V4.3 均通过已有正确性检查，且编译器确实生成 `cp.async` 对应的 `LDGSTS`、`LDGDEPBAR` 和 `DEPBAR`，不是只根据源码 intrinsic 推断异步路径成立。
- V4.1 使用单 buffer，commit 后立即 wait，没有跨 K tile 的 copy/compute overlap；在较稳定的 `2048³` 测量中比 V3 慢约 5.12%。
- V4.2 使用 ping-pong shared buffer。SASS 中下一 tile 的 `LDGSTS` 位于当前 tile 的 LDSM/HMMA 与 wait 之前，形成真实的重叠机会；但额外地址和控制指令抵消了收益，`2048³` 中位延迟仍比 V3 高约 0.45%。
- V4.3 将最后一 tile 从 steady loop 拆出，使 executed warp instructions 相对 V4.2 下降 37.31%，寄存器从 78 降到 72/thread。两批 Event 结果没有一致证明它稳定快于 V3 或 V4.2。
- 因此 V3 继续作为当前有稳定性能证据支持的基线；V4.3 则是结构更干净、适合继续研究的双缓冲基线。V4 系列的价值在于验证 async pipeline 及其编译开销，不应被描述为已经稳定超越 V3。

## 版本演进

| 版本 | 相对前一版本的单一目标 | 结果 |
|---|---|---|
| V3 | 128-bit Global→Shared staging | `LDG.E.128` + `STS.128`，稳定性能基线 |
| V4.1 | 用 async copy 替换同步 staging | 生成 `LDGSTS`，但 single buffer 立即等待 |
| V4.2 | 两套 shared plane 做 ping-pong double buffering | wait 前存在当前 tile 的 LDSM/HMMA，barrier 动态次数减半 |
| V4.3 | 将最后一 tile 移到 loop 外，简化 steady state | 大幅减少条件、地址重建和 system-register 读取 |

V4 各版本保持 V3 的 64×64×16 CTA tile、128 threads/CTA、WMMA fragment 布局和 C 写回。V4.2/V4.3 的 static shared memory 为 8704 B/block；V4.1 仍为单 plane。

## 正确性与编译证据

V4.1–V4.3 都使用 FP16 量化后的输入计算 FP32 CPU reference，并在集成 runner 中与相邻版本逐元素对照。覆盖包括：

- `K=16/32/48`，分别验证单 tile、双 tile和 steady/epilogue 边界；
- `64×64×16`、`128×192×64`、`1024³` 和 `2048³`；
- 输出 NaN 预填充、finite 检查和混合绝对/相对容差。

所有已测用例通过，V4 与 V3 的输出在这些测试中逐元素一致。逐位一致是本次观测，不作为浮点 GEMM 的一般保证。

编译产物提供了以下证据：

```text
V3:    LDG.E.128 -> STS.128 -> BAR -> LDSM/HMMA
V4.1:  LDGSTS -> LDGDEPBAR/DEPBAR -> BAR -> LDSM/HMMA
V4.2+: LDGSTS(next) -> LDSM/HMMA(current) -> DEPBAR -> BAR
```

这能确认 V4.2/V4.3 的静态指令顺序提供了 overlap 机会，但不能单独量化硬件隐藏了多少延迟；端到端结论仍以 CUDA Event 为准。

## 性能结果

所有表格均来自默认 stream、同一进程共享输入与 allocations、每轮 10 次 warmup、100 次 launch 取平均、5 轮 rotating-order 测量后取中位数。不同批次不混合计算中位数。

### V4.1 与 V4.2

| Shape / 实现 | Median 延迟 | Median 吞吐 | 解释 |
|---|---:|---:|---|
| `2048³` V3 | 0.301219 ms | 57.035 TFLOPS | V4.1 同轮基线 |
| `2048³` V4.1 | 0.316649 ms | 54.255 TFLOPS | 比 V3 慢约 5.12% |
| `2048³` V3 | 0.303914 ms | 56.529 TFLOPS | V4.2 正式测量同轮基线 |
| `2048³` V4.2 | 0.305285 ms | 56.275 TFLOPS | 比 V3 慢约 0.45%，比 V4.1 快 1.0383× |

V4.2 恢复了 V4.1 的大部分回退，说明双缓冲的等待安排有效；但它没有形成稳定超过 V3 的端到端收益。`1024³` kernel 很短且轮次离散更明显，两次独立测量中 V4.2 均未超过 V3。

### V4.3 两批复验

| Shape | 批次 | V3 | V4.2 | V4.3 | V4.3 / V4.2 |
|---|---:|---:|---:|---:|---:|
| `1024³` | 1 | 0.045389 ms | 0.050579 ms | 0.047857 ms | 1.0569× |
| `1024³` | 2 | 0.052324 ms | 0.048386 ms | 0.047570 ms | 1.0172× |
| `2048³` | 1 | 0.312167 ms | 0.307365 ms | 0.301270 ms | 1.0202× |
| `2048³` | 2 | 0.306640 ms | 0.308478 ms | 0.309066 ms | 0.9981× |

V4.3 在两个 `1024³` 批次均快于 V4.2，但相对 V3 的方向不一致；两个 `2048³` 批次对 V3/V4.2 的方向也不一致。单批次中出现过 V4.3 超过同轮 cuBLAS，但另一批没有复现，因此不能把它作为稳定结论。

## Profiler 与动态指令

`2048³`、相同 Nsight Compute section 配置下的核心结果为：

| Metric | V3 | V4.1 | V4.2 | V4.3 |
|---|---:|---:|---:|---:|
| Registers/thread | 78 | 72 | 78 | 72 |
| Static shared/block | 4.35 KB | 4.35 KB | 8.70 KB | 8.70 KB |
| Theoretical occupancy | 50.00% | 58.33% | 50.00% | 58.33% |
| Executed warp instructions | 14,270,464 | 15,839,232 | 24,944,640 | 15,638,528 |
| `LDGSTS` instructions | 0 | 1,048,576 | 1,048,576 | 1,048,576 |
| CTA barrier warp instructions | 1,048,576 | 1,048,576 | 524,288 | 524,288 |
| Tensor Pipe Active | 87.316% | 82.768% | 88.065% | 89.697% |
| NCU replay duration | 325.60 µs | 340.13 µs | 328.42 µs | 317.41 µs |

NCU replay duration 只用于 profiler 内部方向比较，不替代 CUDA Event。stall 指标是 cycles/issued；V4.3 删除大量可发射的整数指令后会改变分母，所以不能把 Long Scoreboard 等数值直接当作耗时百分比。

### V4.2 的指令膨胀

V4.2 相对 V4.1 多执行 9,105,408 条 warp instructions。按 `2048³` 的 524,288 个 warp-tile iterations 归一化，热循环由约 29.00 增至 46.36 instructions/tile/warp，增量主要来自：

- integer、地址和 move：约 +14.42；
- branch 与 predicate：+3.00；
- system/constant 读取：+1.00；
- CTA barrier：约 -1.01。

`LDGSTS`、LDSM 和 HMMA 数量没有增加。也就是说，双缓冲确实改善了等待和 barrier，但 backend 为 buffer 选择、边界条件及地址重建生成的额外工作占用了 issue slots。

V4.3 把最后一 tile 单独处理后，branch/predicate 下降 83.70%，integer/address/move 下降 61.17%，热循环中的重复 `S2R LANEID/TID/CTAID` 消失，编译器也能把多数 global 地址计算转成 pointer induction。这解释了总指令相对 V4.2 下降 37.31%。

## 尚未解决的问题

- V4 async path 的 B-side global sector 数仍比 V3 多；V4.2/V4.3 没有改变这一点。
- Shared→fragment 的 A-side LDSM 仍有 8 actual / 4 ideal wavefronts；V4 系列没有修改 shared layout。
- Tensor pipe 已接近 90% active，继续增加 buffer 未必能抵消额外控制、shared memory 和寄存器代价。
- 当前缺少计时期间连续频率、温度和系统负载遥测，短 kernel 的批次波动不能归因于单一机制。

下一步如果继续，应以 V4.3 为基础，只做一个受控变量：针对仍保留的 B-side 64-bit 地址链测试显式 pointer induction，并同时检查 SASS、动态指令、寄存器、spill 与多批 Event 中位数。除非稳定超过 V3，否则不应把实验版替换为性能基线。

## 原始报告

- [V4.1 集成、验证与性能分析](archive/wmma/V4_1_VALIDATION_REPORT.md)
- [V4.2 Double Buffering 验证报告](archive/wmma/V4_2_VALIDATION_REPORT.md)
- [V4.2 动态指令专项诊断](archive/wmma/V4_2_INSTRUCTION_ANALYSIS.md)
- [V4.3 集成、验证与性能分析](archive/wmma/V4_3_VALIDATION_REPORT.md)
