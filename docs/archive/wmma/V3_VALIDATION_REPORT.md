# Tensor Core WMMA V3 接入与验证报告

日期：2026-10-14
平台：NVIDIA GeForce RTX 5070，CUDA `sm_120`

## 结论

- 仅补齐了 V3 常量/编译期约束、修正 `B_SHARED_STRIDE_V3_1` 的三处未定义引用，并将 V3 接入现有正确性与交替 Benchmark 框架；没有改写 V3 的计算、同步或四 accumulator 结构。
- 原始 V2.1 Kernel `wmmaGemmSharedV2_1` 未修改。
- V3 在正确性测试及 1024³、2048³ 两组正式测试中均通过；本次确定性输入下，V3 与 V2.1/cuBLAS 的最大绝对和相对差异均为 0。
- V3 中位延迟为 0.045612 ms（1024³）和 0.307923 ms（2048³），分别达到同轮 cuBLAS 吞吐的 90.28% 和 94.16%，相对 V2.1 加速 2.5905x 和 2.1840x。
- SASS 确认 Global→register 使用 `LDG.E.128`、register→Shared 使用 `STS.128`。V3 使用 78 registers/thread、4352 B 静态 Shared Memory，无 stack frame、无 spill。

## 源码修改范围

文件：`src/tensor_core_gemm.cu`

1. 补齐 V3 的 64×64×16 block tile、32×32 warp tile、2×2 warp layout、128-bit（8 个 half）向量宽度及 `B_SHARED_STRIDE_V3 = 72`。
2. 增加编译期检查：tile/warp 对应关系、WMMA K、向量整除、每线程恰好搬运一个 A/B 向量，以及 B Shared stride 的大小和对齐。
3. 将 Kernel 中三处错误的 `B_SHARED_STRIDE_V3_1` 改为 `B_SHARED_STRIDE_V3`。
4. 在 `KernelVersion`、`kernelName`、`launchKernel`、`runCase` 中接入 V3；增加 V3 正确性、V2.1↔V3/cuBLAS 比较、最终 pass/fail，以及 V2.1/V3/cuBLAS 三者五轮轮转交替计时。

未改变 V3 的 64×64×16 tile、72 half 的 B stride、四个 accumulator、两处 `__syncthreads()`，也未改变 V2.1 Kernel。

## 编译与正确性

编译命令：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
  -Xptxas=-v --ptxas-options=-warn-spills \
  src/tensor_core_gemm.cu -o build/tensor_core_gemm_v3_validation -lcublas
```

编译成功。V3 的 ptxas 资源信息：78 registers/thread、4352 B static shared、0 B stack frame、0 B spill stores、0 B spill loads。唯一警告是既有的未使用变量 `lane_id`，与 V3 接入无关。

先执行小规模正确性测试：

```bash
./build/tensor_core_gemm_v3_validation 128 192 64
```

结果：V3 对 FP16 量化参考和 cuBLAS 均通过；V2.1↔V3、V3↔cuBLAS 最大绝对/相对差异均为 0。程序退出码为 0。附带的 64×96×32 case 因 N 不是 64 的倍数按设计跳过 V2/V2.1/V3。

## 非 NCU 交替 Benchmark

协议沿用现有 runner：每个实现预热 10 次，每个采样连续执行 100 次并取单次平均值，共 5 轮；三种实现每轮轮转启动顺序，最终取五轮中位数。同一 case 复用同一输入、设备分配和默认 stream。以下结果来自普通进程，不来自 NCU replay。

命令：

```bash
./build/tensor_core_gemm_v3_validation 1024 1024 1024
./build/tensor_core_gemm_v3_validation 2048 2048 2048
```

### 1024³

| 实现 | 五轮原始时间（ms） | 中位数（ms） | 中位吞吐（TFLOP/s） |
|---|---|---:|---:|
| V2.1 | 0.118276, 0.121038, 0.114885, 0.114954, 0.118157 | 0.118157 | 18.174814 |
| V3 | 0.045739, 0.045612, 0.047489, 0.045610, 0.045442 | 0.045612 | 47.081058 |
| cuBLAS | 0.040396, 0.040413, 0.041179, 0.051612, 0.041413 | 0.041179 | 52.149719 |

V3/V2.1 speedup = 2.5905x；V3/cuBLAS throughput = 90.28%。cuBLAS 第四轮有明显高延迟，但中位数不受单个异常值支配，原始值如实保留。

### 2048³

| 实现 | 五轮原始时间（ms） | 中位数（ms） | 中位吞吐（TFLOP/s） |
|---|---|---:|---:|
| V2.1 | 0.668515, 0.672493, 0.668491, 0.676734, 0.675319 | 0.672493 | 25.546534 |
| V3 | 0.303127, 0.297736, 0.309426, 0.307923, 0.310699 | 0.307923 | 55.792768 |
| cuBLAS | 0.280985, 0.281015, 0.290786, 0.294990, 0.289940 | 0.289940 | 59.253218 |

V3/V2.1 speedup = 2.1840x；V3/cuBLAS throughput = 94.16%。

## 独立 SASS 与资源检查

导出命令：

```bash
cuobjdump --dump-sass build/tensor_core_gemm_v3_validation \
  | awk '/Function : _Z10wmmaGemmV3/{keep=1} keep && /Function :/ && !/_Z10wmmaGemmV3/ {exit} keep' \
  > build/tensor_core_gemm_v3.sass
cuobjdump --dump-resource-usage build/tensor_core_gemm_v3_validation
```

独立文件 `build/tensor_core_gemm_v3.sass` 只包含 `wmmaGemmV3`。观察到：

- Global load：`LDG.E.128`；Shared store：`STS.128`，证明 `uint4` 搬运没有退化成窄指令。
- Shared→fragment 使用 `LDSM`，Tensor 运算使用 `HMMA`。
- ptxas 报告 V3 为 78 registers/thread、4352 B static shared、0 spill。`cuobjdump` 的 5376 B shared 包含额外的 1024 B driver-reserved shared memory。
- 静态 SASS 中宽指令、LDSM/HMMA 的文本出现次数受编译器展开和调度影响，不能直接当作动态次数或主要瓶颈证据。

## NCU 定向复核（1024³）

V3 新报告：`build/ncu_v3_1024.ncu-rep`。命令：

```bash
ncu --set full --kernel-name-base demangled \
  --kernel-name 'regex:wmmaGemmV3' --launch-count 1 \
  -o build/ncu_v3_1024 \
  ./build/tensor_core_gemm_v3_validation 1024 1024 1024
```

下表将新 V3 报告与已有 `build/ncu_v2_1.ncu-rep` 的同规模结果并列。二者不是同一次 NCU 会话，duration 仅作方向性参考；正式性能结论以上面的非 NCU 同轮交替测试为准。

| 指标 | V2.1 | V3 | 直接观察 |
|---|---:|---:|---|
| Executed instructions | 19,923,968 | 1,863,680 | V3 减少 90.65% |
| Global load instructions/requests | 1,048,576 | 131,072 | 减少 8x |
| Global load sectors | 2,097,152 | 2,097,152 | 搬运字节量未减少 |
| Shared store instructions | 1,048,576 | 131,072 | 减少 8x |
| LDSM instructions | 262,144 | 262,144 | Shared→fragment 工作量不变 |
| Local load/store | 0 / 0 | 0 / 0 | 无 local-memory spill 流量 |
| Registers/thread | 56 | 78 | V3 增加 22 个寄存器 |
| Theoretical occupancy | 75% | 50% | 受寄存器用量影响下降 |
| Achieved occupancy | 44.55% | 38.09% | 实测下降 6.46 pp |
| Tensor pipe active | 17.88% | 62.93% | Tensor Pipe 利用率明显提高 |
| Barrier stall/issued | 0.69 | 5.97 | 比例上升，不能脱离分母解释 |
| Long-scoreboard stall/issued | 18.38 | 21.09 | 比例小幅上升，不能据此断言更慢 |

直接观测事实：V3 用 128-bit 指令将动态 Global load 和 Shared store 指令数各降至 V2.1 的 1/8，同时传输 sectors 不变；LDSM/HMMA 的计算工作没有被删减；整体动态指令数显著下降，Tensor pipe active 大幅提高；代价是每线程寄存器增加、occupancy 降低。NCU source view 还显示 `LDG.E.128` 和 `STS.128` 的 actual/ideal sectors 相符，Shared store 没有 conflict；A 路 LDSM 仍存在 Shared load conflict/excessive wavefront 现象。

合理解释：当前加速主要与搬运指令合并、动态指令开销下降以及 Tensor Pipe 被更充分供给一致。V3 的 barrier/scoreboard stall-per-issued 比例没有同步下降，部分原因是 issued instruction 分母大幅缩小；这些比例不能单独证明 barrier 或 scoreboard 成为新的主瓶颈。

尚未证实：寄存器/occupancy 是否已限制 2048³ 性能，以及剩余 A 路 LDSM conflict 对端到端性能的独立贡献。本轮不为回答这些问题改写核心 Kernel，也没有继续做无对照实验支撑的重复 profiling。

## 工作区说明

- 未覆盖 `FINAL_GEMM_PERFORMANCE_REPORT.md`、`V2_PERFORMANCE_REPORT.md`、`V3_PREOPT_PERFORMANCE_DIAGNOSIS.md` 或其他历史报告。
- 未清理任何 untracked Markdown/文档文件，未修改既有 `README.md` 变更，未执行 Git commit。
- 新增验证产物为 `V3_VALIDATION_REPORT.md`、`build/tensor_core_gemm_v3.sass` 和 `build/ncu_v3_1024.ncu-rep`；构建目录产物受现有忽略规则管理。
