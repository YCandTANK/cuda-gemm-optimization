# Tensor Core V1 Shared Memory Tiling Report

Date: 2026-10-08 (Asia/Shanghai)

## Environment and method

- GPU: NVIDIA GeForce RTX 5070, compute capability 12.0, 12227 MiB
- Driver: 596.49
- CUDA compiler: 13.2 (`nvcc` 13.2.86)
- Nsight Compute: 2026.1.1
- Build: `nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -Xptxas=-v src/tensor_core_gemm.cu -o build/tensor_core_gemm`
- Benchmark: CUDA Events, five independent measurements, ten warmups and 100 timed launches per measurement. Reported comparisons use the median of the five per-launch averages.
- FP32-to-FP16 conversion, both CPU references, H2D, and D2H are outside the timed regions.
- Nsight Compute used `--set full`, one matching launch, and 40 replay passes. Its reported duration is not used as benchmark timing.

## Code changes

- Kept the V0 and V1 kernel computation algorithms unchanged.
- Added compile-time checks for the 32x32x16 block tile and 2x2 warp layout.
- Added a shared V0/V1 launch, correctness, timing, and reporting path instead of duplicating the test framework.
- V1 launches 128 threads with grid `(N / 32, M / 32)` only when M and N are multiples of 32 and K is a multiple of 16.
- V0 keeps its existing support for M, N, and K that are multiples of 16. Unsupported V1 shapes are skipped rather than restricting V0.
- Both kernels are checked against the two existing CPU references. Device output is first filled with NaNs so missing stores also fail correctness.
- Added five independent timings and their median, plus an automatic 64x96x32 non-square mapping test.

No V1 kernel repair was required. Under the stated shape restrictions, the global accesses are in bounds, all four warps reach both block barriers, and the WMMA shared pointers and strides meet the required alignment. `As` has a 32-byte row stride; `Bs` has a 64-byte row stride; the second B warp-column starts at a 32-byte offset.

## Build and correctness

The `sm_120` build succeeds. `ptxas` reports 40 registers/thread and no spills for both kernels; V1 uses 2048 bytes of static shared memory and one barrier resource. The only compiler diagnostic is the pre-existing unused `lane_id` variable in V0.

Reference 1 is informational and contains FP32-to-FP16 quantization error. Reference 2 isolates the kernel arithmetic and is used for pass/fail with `abs_error <= 1e-2 + 1e-2 * abs(reference)`.

| M x N x K | Kernel | Ref. 1 max abs | Ref. 1 max rel | Ref. 2 max abs | Ref. 2 max rel | Result |
|---|---:|---:|---:|---:|---:|---:|
| 64x96x32 | V0 | 5.761981e-4 | 1.952723e+2 | 1.192093e-7 | 5.948920e-6 | Pass |
| 64x96x32 | V1 | 5.761981e-4 | 1.952723e+2 | 1.192093e-7 | 5.948920e-6 | Pass |
| 1024x1024x1024 | V0 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 1024x1024x1024 | V1 | 7.154942e-4 | 1.094607e-3 | 6.079674e-6 | 1.020690e-5 | Pass |
| 2048x2048x2048 | V0 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |
| 2048x2048x2048 | V1 | 9.616613e-4 | 8.727074e-2 | 1.084805e-5 | 3.015384e-4 | Pass |

The large relative error for Reference 1 in the small test comes from results near zero; its absolute error remains small. V0 and V1 produce identical reported maxima in every case.

A 48x80x16 regression case, which is legal for V0 but not V1, also passes V0 correctness; the host skips V1 for that case. This confirms that adding V1 did not narrow V0's multiple-of-16 shape support.

## Benchmark results

| M=N=K | Kernel | Median latency (ms) | Median TFLOPS | V1/V0 speedup |
|---:|---:|---:|---:|---:|
| 1024 | V0 | 0.147621 | 14.547234 | - |
| 1024 | V1 | 0.140112 | 15.326908 | 1.0536x |
| 2048 | V0 | 1.100199 | 15.615242 | - |
| 2048 | V1 | 0.992038 | 17.317751 | 1.1090x |

Independent per-launch averages in milliseconds:

- 1024 V0: 0.147621, 0.145938, 0.152724, 0.146595, 0.148418
- 1024 V1: 0.140740, 0.140112, 0.138385, 0.142146, 0.138385
- 2048 V0: 1.093340, 1.099055, 1.100199, 1.101034, 1.100670
- 2048 V1: 0.988673, 0.995178, 0.989447, 0.996715, 0.992038

These are measured improvements on this GPU and toolchain, not an assumption that shared-memory staging must always be faster.

## PTX and SASS

PTX retains the expected WMMA operations:

- V0: `wmma.load.{a,b}.sync.aligned...global.f16`
- V1: `wmma.load.{a,b}.sync.aligned...shared.f16`
- Both: `wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32`

On `sm_120`, both kernels contain Tensor Core `HMMA.16816.F32` SASS. V1 Shared-to-Fragment loads compile to:

- A: `LDSM.16.M88.4`
- B: `LDSM.16.MT88.4`

They do not appear under the literal SASS mnemonic `LDMATRIX`. V1 Global-to-Shared staging currently uses scalar `LDG.E.U16` and `STS.U16`, with two `BAR.SYNC.DEFER_BLOCKING` operations per K tile. No async-copy instruction is present.

## Nsight Compute, 1024 cubed

| Metric | V0 | V1 |
|---|---:|---:|
| Long Scoreboard | 49.98 | 8.96 |
| MIO Throttle | 26.19 | 1.23 |
| LG Throttle | 19.90 | 0.04 |
| Barrier stall | 0 | 0.98 |
| Tensor pipe cycles active | 22.07% | 23.12% |
| Registers/thread | 40 | 40 |
| Static shared memory/block | 0 B | 2048 B |
| Theoretical occupancy | 100% | 100% |
| Achieved occupancy | 88.57% | 87.94% |
| Active warps/scheduler | 10.65 | 10.72 |
| Eligible warps/scheduler | 0.29 | 1.52 |
| L1/TEX throughput | 98.27% | 62.19% |
| L2 throughput | 56.65% | 29.54% |
| DRAM throughput | 3.84% | 3.72% |
| Average bytes/global-load sector | 16/32 B | 32/32 B |
| Global-load sectors | 16,777,216 | 4,194,304 |
| L1 global-load hit rate | 70.45% | 26.47% |
| L2 load hit rate | 95.45% | 94.19% |

The lower V1 L1 hit rate is not by itself a regression: V0 issues duplicated, half-utilized loads that often hit in cache, whereas V1 removes that duplication and fully utilizes each sector. The sector count and stall metrics are the more relevant evidence here.

V0 has no shared-memory accesses or bank conflicts. For V1, Nsight Compute reports an average 12.1-way conflict across 524,288 shared-load requests, 4,263,857 bank conflicts, and 67.02% of shared-load wavefronts affected. Source correlation attributes the conflicts as follows:

- `LDSM.16.M88.4` for A: 8-way, 1,048,576 excessive wavefronts out of 2,097,152.
- `LDSM.16.MT88.4` for B: 16-way, 3,145,728 excessive wavefronts out of 4,194,304.

The B fragment load is therefore the larger measured bank-conflict source.

## Why V1 is faster here

Measured evidence supporting the speedup:

1. Each staged A or B tile is reused by two warps, halving the useful global input loads relative to four independent V0 warp tiles.
2. Cooperative loads are fully coalesced: 32/32 useful bytes per global-load sector instead of 16/32 for V0.
3. The two effects together reduce measured global-load sectors by 4x.
4. Long Scoreboard drops from 49.98 to 8.96, MIO Throttle from 26.19 to 1.23, and LG Throttle from 19.90 to 0.04. Eligible warps per scheduler rise from 0.29 to 1.52.

Measured costs limiting the speedup:

1. Severe Shared-to-Fragment bank conflicts, especially the B `LDSM.16.MT88.4` load.
2. Two block barriers per K tile and a measured barrier-stall value of 0.98.
3. Scalar Global-to-Shared `LDG.E.U16`/`STS.U16` staging adds instruction and shared-memory traffic.
4. Tensor-pipe activity changes only slightly, from 22.07% to 23.12%, so the staging improvement has not made the Tensor pipe close to saturated.

Together these measurements explain a modest 1.05x gain at 1024 and a larger 1.11x gain at 2048, rather than a speedup proportional to the 4x reduction in global-load sectors.

## Next hypothesis to validate

The highest-value next isolated experiment is to change only the shared-memory layout for the B tile, using padding or a bank-aware swizzle, then verify whether the measured 16-way `LDSM.16.MT88.4` conflict falls without damaging alignment or coalescing. This report does not implement that experiment, because the current scope is V1 only. Double buffering, async copy, warp-tile changes, and handwritten PTX should remain separate later experiments.
