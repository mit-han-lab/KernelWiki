---
id: kernel-nvfp4-gemv
title: NVFP4 Batched GEMV
type: kernel
architectures:
- sm100
- sm100a
tags:
- gemv
- nvfp4
- fp4
- block-scale
- cache-policy
- register-budgeting
- vectorized-loads
confidence: source-reported
reproducibility: concept
kernel_types:
- gemv
- batched-gemv
languages:
- cuda-cpp
- ptx
- cute-dsl
related:
- hw-nvfp4
- kernel-nvfp4-gemm
- pattern-memory-bound
sources:
- contest-gpumode-p1
- blog-yue-nvfp4
- blog-amandeep-nvfp4
performance_claims: []
---

# NVFP4 Batched GEMV

## Verified Scope

GPU Mode's Problem 1 asks for a block-scaled NVFP4 batched matrix-vector product on NVIDIA B200. The official task models its three benchmark cases against the slower of FFMA math and DRAM transfer time and reports DRAM-limited theoretical times. That task-specific model does not make every GEMV implementation or shape bandwidth-bound.

The logical operation has one B row, reused by all M output rows. The reference harness pads B and its scales to 128 rows so it can call `torch._scaled_mm`, then retains only result column zero. Consequently, A values are row-specific while the logical B vector is reused across M; it is incorrect to say every FP4 input value is consumed only once.

## Official Callable Contract

At pinned commit `ae679486`, `custom_kernel` receives seven tensors, not five tensors plus global FP32 scale arguments:

| Tensor | Physical shape | Role |
|---|---|---|
| `a` | `[M, K/2, L]` | Packed E2M1 A; two logical values per byte |
| `b` | `[128, K/2, L]` | Packed E2M1 B, physically padded; logical row 0 is used |
| `sfa` | `[M, K/16, L]` | Logical/reference A block scales |
| `sfb` | `[128, K/16, L]` | Logical/reference B block scales, padded to 128 rows |
| `sfa_reordered` | `[32, 4, ceil(M/128), 4, K/64, L]` | Swizzled A scales for custom kernels |
| `sfb_reordered` | `[32, 4, 1, 4, K/64, L]` | Swizzled padded-B scales for custom kernels |
| `c` | `[M, 1, L]` | FP16 output buffer |

`K` must be divisible by 64, and M must be divisible by the implementation's selected M tile. The task prose labels scales `e4m3fnuz`, but the pinned generator constructs `torch.float8_e4m3fn`; implementations should follow the actual harness revision they are tested against. This contest ABI exposes no per-tensor FP32 global scales.

## Official Benchmark Model

The official table assumes a 1.5 GHz B200 clock and ranks submissions by the geometric mean across all three rows:

| M | K | L | Theoretical time (µs) |
|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 8.622 |
| 4096 | 7168 | 8 | 17.275 |
| 7168 | 2048 | 4 | 4.317 |

NVIDIA's Blackwell technical brief specifies 8 TB/s of HBM3e bandwidth for one GB200 GPU. The task's theoretical numbers are model values, not measured kernel timings.

The public leaderboard is mutable and reports aggregate scores rather than per-shape latencies. In the snapshot fetched on 2026-08-08, the first three rows were `s.am._` at 18.549562452 µs, `gau.nernst` at 18.552844757 µs, and `shellsmile15795` at 18.707609314 µs. Yue's 22.392217755 µs submission was rank 11. The official cutoff was November 28, 2025 at 11:59 p.m. PT; current ranks 2 and 3 have November 30 timestamps, so this endpoint is not an official prize-placement snapshot.

## PTX Semantics and Reported Techniques

PTX ISA 9.0 defines the packed conversion and typed register decomposition used in Yue's author-reported decode path:

```asm
cvt.rn.f16x2.e2m1x2 %result, %packed_fp4_pair;
mov.b32 {%b0, %b1, %b2, %b3}, %packed_word;
```

It also defines vector loads such as `ld.global.v2.u64` and `ld.global.v4.u64`, which move 16 and 32 bytes and can carry 32 and 64 packed FP4 values. The ISA specifies behavior, not that one width or decomposition is universally faster.

Amandeep reports that three solutions inspected after the event used `L1::no_allocate` for streamed A loads, `L1::evict_last` for reused B loads, wider PTX loads, and exact-K specializations. Those are author-reported observations; the public leaderboard API supplies no contestant code or technique field. Amandeep's own wider `uint2` experiment was 16–25% slower, and reducing `-maxrregcount` from 80 to 64 had no effect because the kernel already used fewer than 64 registers. Load width, cache hints, and register caps therefore require measurement in the exact implementation.

## Yue's Author-Reported Progression

The following timings are from Yue's post, not an independent reproduction. Several stages combine multiple changes and cannot establish isolated causality:

| Stage | Combined change | Reported latency |
|---|---|---:|
| Initial CuTe DSL | First working CuTe implementation | ~100 µs |
| Optimized CuTe DSL | Scale-load and arithmetic changes plus thread collaboration | ~33 µs |
| Initial CUDA | Naive hand-written path | ~2000 µs |
| CUDA step 1 | Coalescing, shared B, thread collaboration, warp reduction | ~443 µs |
| CUDA step 2 | Remove shared B, per-thread tiles, `float4` loads, hardware intrinsics | ~39 µs |
| CUDA step 3 | Vectorized PTX FP4/scale decode | ~27 µs |
| Parameter tuning | Threads per row and rows per block | ~26 µs |
| Two-tile ILP | Interleave two tiles per loop | ~22.9 µs |
| Aggressive PTX fusion | Fuse decode, scale, multiply, and accumulation | ~22.3 µs |
| Submitted score | Geometric mean shown by the leaderboard | 22.392 µs |

Yue also reports that loading the entire B vector into shared memory did not improve the CuTe attempt. This is a useful counterexample to treating shared-memory B staging or an exact `BLOCK_M` traffic reduction as automatic.

## Practical Use and Caveats

- Use this case study for an N=1 decode-style matrix-vector operation only when payloads, scales, layouts, output type, and target match the chosen NVFP4 implementation contract.
- Profile bytes moved, instruction count, decode and reduction work, cache behavior, spills, and occupancy before deciding which resource is limiting performance.
- Exact-K dispatch can enable full unrolling and size-specific tuning, but retaining multiple specializations adds compiled kernel entries.
- Inline PTX must be revalidated against the selected PTX ISA, toolkit, and target. The contest ran on B200; that does not imply packed FP4 conversion is restricted to SM100/SM100a forever. Transformer Engine 2.13 lists NVFP4 inference support for SM 10.0 and later.
- The official task accepts any implementation that satisfies its checker. It does not require the cache policies, load widths, register limits, or code structures described in participant posts.

## Primary Sources

- [Pinned Problem 1 task](https://github.com/gpu-mode/reference-kernels/tree/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemv)
- [Official contest rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief)
- [PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html)
- [Transformer Engine 2.13 NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [Yue Zhang's hackathon journey](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [Amandeep Singh's twelve attempts](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [Simon Veitner's reference](https://veitner.bearblog.dev/nvfp4-gemv/) and [improved variants](https://veitner.bearblog.dev/nvfp4-gemv-improved/)
