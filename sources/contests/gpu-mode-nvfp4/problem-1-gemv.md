---
id: contest-gpumode-p1
title: 'GPU Mode NVFP4 Hackathon - Problem 1: Batched GEMV'
source_category: contest-report
architectures:
- sm100
- sm100a
tags:
- nvfp4
- gemv
- fp4
- block-scale
hardware_features:
- nvfp4
- fp4
- block-scale
kernel_types:
- batched-gemv
- gemv
languages:
- python
- cute-dsl
url: https://github.com/gpu-mode/reference-kernels/tree/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemv
retrieved_at: 2026-08-08
---

# Problem 1: NVFP4 Batched GEMV

## Evidence Scope

This card records the official task at `gpu-mode/reference-kernels` commit `ae67948685dfccf54ae8374dc9402addb7aae4f6`, the official contest rules, and a dated public leaderboard response. It does not attribute implementations to contestants because the public API exposes no source or technique field.

## Task Contract

The operation is an N=1 block-scaled NVFP4 matrix-vector product on B200 with FP16 output. Although `task.yml` describes a logical five-tensor tuple, the pinned Python callable receives seven physical tensors:

| Tensor | Physical shape |
|---|---|
| A payload | `[M, K/2, L]` |
| B payload | `[128, K/2, L]` |
| logical SFA | `[M, K/16, L]` |
| logical SFB | `[128, K/16, L]` |
| reordered SFA | `[32, 4, ceil(M/128), 4, K/64, L]` |
| reordered SFB | `[32, 4, 1, 4, K/64, L]` |
| C | `[M, 1, L]` |

B is physically padded to 128 rows to support the `torch._scaled_mm` reference, but only logical result column zero is copied to C. The generator constructs E4M3FN scale tensors even though the task prose says E4M3FNUZ. There are no FP32 global-scale arguments in this task ABI. Correctness uses `rtol=1e-3` and `atol=1e-3`; K must be divisible by 64, and M must satisfy the submitted kernel's M-tile divisibility rule.

## Benchmarks and Ranking

The task publishes these theoretical times under a 1.5 GHz B200 model based on the slower of FFMA math and DRAM transfer time:

| M | K | L | Theoretical time (µs) |
|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 8.622 |
| 4096 | 7168 | 8 | 17.275 |
| 7168 | 2048 | 4 | 4.317 |

Submissions are ranked by the geometric mean of the three benchmark results. The pinned task also contains ten correctness cases.

## Timeline and Public Snapshot

The official rules run Kernel 1 from November 10, 2025 at 12:00 a.m. PT through November 28, 2025 at 11:59 p.m. PT and assign it 10% of the four-kernel grand-prize score.

The mutable public endpoint fetched on 2026-08-08 showed `s.am._`, `gau.nernst`, and `shellsmile15795` in its first three rows with aggregate scores of 18.549562452, 18.552844757, and 18.707609314 microseconds. Yue's 22.392217755 score was rank 11 and Simon's 25.112153955 score was rank 25. Current ranks 2 and 3 have post-cutoff timestamps, so this response must not be presented as an official prize-placement snapshot.

## Primary Sources

- [Pinned task directory](https://github.com/gpu-mode/reference-kernels/tree/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemv)
- [Official terms and conditions](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Public leaderboard endpoint](https://site--bot--dxfjds728w5v.code.run/submissions/nvfp4_gemv/NVIDIA?limit=100)
