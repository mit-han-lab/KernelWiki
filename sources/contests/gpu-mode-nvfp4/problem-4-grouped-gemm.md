---
id: contest-gpumode-p4
title: 'GPU Mode NVFP4 Hackathon - Problem 4: Grouped GEMM'
source_category: contest-report
architectures:
- sm100
- sm100a
tags:
- nvfp4
- grouped-gemm
- fp4
- block-scale
- moe
kernel_types:
- grouped-gemm
- gemm
- moe
languages:
- python
url: https://github.com/gpu-mode/reference-kernels/tree/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm
problem_number: 4
description: Exact public task and correctness-reference scope after the published
  K-divisibility correction; no unpublished legitimate leaderboard is asserted.
---

# Problem 4: NVFP4 Grouped GEMM

## Verified identity

The official NVIDIA rules name this Kernel Challenge 4, give its entry window as January 17 through February 13, 2026, and assign it 40% of the four-problem grand-prize score. The public GPU Mode task at commit `ae67948685dfccf54ae8374dc9402addb7aae4f6` targets NVIDIA B200.

## Exact group contract

The operation is a list of independent block-scaled products. Group `i` computes `C_i = A_i @ B_i.T`. Unlike an M-grouped MoE-only interface, the official correctness cases can vary `M_i`, `N_i`, and `K_i` within the same list.

| Per-group value | Published dtype | Logical shape |
| --- | --- | --- |
| `a_i` | packed NVFP4 E2M1, two values per byte | `[M_i,K_i/2,L_i]` |
| `b_i` | packed NVFP4 E2M1, two values per byte | `[N_i,K_i/2,L_i]` |
| `c_i` | FP16 | `[M_i,N_i,L_i]` |
| `sfa_i` | FP8 E4M3FNUZ in task/template | `[M_i,K_i/16,L_i]` |
| `sfb_i` | FP8 E4M3FNUZ in task/template | `[N_i,K_i/16,L_i]` |
| size | integers | `(M_i,N_i,K_i,L_i)` |

The actual Python tuple contains four lists: `(abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)`. `task.yml` initially describes only three names, while `template.py`, `task.py`, and `reference.py` expose the reordered-scale list as the fourth value. The first two lists contain logical scales; the third contains layout-reordered scale copies intended for the custom kernel.

At this revision, the prose/template label scales `float8_e4m3fnuz`, but the generator constructs `torch.float8_e4m3fn`. This source capture preserves that upstream inconsistency. Each published case uses `L=1`; `K_i` is divisible by 256, and each `M_i`/`N_i` must satisfy the selected MMA tile divisibility.

The correctness reference independently invokes `torch._scaled_mm` for each group with `B_i` transposed, writes FP16 output, and checks with `rtol=1e-3` and `atol=1e-3`.

## Published benchmark and scoring contract

| Groups | M values | N | K | L | Theoretical speed-of-light time (µs) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 80, 176, 128, 72, 64, 248, 96, 160 | 4096 | 7168 | 1 | 18.833 |
| 8 | 40, 76, 168, 72, 164, 148, 196, 160 | 7168 | 2048 | 1 | 10.667 |
| 2 | 192, 320 | 3072 | 4096 | 1 | 2.406 |
| 2 | 128, 384 | 4096 | 1536 | 1 | 1.525 |

Ranking uses the geometric mean of benchmark results. The task labels the final column a speed-of-light analysis based on the maximum of B200 FP4 Tensor Core math time and DRAM-memory time at a 1.5 GHz clock. These are theoretical comparison values, not contestant measurements.

## Reward-hack boundary

GPU Mode's official postmortem records a submission that temporarily reported `11.191 µs`, roughly `2 µs` ahead of the next entry, before being scrubbed. During correctness, it ran a real padded 8-group kernel on each of 15 cloned objects. During timing, one call launched a merged 120-group kernel for all 15 objects and calls 2 through 15 returned cached output pointers; the harness divided the combined work by 15. This number is therefore an invalid amortized exploit result, not a legitimate performance record.

The postmortem points to `gpu-mode/reference-kernels` PR #104 as the evaluation-harness response. It does not establish a FlashInfer-Bench or MLSys 2026 causal link.

## Evidence boundary

The task fixes observable tensors, correctness, benchmark workloads, theoretical estimates, timeout, and scoring. It does not require one GPU launch, CUTLASS, CLC, TMA, TMEM, a persistent scheduler, or a particular bottleneck classification. No legitimate final ranks, winner source, or implementation techniques are published in the pinned task, so none are asserted here.

## Primary sources

- [Official challenge rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Pinned public task](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm/task.yml)
- [Pinned starter template](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm/template.py)
- [Pinned correctness reference](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm/reference.py)
- [Official reward-hack postmortem](https://www.gpumode.com/news/reward-hacking-nvfp4)
