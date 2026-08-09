---
id: contest-gpumode-p3
title: 'GPU Mode NVFP4 Hackathon - Problem 3: Gated Dual GEMM'
source_category: contest-report
architectures:
- sm100
- sm100a
tags:
- nvfp4
- fp4
- block-scale
- gemm
- gated-dual-gemm
kernel_types:
- gated-dual-gemm
- gemm
languages:
- python
url: https://github.com/gpu-mode/reference-kernels/tree/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm
problem_number: 3
description: Exact public task and correctness-reference scope at the challenge-opening
  commit; no unpublished leaderboard or submission details are asserted.
---

# Problem 3: NVFP4 Gated Dual GEMM

## Verified identity

The official NVIDIA rules name this Kernel Challenge 3 and give its entry window as December 20, 2025 through January 16, 2026. The public GPU Mode task at commit `c5b2f7c062d5015f29c3a1043cfd04954397944c` targets NVIDIA B200.

## Exact operation

For each batch index `l`, the reference computes:

```python
gate = scaled_mm(a[:, :, l], b1[:, :, l].T, sfa, sfb1)
up = scaled_mm(a[:, :, l], b2[:, :, l].T, sfa, sfb2)
output[:, :, l] = silu(gate) * up
```

The two products share `a` and its scale tensor `sfa`; `b1` and `b2` have separate scale tensors. The FP32 product results are combined and converted to FP16 by the correctness reference.

## Published tensor contract

| Tensor | Dtype | Logical shape |
| --- | --- | --- |
| `a` | NVFP4 E2M1 | `[M,K,L]`, K-major |
| `b1`, `b2` | NVFP4 E2M1 | `[N,K,L]`, K-major |
| `sfa` | FP8 E4M3FNUZ | `[M,K/16,L]`, K-major |
| `sfb1`, `sfb2` | FP8 E4M3FNUZ | `[N,K/16,L]`, K-major |
| `c` | FP16 | `[M,N,L]` |

The submission tuple also includes reordered copies of all three scale tensors and a preallocated output. There is an upstream dtype-label inconsistency at this commit: `task.yml` and `template.py` call the scales E4M3FNUZ, while `reference.py` constructs `torch.float8_e4m3fn`. This capture preserves that distinction instead of silently choosing one spelling.

`K` must be divisible by 256. The task additionally requires `M` and `N` to be divisible by the selected MMA tile dimensions. The correctness checker uses relative and absolute tolerances of `1e-3`.

## Published benchmark and scoring contract

| M | N | K | L | Theoretical speed-of-light time (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

Ranking uses the geometric mean of benchmark times. The task labels the last column a speed-of-light analysis based on the maximum of FP4 Tensor Core math time and DRAM-memory time for B200 at a 1.5 GHz clock. These are theoretical comparison values, not measured winning latencies.

## Evidence boundary

The public task and reference do not publish a final leaderboard, winning source, launch count, TMEM partition, TMA pipeline, CUTLASS schedule, physical input-load count, or compute-/memory-bound verdict for a submission. No such implementation or result claims are retained in this source capture.

## Primary sources

- [Official challenge rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Pinned public task](https://github.com/gpu-mode/reference-kernels/blob/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm/task.yml)
- [Pinned correctness reference](https://github.com/gpu-mode/reference-kernels/blob/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm/reference.py)
