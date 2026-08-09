---
id: contest-gpumode-p2
title: 'GPU Mode NVFP4 Hackathon - Problem 2: NVFP4 GEMM'
source_category: contest-report
architectures:
- sm100
- sm100a
tags:
- nvfp4
- gemm
- fp4
- block-scale
hardware_features:
- nvfp4
- fp4
- block-scale
kernel_types:
- gemm
languages:
- python
url: https://github.com/gpu-mode/reference-kernels/tree/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm
captured_at: 2026-08-08
---

# Problem 2: NVFP4 GEMM

## Evidence-scoped task record

The official NVIDIA rules identify this as Kernel Challenge 2, running from November 29 through December 19, 2025. It targets NVIDIA B200 and contributes 20% of the four-problem grand-prize score.

At `gpu-mode/reference-kernels` commit `ae67948685dfccf54ae8374dc9402addb7aae4f6`, the public task implements block-scaled matrix multiplication over each `L` slice with packed E2M1 A and B, per-16 logical scales, reordered scale copies, and preallocated FP16 C. The correctness checker uses `rtol=1e-3` and `atol=1e-3`.

The task description abbreviates the input as five tensors, but `task.py`, `template.py`, and `reference.py` expose seven: A, B, logical SFA/SFB, reordered SFA/SFB, and C. The prose/template call the scales E4M3FNUZ, while `reference.py` constructs `torch.float8_e4m3fn`; consumers must follow the actual tensors supplied by the harness.

`K` is divisible by 256. `M` and `N` divisibility depends on the submission's MMA tile. The repository publishes ten correctness cases and three ranking cases.

## Ranking contract

Ranking uses the geometric mean of three benchmark results. The task labels these B200 values a theoretical speed-of-light analysis at 1.5 GHz, using the maximum of FP4 Tensor Core math time and DRAM-memory time:

| M | N | K | L | Theoretical time (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 7168 | 16384 | 1 | 8.994 |
| 128 | 4096 | 7168 | 1 | 2.354 |
| 128 | 7168 | 2048 | 1 | 1.333 |

These rows are not measured contestant results or a cuBLAS comparison.

## Public leaderboard boundary

The public Popcorn API snapshot fetched August 8, 2026 places `gau.nernst`, `s.am._`, and `billcarson` at current ranks 1-3. It places `Simon`, `yue`, and `currybab` at ranks 8-10; their API scores, multiplied by one million, are 10.806750, 10.914084, and 10.930623.

The current top-three timestamps are December 20-21, after the official December 19 cutoff. The endpoint therefore does not by itself establish prize winners or cutoff rankings. It exposes score metadata and submission filenames, not contestant source code, optimization techniques, a cuBLAS baseline, trials, or variance.

## Primary references

- [Official contest rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Pinned task definition](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/task.yml)
- [Pinned task types](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/task.py)
- [Pinned starter template](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/template.py)
- [Pinned correctness reference](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/reference.py)
- [Public Popcorn leaderboard API](https://site--bot--dxfjds728w5v.code.run/submissions/nvfp4_gemm/NVIDIA?limit=12)
