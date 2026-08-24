---
id: contest-gpumode-p2
title: "GPU Mode NVFP4 Hackathon - Problem 2: NVFP4 GEMM"
source_category: contest-report
architectures: [sm100]
tags: [nvfp4, gemm, fp4, block-scale]
kernel_types: [gemm]
url: https://www.gpumode.com/api/leaderboard/597
captured_at: 2026-08-18
---

# Problem 2: NVFP4 GEMM

The organizer's API defines a block-scaled matrix-matrix multiplication for NVIDIA B200. Its task prose labels the per-16 scale `fp8(e4m3fnuz)`, while its executable reference documents and constructs the scale tensors as `torch.float8_e4m3fn`. The organizer artifacts agree that inputs use NVFP4 E2M1 values with one FP8 scale per 16 values and FP16 output, but disagree on the signed FP8 encoding name. Rankings use the geometric mean across three configurations.

## Benchmark configurations and organizer speed-of-light values

| M | N | K | L | Speed of light (microseconds) |
|---:|---:|---:|---:|---:|
| 128 | 7168 | 16384 | 1 | 8.994 |
| 128 | 4096 | 7168 | 1 | 2.354 |
| 128 | 7168 | 2048 | 1 | 1.333 |

## Captured leaderboard state

On 2026-08-18, the API's NVIDIA ranking listed `gau.nernst` first with a score of `9.981888843481874e-06` seconds, or 9.981888843481874 microseconds. This is a live API snapshot, not a claim about the leaderboard at the original deadline.

Source locators: [GPU Mode leaderboard API 597](https://www.gpumode.com/api/leaderboard/597), `data.description`, `data.benchmarks`, and `data.rankings.NVIDIA[0]`; [organizer executable reference](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemm/reference.py), `generate_input` scale-factor dtype and construction.
