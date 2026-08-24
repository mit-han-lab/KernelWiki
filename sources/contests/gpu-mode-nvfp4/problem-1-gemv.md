---
id: contest-gpumode-p1
title: "GPU Mode NVFP4 Hackathon - Problem 1: Batched GEMV"
source_category: contest-report
architectures: [sm100]
tags: [nvfp4, gemv, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
url: https://www.gpumode.com/api/leaderboard/595
captured_at: 2026-08-18
---

# Problem 1: NVFP4 Batched GEMV

The organizer's API defines a batched matrix-vector multiplication for NVIDIA B200. Its task prose labels the per-16 scale `fp8(e4m3fnuz)`, while its executable reference documents and constructs the scale tensors as `torch.float8_e4m3fn`. The organizer artifacts agree that inputs use NVFP4 E2M1 values with one FP8 scale per 16 values and FP16 output, but disagree on the signed FP8 encoding name. Rankings use the geometric mean across three benchmark configurations.

## Benchmark configurations and organizer speed-of-light values

| M | K | L | Speed of light (microseconds) |
|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 8.622 |
| 4096 | 7168 | 8 | 17.275 |
| 7168 | 2048 | 4 | 4.317 |

The API says this analysis uses the maximum of FFMA throughput and DRAM throughput on B200 at a 1.5-GHz clock.

## Captured leaderboard state

On 2026-08-18, the API's NVIDIA ranking listed `s.am._` first with a score of `1.854956245154537e-05` seconds, or 18.54956245154537 microseconds. This is a live API snapshot, not a claim about the leaderboard at the original deadline.

Source locators: [GPU Mode leaderboard API 595](https://www.gpumode.com/api/leaderboard/595), `data.description`, `data.benchmarks`, and `data.rankings.NVIDIA[0]`; [organizer executable reference](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemv/reference.py), `generate_input` scale-factor dtype and construction.
