---
id: contest-gpumode-p3
title: "GPU Mode NVFP4 Hackathon - Problem 3: Gated Dual GEMM"
source_category: contest-report
architectures: [sm100]
tags: [nvfp4, gemm, gated-dual-gemm, fp4, block-scale]
kernel_types: [gated-dual-gemm, gemm]
url: https://www.gpumode.com/api/leaderboard/598
captured_at: 2026-08-18
---

# Problem 3: NVFP4 Dual GEMM with SiLU

The organizer's API defines two block-scaled NVFP4 matrix multiplications followed by SiLU activation, optimized for NVIDIA B200. Rankings use the geometric mean across four configurations.

## Benchmark configurations and organizer speed-of-light values

| M | N | K | L | Speed of light (microseconds) |
|---:|---:|---:|---:|---:|
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

## Captured leaderboard state

On 2026-08-18, the API's NVIDIA ranking listed `gau.nernst` first with a score of `1.312365976846222e-05` seconds, or 13.12365976846222 microseconds. This is a live API snapshot, not a claim about the leaderboard at the original deadline.

Source locator: [GPU Mode leaderboard API 598](https://www.gpumode.com/api/leaderboard/598), `data.description`, `data.benchmarks`, and `data.rankings.NVIDIA[0]`.
