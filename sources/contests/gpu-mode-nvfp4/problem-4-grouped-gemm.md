---
id: contest-gpumode-p4
title: "GPU Mode NVFP4 Hackathon - Problem 4: Grouped GEMM"
source_category: contest-report
architectures: [sm100]
tags: [nvfp4, grouped-gemm, gemm, fp4, block-scale]
kernel_types: [grouped-gemm, gemm]
url: https://www.gpumode.com/api/leaderboard/730
captured_at: 2026-08-18
---

# Problem 4: NVFP4 Grouped GEMM

The organizer's API defines a list of block-scaled NVFP4 GEMMs with per-group M, N, and K sizes, optimized for NVIDIA B200. Rankings use the geometric mean across four grouped benchmark configurations.

## Organizer speed-of-light values

| Groups | Summary | Speed of light (microseconds) |
|---:|---|---:|
| 8 | N=4096, K=7168, variable M | 18.833 |
| 8 | N=7168, K=2048, variable M | 10.667 |
| 2 | N=3072, K=4096, M=[192, 320] | 2.406 |
| 2 | N=4096, K=1536, M=[128, 384] | 1.525 |

## Captured leaderboard state

On 2026-08-18, the API's `B200` ranking listed `gau.nernst` first with a score of `1.3092433502839783e-05` seconds, or 13.092433502839783 microseconds. The API separately exposes an `NVIDIA` ranking; this page does not merge the two categories. This is a live API snapshot, not a claim about the leaderboard at the original deadline.

A participant post-mortem by Natalia Kokoromyti, published on GPU Mode's news page, records an invalidated 11.191-microsecond exploit, roughly 2 microseconds ahead of the next entry. That number is an invalid benchmark artifact, not kernel performance.

Source locators: [GPU Mode leaderboard API 730](https://www.gpumode.com/api/leaderboard/730), `data.description`, `data.benchmarks`, and `data.rankings.B200[0]`; [Natalia Kokoromyti's GPU Mode-hosted reward-hacking post-mortem](https://www.gpumode.com/news/reward-hacking-nvfp4).
