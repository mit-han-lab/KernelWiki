---
id: contest-gpumode-p2
title: 'GPU Mode NVIDIA Problem: NVFP4 GEMM'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, gemm, fp4, block-scale, tcgen05, tmem, tma]
techniques: [warp-specialization, pipeline-stages, swizzling, register-reuse]
hardware_features: [nvfp4, fp4, block-scale, tcgen05, tmem, tma]
kernel_types: [gemm]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels/blob/51e22db671d36c1c76091c43c36a44546ba324a1/problems/nvidia/nvfp4_gemm/task.yml
leaderboard_url: https://www.gpumode.com/leaderboard/597
leaderboard_api_url: https://www.gpumode.com/api/leaderboard/597
leaderboard_retrieved_at: 2026-08-16
---

# NVFP4 GEMM

## Pinned task contract

The task computes an NVFP4 block-scaled matrix product on NVIDIA B200. A and B have logical K-major shapes `M x K x L` and `N x K x L`, scale-factor tensors use FP8 E4M3 with one entry per 16 logical K elements, and C is FP16. K must be divisible by 256. The ranking metric is the geometric mean of these three benchmark latencies:

| M | N | K | L | Task-file speed-of-light estimate at 1.5 GHz |
|---:|---:|---:|---:|---:|
| 128 | 7,168 | 16,384 | 1 | 8.994 us |
| 128 | 4,096 | 7,168 | 1 | 2.354 us |
| 128 | 7,168 | 2,048 | 1 | 1.333 us |

The task file itself does not publish contestant identities or implementation
details, but the unauthenticated [public leaderboard API](https://www.gpumode.com/api/leaderboard/597)
reports `time_left: ended` for its 2025-12-21 07:59 UTC deadline. Its 118-entry NVIDIA snapshot on 2026-08-16 had
`gau.nernst` first at 9.981889 us, `s.am._` second at 10.060110 us, and
`billcarson` third at 10.137411 us. The previously recorded values were real but
mislabelled as a podium: `Simon` was 8th at 10.806750 us, `yue` 9th at
10.914084 us, and `currybab` 10th at 10.930623 us in this snapshot. The API
does not expose their implementation techniques.

All displayed submissions precede that deadline. The page still calls this a
dated API snapshot rather than independently asserting the organizer's prize or
award decision.

The task YAML says `e4m3fnuz`, whereas the companion reference constructs `torch.float8_e4m3fn`; that upstream suffix mismatch requires runner-level verification.
