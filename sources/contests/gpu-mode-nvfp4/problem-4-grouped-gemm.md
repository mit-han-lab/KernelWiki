---
id: contest-gpumode-p4
title: 'GPU Mode NVIDIA Problem: NVFP4 Grouped GEMM'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, grouped-gemm, fp4, block-scale, tcgen05, tmem, tma, moe]
techniques: [warp-specialization, tile-scheduling, pipeline-stages, kernel-fusion]
hardware_features: [nvfp4, fp4, block-scale, tcgen05, tmem, tma, clc]
kernel_types: [grouped-gemm, gemm, moe]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels/blob/51e22db671d36c1c76091c43c36a44546ba324a1/problems/nvidia/nvfp4_group_gemm/task.yml
leaderboard_url: https://www.gpumode.com/leaderboard/730
leaderboard_api_url: https://www.gpumode.com/api/leaderboard/730
leaderboard_retrieved_at: 2026-08-16
---

# NVFP4 Grouped GEMM

## Pinned task contract

The task evaluates a list of independent NVFP4 matrix products on NVIDIA B200. Each group supplies packed A and B tensors, an FP16 C tensor, FP8 E4M3FNUZ block-scale tensors, and an `(M,N,K,L)` tuple. M and N must satisfy the implementation's MMA tile divisibility constraints, and K must be divisible by 256. Ranking uses the geometric mean of four grouped benchmark cases containing either two or eight products.

The task's analytical 1.5-GHz speed-of-light estimates are 18.833, 10.667, 2.406, and 1.525 us for the four published benchmark cases. These are model estimates, not contestant results.

The task YAML labels scale tensors `e4m3fnuz`, while the companion reference constructs `torch.float8_e4m3fn`. That upstream suffix mismatch should be resolved against the actual runner/revision before reproducing results.

## Result boundary

The unauthenticated [public leaderboard API](https://www.gpumode.com/api/leaderboard/730)
reported `time_left: ended` for its 2026-02-21 07:30 UTC deadline and contains
separate ranking groups. In its 69-entry B200 group on 2026-08-16,
`gau.nernst` was first at 13.092434 us, `guaguabear` second at 13.865532 us,
and `Ouye Xie` third at 14.116127 us. `nataliakokoromyti` was 9th at
16.021590 us, `currybab` 16th at 20.731629 us, and `Simon` 35th at
28.914693 us. The separate NVIDIA group had 55 entries and a different ranking,
so every score must retain its group label.

These are dated ended-leaderboard ranks rather than an independently verified award table, and the
API does not publish participant implementation details. It also no longer lists
the separate invalid 11.191-us reward-hacking score; that incident is retained
only with its author-published post-mortem (`blog-gpu-mode-reward-hack`).
