---
id: contest-gpumode-p3
title: 'GPU Mode NVIDIA Problem: NVFP4 Dual GEMM with SiLU'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, gemm, fp4, block-scale, tcgen05, tmem, tma]
techniques: [warp-specialization, kernel-fusion, epilogue-fusion, pipeline-stages]
hardware_features: [nvfp4, fp4, block-scale, tcgen05, tmem, tma]
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels/blob/51e22db671d36c1c76091c43c36a44546ba324a1/problems/nvidia/nvfp4_dual_gemm/task.yml
leaderboard_url: https://www.gpumode.com/leaderboard/598
leaderboard_api_url: https://www.gpumode.com/api/leaderboard/598
leaderboard_retrieved_at: 2026-08-16
---

# NVFP4 Dual GEMM with SiLU

## Pinned task contract

The task accepts one NVFP4 A tensor, two NVFP4 B tensors, their FP8 E4M3 block scales, and an FP16 output. Its reference operation is a dual matrix product followed by SiLU gating. A, B1, and B2 are K-major; K must be divisible by 256; and results are ranked by the geometric mean of four benchmark latencies.

| M | N | K | L | Task-file speed-of-light estimate at 1.5 GHz |
|---:|---:|---:|---:|---:|
| 256 | 4,096 | 7,168 | 1 | 4.708 us |
| 512 | 4,096 | 7,168 | 1 | 8.714 us |
| 256 | 3,072 | 4,096 | 1 | 2.125 us |
| 512 | 3,072 | 7,168 | 1 | 6.535 us |

The task file does not include participant implementations, but the unauthenticated
[public leaderboard API](https://www.gpumode.com/api/leaderboard/598) reported
`time_left: ended` for its 2026-01-20 07:59 UTC deadline and contained
109 NVIDIA-group entries on 2026-08-16. Its current top three were
`gau.nernst` at 13.123660 us, `guaguabear` at 13.794906 us, and
`arseni_ivanov` at 13.875511 us. `Simon` was 4th at 14.054539 us, `yue`
26th at 15.225916 us, and `currybab` 38th at 20.412551 us. All displayed
submissions precede that deadline. These dated API ranks are not independent
award claims, and the API does not establish the
participant-specific fusion descriptions previously attributed here.

The task YAML says `e4m3fnuz`, while the companion reference constructs `torch.float8_e4m3fn`; that upstream suffix mismatch remains an explicit reproduction caveat.
