---
id: contest-gpumode-p1
title: 'GPU Mode NVIDIA Problem: NVFP4 Batched GEMV'
source_category: contest-report
architectures: [sm100, sm100a]
tags: [nvfp4, gemv, fp4, block-scale]
techniques: [vectorized-loads, cache-policy, register-reuse, per-k-specialization, data-reuse, register-budgeting, loop-unrolling]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx, cute-dsl]
url: https://github.com/gpu-mode/reference-kernels/blob/51e22db671d36c1c76091c43c36a44546ba324a1/problems/nvidia/nvfp4_gemv/task.yml
leaderboard_url: https://www.gpumode.com/leaderboard/595
leaderboard_api_url: https://www.gpumode.com/api/leaderboard/595
leaderboard_retrieved_at: 2026-08-16
submissions:
  - rank: 11 in the public API snapshot retrieved 2026-08-16
    participant: yue
    score: 22.392 us geometric mean (author-reported)
    technique: shape specialization, PTX load/decode work, vector reuse, and instruction-level parallelism
    submission_truth: unavailable
    code_unavailable_reason: The public write-up describes the optimization progression but does not publish the complete submission; earlier local files were illustrative reconstructions and have been removed.
  - rank: not established by public source
    participant: Amandeep
    score: 26.7 us, 45.1 us, and 16.4 us for the three benchmark shapes (author-reported)
    technique: shape-specific PTX kernels, packed loads, cache-policy experiments, and register tuning
    submission_truth: unavailable
    code_unavailable_reason: The public retrospective reports measurements and methods but does not publish a complete submission artifact suitable for this local contest bundle.
---

# NVFP4 Batched GEMV

## Pinned task contract

The pinned GPU Mode task defines a batched matrix-vector product on NVIDIA B200. Packed NVFP4 E2M1 inputs have logical shapes `M x K x L` and `1 x K x L`; their FP8 E4M3 block-scale tensors have shapes `M x (K/16) x L` and `1 x (K/16) x L`. The output is FP16 with shape `M x 1 x L`. The task requires K to be divisible by 64 and ranks submissions by the geometric mean of three benchmark latencies.

| M | K | L | Task-file speed-of-light estimate at 1.5 GHz |
|---:|---:|---:|---:|
| 7,168 | 16,384 | 1 | 8.622 us |
| 4,096 | 7,168 | 8 | 17.275 us |
| 7,168 | 2,048 | 4 | 4.317 us |

Those values are estimates recorded by the task author, not measured winning latencies.

The task YAML labels the scale dtype `e4m3fnuz`, while its companion pinned Python reference constructs `torch.float8_e4m3fn`. This upstream suffix mismatch is not silently resolved here; reproduction should follow the actual contest runner/revision and verify scale interpretation.

## Result boundary

The unauthenticated [public leaderboard API](https://www.gpumode.com/api/leaderboard/595)
reported `time_left: ended` for its 2025-11-29 06:59 UTC deadline and contained
207 NVIDIA-group entries when retrieved on 2026-08-16. Its displayed top
three were `s.am._` at 18.549562 us, `gau.nernst` at 18.552845 us, and
`shellsmile15795` at 18.707609 us. It placed `yue` 11th at 22.392218 us,
the account `Simon` 25th at 25.112154 us, and `currybab` 52nd at 32.475188 us.

This is a dated snapshot of an ended leaderboard; 33 displayed submissions
postdate the task deadline, so the display is not labeled the deadline-final
podium or prize outcome. Yue's public write-up independently reports a 22.392-us final
submission, while Amandeep's retrospective reports 26.7, 45.1, and 16.4 us for
the three individual shapes.

No participant submission is archived locally. Simon Veitner's two public
articles are available and summarized in `blog-simon-nvfp4-gemv`, but neither
article identifies its author with the leaderboard account `Simon`; no identity
mapping or byte-identical ranked submission is asserted here.
