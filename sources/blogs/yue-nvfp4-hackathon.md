---
id: blog-yue-nvfp4
title: Blackwell NVFP4 Kernel Hackathon Journey
author: Yue Zhang
url: https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html
source_category: community-note
architectures:
- sm100
- sm100a
tags:
- nvfp4
- gemv
- fp4
- block-scale
- batched-gemv
techniques:
- vectorized-loads
- cache-policy
- loop-unrolling
hardware_features:
- nvfp4
- fp4
- block-scale
kernel_types:
- batched-gemv
- gemv
languages:
- cuda-cpp
- ptx
- cute-dsl
retrieved_at: 2026-08-08
---

# Blackwell NVFP4 Kernel Hackathon Journey

## Evidence Scope

This is an evidence-scoped summary of Yue Zhang's own optimization report for Problem 1. Its performance values and explanations are author-reported; the post does not provide raw repeated-trial data or a complete released submission. At retrieval, the post still said its source-code link was coming soon.

## Reported Progression

| Stage | Combined change | Author-reported latency |
|---|---|---:|
| Initial CuTe DSL | First working CuTe path | ~100 µs |
| Optimized CuTe DSL | Scale-load/arithmetic changes and thread collaboration | ~33 µs |
| Initial CUDA | Naive hand-written path | ~2000 µs |
| CUDA optimization 1 | Coalescing, shared B, thread collaboration, warp reduction | ~443 µs |
| CUDA optimization 2 | Remove shared B, per-thread tiles, `float4` loads, hardware intrinsics | ~39 µs |
| CUDA optimization 3 | Vectorized PTX FP4 and scale decode | ~27 µs |
| Parameter tuning | Threads per row and rows per block | ~26 µs |
| ILP | Two tiles per loop iteration | ~22.9 µs |
| Aggressive PTX fusion | Decode, scales, multiply, and accumulation in a larger PTX block | ~22.3 µs |
| Submitted leaderboard score | Geometric mean | 22.392 µs |

The 443-to-39 step is not a coalescing-only result, and the 443-to-27 endpoints do not isolate C intrinsics versus PTX. Each spans multiple simultaneous changes.

## Reported Technical Details

- The CuTe path reduced duplicate scale loads and scale-product arithmetic, then used multiple threads per output with a shared-memory partial-sum reduction.
- Loading the entire B vector into shared memory and double buffering with asynchronous copy did not improve that CuTe attempt.
- The first CUDA improvement combined coalescing, B/SFB shared-memory staging, multiple threads per row, and warp reduction.
- The next CUDA stage removed B shared-memory staging and combined per-thread K tiles, 16-byte `float4` loads, and hardware FP4 conversion intrinsics.
- The PTX stage used `mov.b32` decomposition and packed `cvt.rn.f16x2.e2m1x2` conversion as part of a vectorized decode path.
- Processing two tiles per loop improved the author's implementation; three or four tiles were slightly slower.

These observations do not establish that shared memory, load width, inline PTX, or ILP has the same effect in another kernel.

## Primary Source

- [Yue Zhang, “My Blackwell NVFP4 Kernel Hackathon Journey”](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
