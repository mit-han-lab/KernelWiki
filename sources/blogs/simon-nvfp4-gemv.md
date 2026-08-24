---
id: blog-simon-nvfp4-gemv
title: NVFP4 GEMV
author: Simon Veitner
url: https://veitner.bearblog.dev/nvfp4-gemv/
source_category: community-note
architectures: [sm100]
tags: [nvfp4, gemv, fp4, block-scale, cute-dsl, batched-gemv]
retrieved_at: 2026-04-17
---

# NVFP4 GEMV (Simon Veitner)

## Scope

The post introduces the CuTe DSL reference kernel for the first GPU Mode NVFP4 challenge. It explains that NVFP4 combines FP4 values with FP8 scale factors applied to blocks of 16 values.

## Reference configuration

- MMA tiler: `(128, 1, 64)` for M, N, and K
- A and B: `Float4E2M1FN`
- Scale factors: `Float8E4M3FN`
- Output: `Float16`
- Scale-factor vector size: 16
- Threads per CTA: 128

For the worked `m=128`, `k=256`, `l=1` example, the post explains how `cute.local_tile` produces the A, B, scale-factor, and output views. Each thread loads a K tile, converts the FP4 values and FP8 scales to FP32, accumulates the scaled products in FP32, and stores the final result as FP16.

## Evidence boundary

The public URL currently contains this reference-kernel walkthrough. It does not contain the earlier locally summarized “improved” strategies or a 6.4x result, so those claims and the pseudo-code artifacts derived from them were removed.

Source locator: [NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv/), “Reference Kernel” section.
