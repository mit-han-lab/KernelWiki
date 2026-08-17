---
id: kernel-nvfp4-gemv
title: NVFP4 Batched GEMV
type: kernel
architectures: [sm100, sm100a]
tags: [gemv, nvfp4, fp4, block-scale, cache-policy, register-budgeting, vectorized-loads]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemv, batched-gemv]
languages: [cuda-cpp, ptx]
related: [hw-nvfp4, kernel-nvfp4-gemm, pattern-memory-bound]
sources: [contest-gpumode-p1, blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv]
performance_claims:
  - gpu: B200
    software: "Yue participant implementation/write-up; CUDA/toolchain versions not stated in the retained result"
    dtype: nvfp4
    shape: geometric mean across the three contest configurations
    workload: "GPU Mode NVIDIA Problem 1 batched NVFP4 GEMV across all three official configurations"
    metric: latency_us_geomean
    value: 22.392
    measurement_method: "participant-reported geometric mean, cross-checked against the ended public leaderboard snapshot"
    baseline: "none; absolute latency geometric mean"
    limitations: "account/result attribution does not expose private submission code; exact toolchain and timing protocol are not stated"
    source_id: blog-yue-nvfp4
artifact_dir: artifacts/kernels/nvfp4-gemv
---

# NVFP4 Batched GEMV

## Contest operation

The GPU Mode Problem 1 workload computes batched matrix-vector products from NVFP4 E2M1 inputs. Each group of 16 input values has an E4M3 block scale. The FP16 output has shape `M × 1 × L`; this task interface does not supply a separate tensor-level FP32 scale.

The three recorded configurations are `(M,K,L) = (7168,16384,1)`, `(4096,7168,8)`, and `(7168,2048,4)`. Yue's public write-up reports a final 22.392-microsecond geometric mean across all three—not the latency of the first shape. The ended public API snapshot retrieved on 2026-08-16 showed `yue` 11th at 22.392218 us and an account named `Simon` 25th at 25.112154 us. Simon Veitner's articles do not establish that the latter account is his.

```python
def nvfp4_dot(a_codes, b_codes, a_scales, b_scales):
    a = decode_e2m1(a_codes) * expand_blocks(a_scales, block=16)
    b = decode_e2m1(b_codes) * expand_blocks(b_scales, block=16)
    return sum(a * b)
```

This is the numerical contract, not a performance implementation.

## Optimization evidence

The collected participant reports describe:

- shape-specialized compilation and loop structure;
- different cache hints for streamed matrix data and reused vector data;
- packed loads and PTX conversion/unpacking sequences;
- register-limit and instruction-level-parallelism tuning;
- sharing vector values across multiple matrix rows.

Simon Veitner's public follow-up also compares K-parallel reductions using extra blocks,
more threads, and variants with or without FP32 atomics. Its best reported first-
shape comparison is about 6.36× from the article's raw, unitless values; that experiment
is not the current leaderboard submission score.

These are source-reported choices, not universal prescriptions. A lower register cap can increase occupancy but can also spill; a cache hint depends on the working set; and wider memory instructions help only when alignment and the surrounding access pattern permit them.

The task file's 8.622-microsecond “speed of light” is an estimate for the first configuration; the largest listed estimate is 17.275 microseconds for the second. Comparing a per-shape estimate with a suite geometric mean mixes different quantities, so this page does not publish the former “2.6× of SOL” utilization label or attribute the whole gap to decoding and scale application.

## Reproduction

The `full/` bundle contains a byte-pinned upstream vLLM NVFP4 scaled-matrix-multiplication source file. It is implementation evidence for NVFP4 handling, not the unavailable contest GEMV submission. No reconstructed contestant code is retained.
