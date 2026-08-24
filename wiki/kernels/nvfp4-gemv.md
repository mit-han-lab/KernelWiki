---
id: kernel-nvfp4-gemv
title: NVFP4 batched GEMV
type: kernel
architectures: [sm100]
tags: [gemv, nvfp4, fp4, block-scale, cache-policy, register-budgeting, vectorized-loads]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemv, batched-gemv]
languages: [cuda-cpp, ptx, cute-dsl]
related: [hw-nvfp4, kernel-nvfp4-gemm, pattern-memory-bound]
sources: [contest-gpumode-p1, blog-yue-nvfp4, blog-amandeep-nvfp4]
performance_claims:
  - gpu: B200
    dtype: nvfp4
    shape: author-reported leaderboard aggregate
    metric: latency_us
    value: 22.392
    source_id: blog-yue-nvfp4
    source_locator: https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html
---

# NVFP4 batched GEMV

GPU Mode problem 1 defines a batched matrix-vector product using E2M1 data,
per-16 FP8 scales, and FP16 output on B200; it has no separate tensor-level
scale operand. Its task prose calls the scales E4M3FNUZ, while its executable
reference constructs them as `torch.float8_e4m3fn`; the source artifacts agree
on granularity but disagree on the encoding name. The organizer's three
configurations differ in M, K, and batch L, so a submission may specialize
their implementation while preserving one reference function.

This source-backed excerpt from Yue Zhang's CuTe DSL worklog shows one local
change: storing the loaded A and B register tensors in FP16 rather than FP32.
It is a partial optimization snippet, not the contest reference or a complete
kernel:

```python
tArA = cute.make_rmem_tensor_like(tAgA, cutlass.Float16)
tBrB = cute.make_rmem_tensor_like(tBgB, cutlass.Float16)
```

The author writeups report useful experiments with coalescing, conversion
instructions, cache hints, vector width, unrolling, and register allocation.
Their before/after stages combine multiple edits, so none of the full delta is
assigned to a single technique here. Yue Zhang reports a final aggregate of
22.392 microseconds; it is author-reported context, not the live organizer
leaderboard at every later date.

For reproduction, verify the exact PTX spelling against the toolkit's PTX ISA,
check pointer alignment before vector reinterpretation, inspect generated SASS,
and compare each contest shape independently. Use the organizer definition for
packing, scale indexing, the permuted scale layout, and output semantics. The
former local artifact bundle was removed because its “full” file was an
unrelated vLLM GEMM and its five GEMV stages were synthesized reconstructions.
