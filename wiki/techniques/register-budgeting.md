---
id: technique-register-budgeting
title: Register budgeting
type: technique
architectures: [sm100, sm90]
tags: [register-budgeting, register-reuse]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [pattern-memory-bound, pattern-register-pressure, kernel-nvfp4-gemv]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv]
blackwell_relevance: TMEM can reduce accumulator-register demand in tcgen kernels, but other live values and compiler allocation still determine register pressure.
---

# Register budgeting

Registers are one input to occupancy, alongside threads, shared memory, barrier
state, architectural limits, and the compiled kernel. Occupancy is stepwise,
not simply inverse to registers per thread. A lower budget can increase active
warps, but it can also cause spills, recomputation, or lost instruction-level
parallelism.

A reproducible comparison keeps source and launch shape fixed and builds
multiple variants, for example:

```bash
nvcc -arch=sm_100a -Xptxas=-v kernel.cu -o kernel-default
nvcc -arch=sm_100a -Xptxas=-v --maxrregcount=80 kernel.cu -o kernel-r80
nvcc -arch=sm_100a -Xptxas=-v --maxrregcount=64 kernel.cu -o kernel-r64
```

Record ptxas registers and spill bytes, achieved occupancy, memory stalls, and
kernel time. Amandeep Singh reports that changing the cap from 80 to 64 did not
affect the tested kernel because its allocation was already below the cap. That
observation is a useful control: a compiler flag that does not bind cannot
explain a speedup.

The former page's fixed “32 registers gives four blocks per SM” and “spills can
be hidden” rules were removed; both depend on the complete resource and access
profile.
