---
id: technique-register-budgeting
title: "Register Budgeting for Occupancy"
type: technique
architectures: [sm100, sm90]
tags: [register-budgeting, register-reuse]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [pattern-memory-bound, pattern-register-pressure, kernel-nvfp4-gemv]
sources: [doc-nvidia-tuning-guide, blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv]
blackwell_relevance: "TMEM can reduce MMA-accumulator register demand on SM100, while occupancy still depends on registers, shared memory, threads, barriers, clusters, and launch constraints."
---

# Register Budgeting

## Overview

Registers are allocated to resident thread blocks in hardware-defined granularities. A kernel can become register-limited, but occupancy is not simply the inverse of registers per thread: threads per block, shared memory, architectural block/warp limits, cluster requirements, and allocation rounding all participate.

Use compiler resource output and the CUDA occupancy APIs/calculator for the exact kernel and target.

## Controls

```cuda
// Requests launch compatibility with 256 threads and at least two blocks/SM.
// It does not promise that two blocks will reside or that spills are profitable.
__launch_bounds__(256, 2)
__global__ void kernel(/* ... */) {
    // implementation
}
```

`__launch_bounds__`, `__maxnreg__` where supported, and compiler register limits constrain allocation decisions. They can reduce unrolling or introduce local-memory spills. Always inspect the generated register/spill report and benchmark rather than assuming a lower count is better.

## Tuning loop

1. Identify whether achieved occupancy or eligible warps actually limit the kernel.
2. Record registers, spill loads/stores, shared memory, and block size.
3. Sweep a small set of launch bounds or code variants.
4. Validate numerical results and benchmark representative shapes.
5. Prefer the fastest stable configuration, not the highest occupancy percentage.

Spill traffic is ordinary memory-system work and is not automatically hidden because the original kernel was memory-bound. It can worsen the same bottleneck.

## Evidence boundary

Register/latency pairs reported by NVFP4 GEMV competition write-ups describe specific implementations and benchmark cases. They do not establish that 32 registers is an SM100 optimum or that the latency difference is caused solely by occupancy. Preserve those measurements as source-reported case studies unless controlled variants establish causality.
