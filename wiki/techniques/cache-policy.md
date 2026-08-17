---
id: technique-cache-policy
title: "PTX Cache Policy Differentiation"
type: technique
architectures: [sm100, sm90]
tags: [cache-policy, vectorized-loads]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-vectorized-loads, kernel-nvfp4-gemv, pattern-memory-bound]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv, doc-ptx-isa-sm100]
blackwell_relevance: "Applicable PTX cache and eviction hints can distinguish streaming and reused accesses on SM100, but remain non-binding performance hints."
---

# Cache Policy Differentiation

## Overview

PTX supplies cache operators and eviction-priority hints for eligible memory operations. Examples include `L1::no_allocate`, `L1::evict_first`, and `L1::evict_last`. They affect cache-management policy but do not change value semantics or guarantee residency/bypass behavior.

```ptx
// Four adjacent 32-bit words; required alignment must be satisfied.
ld.global.L1::no_allocate.v4.u32 {a0,a1,a2,a3}, [streaming_addr];
ld.global.L1::evict_last.v4.u32  {b0,b1,b2,b3}, [reused_addr];
st.global.v4.u32 [sink], {a0,a1,a2,a3};
```

Use exact instruction forms supported by the target PTX ISA. A wider vector form changes alignment and register demand, so cache-policy experiments should keep width and work constant.

## Selection logic

- `no_allocate` can reduce L1 admission pressure for a stream with little reuse.
- `evict_last` gives a line lower eviction priority when reuse is expected.
- `evict_first` gives a line higher eviction priority after the access.
- L2 prefetch-size hints change fetch behavior and can over-fetch.

These choices interact with access order across warps/CTAs, sector utilization, L2 reuse, and other kernels. “Input A streams, vector B reuses” is a hypothesis to profile, not a universal GEMV policy.

## Evidence boundary

The GPU Mode NVFP4 write-ups describe progressions that also changed coalescing, decoding, vector width, inline PTX, unrolling, and register allocation. Their large end-to-end deltas cannot be attributed to cache qualifiers alone.

Benchmark default and candidate hints for every representative K/shape class. Keep a hint only when profiler traffic and repeated timing agree, and re-check after layout or block-size changes.
