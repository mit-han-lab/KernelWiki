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
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, doc-ptx-isa-sm100, pr-flashinfer-1039]
blackwell_relevance: PTX cache hints let a kernel distinguish streaming data from data expected to be reused; their benefit is workload-specific.
---

# PTX Cache Policy Differentiation

PTX global-load cache operators and eviction-priority hints let a kernel express different reuse expectations for different input streams. A common experiment is to avoid allocating one-pass matrix data in L1 while giving a reused vector a longer-lived eviction hint.

## Evidence from the NVFP4 GEMV writeup

Yue Zhang reports an optimization sequence that moves from 443 microseconds after a coalescing rewrite to 39 microseconds after using hardware intrinsics and then 27 microseconds after a PTX rewrite. The PTX stage includes byte-unpacking and cache-control changes. The source does not isolate cache policy as the cause of the whole 443-to-27-microsecond delta, so this page does not assign a standalone speedup to it.

## Evaluation procedure

1. Establish reuse distance and working-set size for each load stream.
2. Implement one cache-policy change at a time while keeping address mapping and instruction width fixed.
3. Check generated SASS and profile cache hit rates, sectors, and DRAM traffic.
4. Re-run every supported shape: a policy that helps a large streaming K dimension can hurt a smaller, cache-resident one.

Cache operators are hints with architecture- and workload-dependent effects. They do not override alignment, coalescing, capacity, or correctness requirements.

FlashInfer PR 1039 contains a concrete cache-operation choice in an SM100
attention load path. The following contiguous excerpt performs guarded 16-byte
`cp.async` copies using CUTLASS's `CacheOperation::Always` policy:

```cpp
Vec* dst_ptr = &dst(i);
const Vec* src_ptr = &src(i);
bool guard = elem_less(cc, limitQ);
cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
    dst_ptr, src_ptr, guard);
```

The source does not establish that this policy is best for another stream or
kernel; it is a reproducible example of an explicit policy at one call site.
