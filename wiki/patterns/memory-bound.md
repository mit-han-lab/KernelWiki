---
id: pattern-memory-bound
title: "Memory Bandwidth Bound"
type: pattern
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound, kernel-nvfp4-gemv]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, doc-nvidia-tuning-guide]
---

## Symptom

A roofline analysis or profile attributes the runtime primarily to data
movement rather than instruction throughput. High measured bandwidth alone is
not sufficient: compare achieved traffic, cache behavior, and useful work with
the selected GPU and workload.

## Likely Causes

1. **Low arithmetic intensity**: for example, some GEMV, small-batch decode, or
   reduction workloads
2. **Limited data reuse**: values are evicted or used too few times to amortize
   their movement
3. **Inefficient access**: uncoalesced transactions, misalignment, redundant
   loads, or an unsuitable cache policy

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Vectorized loads](../techniques/vectorized-loads.md) | Can reduce instruction count when width, alignment, and access order are legal |
| [Cache policies](../techniques/cache-policy.md) | May protect reused data or avoid allocating one-pass traffic; effects are workload-specific |
| [Register budgeting](../techniques/register-budgeting.md) | Trades per-thread state against occupancy and spilling; a lower cap is not automatically faster |
| [TMA multicast](../hardware/tma.md) | Can avoid redundant global-to-shared transfers when multiple CTAs in a cluster consume the same tile |
| [Swizzling](../techniques/swizzling.md) | Can remove measured shared-memory bank conflicts for a compatible layout |

## Caveats
- Confirm the bottleneck with measurements; a kernel may move between memory,
  instruction, and latency limits as shapes change.
- Use the exact GPU SKU's published bandwidth and the workload's measured bytes
  transferred when constructing a speed-of-light bound.
- Reducing decode or address-generation instructions can still matter in a
  nominally memory-heavy kernel. Yue Zhang's reported NVFP4 GEMV stages changed
  memory access, conversion, instruction count, and ILP together, so they do not
  establish a universal optimization order.
