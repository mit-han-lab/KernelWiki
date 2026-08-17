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

# Memory Bandwidth Bound

## Diagnosis

A kernel is bandwidth-bound for a workload when measured useful performance is constrained by a memory level's sustainable bandwidth rather than instruction throughput. Low arithmetic intensity is a warning, not proof: determine which level (DRAM, L2, L1, shared memory, or a communication link) is limiting with a roofline model and profiler counters.

## Common causes

- Little reuse, as in many GEMV and reduction shapes.
- Uncoalesced, redundant, or over-fetched transactions.
- A working set or access order that defeats cache reuse.
- Shared-memory bank conflicts or pipeline backpressure.
- Data-format conversion that expands traffic or blocks issue.

## Candidate techniques

- [Vectorized loads](../techniques/vectorized-loads.md) when aligned wider instructions reduce instruction pressure without over-fetch.
- [Cache policy](../techniques/cache-policy.md) when streams have demonstrably different reuse.
- [Register budgeting](../techniques/register-budgeting.md) only if latency hiding is occupancy-limited and spills remain acceptable.
- [TMA multicast](../hardware/tma.md) when cluster peers truly share an operand.
- [Swizzling](../techniques/swizzling.md) for a measured shared-memory bank-conflict pattern.

The B200 SXM product is advertised with a particular peak HBM bandwidth, but actual devices/SKUs, clocks, ECC, access mix, and benchmark methodology determine sustainable bandwidth. Use the installed device's specification and a measured copy/stream baseline rather than an architecture-wide constant.

## Caveats

Optimizing arithmetic can still matter in a nominally memory-bound kernel if it reduces address/decode dependencies or enables more concurrent requests. Likewise, raising occupancy can worsen cache pressure. Rebuild the roofline estimate after each material change.
