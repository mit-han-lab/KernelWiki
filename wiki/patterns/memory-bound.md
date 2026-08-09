---
id: pattern-memory-bound
title: "Memory Bandwidth Bound"
type: pattern
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-cache-policy, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound, kernel-nvfp4-gemv]
sources: [blog-amandeep-nvfp4, blog-yue-nvfp4, doc-nvidia-tuning-guide, doc-ptx-isa-sm100]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
reproducibility: concept
---

## Establish the Memory Roof

A kernel is memory-bandwidth bound when its measured arithmetic intensity places it on the memory side of the relevant roofline and its useful throughput is limited by an attained memory ceiling. Compute intensity from the operations actually performed and bytes transferred at the memory level under study. Compare against a measured ceiling for the same device, clocks, memory level, datatype path, and environment; a nominal product bandwidth is only an upper-bound input.

High DRAM throughput, low tensor-core activity, or a workload label such as GEMV is not sufficient alone. Record useful/requested bytes, transferred sectors or bytes, achieved bandwidth, cache behavior, scheduler issue/stalls, and end-to-end time. This separates four cases that can otherwise look similar:

| Evidence | More precise interpretation |
|---|---|
| Useful bytes and transferred bytes are close; attained bandwidth is near the measured roof | Plausible bandwidth-ceiling limit |
| Transferred bytes substantially exceed useful bytes | Coalescing, overfetch, cache, or redundant-traffic problem |
| Bandwidth is below the roof and warps lack ready work | Latency, dependency, insufficient concurrency, or issue problem |
| Bandwidth is high only during one phase | Phase balance or fusion opportunity; whole-kernel boundedness remains unproven |

Single-use data can lower operations per transferred byte, but “poor reuse” is actionable only if additional legal reuse exists. Uncoalesced access and cache interference may increase traffic or latency; they are not synonyms for saturation of the DRAM bandwidth ceiling.

## Controlled Candidate Tests

### Load width and coalescing

Choose only legal, naturally aligned vector forms and handle tails separately. Compare scalar/narrow and wider variants with identical mapping. Record instruction count, requested and transferred bytes, transactions, registers, spills, achieved bandwidth, and time. Wider instructions do not guarantee fewer hardware transactions or higher bandwidth.

### Cache policy

PTX L1 eviction priorities and L2 prefetch controls are hints. `no_allocate` does not guarantee bypass, and `evict_last` does not guarantee residence. Characterize address order, reuse distance, working set, and interfering streams, then vary one hint at a time against the default. Retain it only when a representative end-to-end comparison improves.

### Register budget and concurrency

Inspect compiled registers, spills, SMEM, threads, and the actual occupancy-limiting resource before applying `-maxrregcount`. A cap may be clamped by ABI requirements, introduce spill traffic, or leave occupancy unchanged. Higher theoretical occupancy is useful only if it raises ready work or attained bandwidth enough to reduce runtime.

### TMA multicast and shared-memory layout

TMA multicast can issue one global-to-shared tensor copy to selected CTAs' shared-memory destinations within a cluster. It is relevant when those CTAs consume the same operand and the cluster/lifetime/barrier costs are valid. It does not help a stream with no inter-CTA reuse.

Swizzling changes a shared-memory address mapping and can reduce conflicts for a specified access pattern. It does not universally eliminate conflicts and is not by itself a DRAM optimization. Validate the legal tensor-map/layout constraints and compare bank-conflict, traffic, and time measurements.

## NVFP4 GEMV Negative Controls

The cited Amandeep NVFP4 GEMV report is useful precisely because plausible memory-oriented changes failed. A wider `uint2` load was 16–25% slower on the reported shapes, and reducing `maxrregcount` from 80 to 64 had no effect. These observations do not prove that wide loads or register caps are generally harmful; they refute a fixed optimization priority based only on the “memory-bound GEMV” label.

Use the same discipline for compute work. If the kernel is genuinely at its attainable memory roof, compute-only instruction reductions may have little effect. But address calculation, decoding, and instruction-level parallelism can affect memory issue and latency hiding before that roof is reached. State a predicted counter change, alter one variable, and keep or reject the hypothesis from matched time and profiler evidence.

## Reproduction Record

Report input shapes/distributions, useful work and bytes, cache state, warmup/repetitions/statistic, GPU and clocks, software versions, generated instructions, resource usage, roofline assumptions, achieved memory-level bandwidth, requested/transferred efficiency, scheduler activity, and correctness tolerance. Include negative and regressing variants so a category heuristic is not mistaken for causal evidence.

## Primary References

- [Nsight Compute 2025.3 Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
- [PTX ISA 9.0 cache-operation semantics](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
- [PTX ISA 9.0 TMA multicast](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [Amandeep NVFP4 GEMV attempts](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
