---
id: doc-ptx-isa-sm100
title: "PTX ISA 9.0 SM100 Instruction Reference"
url: https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [ptx, tcgen05, tmem, clc, tma, nvfp4, fp4, fp8, fp6, block-scale, mbarrier]
retrieved_at: 2026-08-08
---

# PTX ISA 9.0 SM100 Instruction Reference

## Evidence Scope

This card routes KernelWiki claims to the archived CUDA 13.0.2 PTX ISA 9.0 reference. Exact grammar, target support, and ordering are version-sensitive; the rolling PTX URL must not substitute for this archive when a page claims PTX ISA 9.0 behavior.

## Relevant Normative Sections

- `tcgen05.mma`: unscaled, block-scaled, and integer operand grammar; CTA-group rules; descriptors; target restrictions; asynchronous completion.
- Tensor Memory: 128 lanes by 512 columns of 32-bit cells per SM, 32-bit addresses, allocation/deallocation, collective ld/st, cp, shift, and waits.
- `tcgen05.commit` and `tcgen05.fence`: asynchronous completion and cross-thread execution-ordering mechanisms with distinct roles.
- `clusterlaunchcontrol.try_cancel` and `query_cancel`: 16-byte shared response, mbarrier completion, success query, and first-CTA-coordinate decoding.
- `cp.async.bulk.tensor`: tensor-map loads/stores, CTA/cluster destinations, mbarrier or bulk-group completion, and `.multicast::cluster`.
- Packed `cvt`: E2M1, E3M2, E2M3, and FP8 conversion forms and their target notes.
- `mov` and `ld`: typed scalar/vector moves, vector load widths, cache operators, and eviction-priority hints.
- `mbarrier`: phase, arrival, transaction-count, wait, scope, and memory-ordering semantics.

## Scope Limits

The ISA defines legal behavior, not a universal performance ranking between instruction sequences, cache hints, vector widths, swizzles, or pipeline depths. A legal fragment is not a complete kernel: operand declarations, descriptors, collective participation, proxy visibility, lifetimes, completion waits, and target/toolchain selection still matter.

## Direct Links

- [PTX ISA 9.0 contents](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/contents.html)
- [`tcgen05.mma`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma)
- [Tensor Memory](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensor-memory)
- [`clusterlaunchcontrol.try_cancel`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [`cvt`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- [`ld`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
