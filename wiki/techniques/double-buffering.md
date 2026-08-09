---
id: technique-double-buffering
title: "Double/Multi-Buffering Patterns"
type: technique
architectures: [sm100, sm90]
tags: [double-buffering, tmem, pipeline-stages]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
reproducibility: pseudocode
prerequisites: [hw-tmem]
related: [hw-tmem, technique-pipeline-stages, technique-epilogue-fusion]
sources: [blog-tcgen05-tutorial, doc-nvidia-tuning-guide, pr-flashinfer-2387]
blackwell_relevance: "SM100 can keep independent accumulator regions in TMEM while a separate SMEM pipeline stages operands; either mechanism remains an implementation and tuning choice."
---

# Double/Multi-Buffering Patterns

## What the pattern guarantees

Double- or multi-buffering reserves disjoint storage regions and transfers ownership between producers and consumers. While a consumer reads stage `s`, a producer may fill another stage. The storage alone does not establish overlap: the program also needs a completion edge before consumption and a read-complete edge before reuse.

On SM100 these ideas can be applied independently:

- SMEM stages can hold operand tiles while TMA production overlaps MMA consumption.
- Separate TMEM regions can hold output accumulators while an epilogue drains a completed region and MMA writes a different region.

The pinned `matmul_v6.cu` from Gau Nernst's tutorial uses both mechanisms. That is a concrete design, not a rule that every optimized Blackwell GEMM needs both.

## TMEM regions

The CTA-visible TMEM address structure has 128 lanes and 512 columns of 32-bit cells. Allocation is column-granular across all 128 lanes. A kernel that allocates 512 columns may choose two 256-column regions:

```text
region[0] = columns [0, 256)
region[1] = columns [256, 512)
```

This equal split is one implementation choice. The actual invariant is that every simultaneously live region is inside the allocation and does not alias another region whose MMA or epilogue access is still outstanding. Region sizes can differ, and logical element capacity depends on the tcgen05 instruction kind and packing rather than the raw cell count alone.

For each region, prove this lifecycle:

| Transition | Required edge |
|---|---|
| free → MMA-owned | every prior epilogue load from the region has completed |
| MMA-owned → ready | relevant tcgen05 work is committed to an mbarrier and completion is observed |
| ready → epilogue-owned | readers observe the matching barrier phase before `tcgen05.ld` |
| epilogue-owned → free | all collective loads complete with `tcgen05.wait::ld`, then every reader reports completion |

TMEM must first be allocated by the required participating warp and its address safely published. Matching collective deallocation is required before every kernel exit. Reused mbarriers need the correct arrival count and phase/parity state; a one-time wait on an address is not a reusable two-buffer protocol.

## SMEM stages

An SMEM operand stage normally has its own full/empty state:

1. The producer waits until stage `s` is empty.
2. It issues the stage's TMA copies and accounts for expected transaction bytes.
3. The consumer observes the full barrier's matching phase before using the stage.
4. After the final dependent read, the consumer releases the empty barrier for reuse.

For binary16 `A[128,64]` and `B[64,256]`, the unpadded payload arithmetic is:

```text
A = 128 × 64 × 2 bytes = 16 KiB
B =  64 × 256 × 2 bytes = 32 KiB
one stage = 48 KiB; three stages = 144 KiB
```

That is not a complete C++ shared-storage layout. Barriers, descriptors, epilogue scratch, padding, alignment, and swizzled physical layouts also consume or constrain shared memory. Compute capability 10.0 supports up to 228 KiB of shared memory per SM, while per-block opt-in limits and all other residency resources still apply.

FlashInfer PR 2387 is a second pinned example. Its merged `selective_state_update.cuh` has `state[numStages][...]` plus `bar_full` and `bar_empty` arrays and uses stage-specific producer/consumer handoffs. One path selects three stages and another caps a geometry-derived count at four, illustrating why stage count is policy rather than an architecture-wide default.

## Hopper and Blackwell storage tradeoff

For Hopper `wgmma.mma_async.m64nNk16` with FP32 D, each warpgroup thread holds `N/2` accumulator registers. At `N=256`, one D fragment is 128 registers per thread; two simultaneously live fragments are 256 registers, or 1024 bytes, per thread before other live values. That arithmetic describes the ISA fragment, not a promise that a C++ array stays in registers or that a particular launch is resident.

On SM100, tcgen05 keeps resident D in TMEM. This avoids a long-lived per-thread D vector, but the kernel still uses registers for addresses, descriptors, loop and barrier state, `tcgen05.ld` destinations, and epilogue temporaries. TMEM buffering also consumes TMEM capacity and synchronization state. Neither architecture therefore has a fixed occupancy result from “double buffering” alone.

| Concern | Hopper WGMMA D | Blackwell tcgen05 D |
|---|---|---|
| Resident storage | per-thread register fragment | allocated TMEM region |
| Two live outputs | two disjoint register fragments | two disjoint TMEM regions |
| Epilogue access | dependent use after WGMMA completion | collective `tcgen05.ld` and `tcgen05.wait::ld` after MMA completion |
| Main resource question | compiled registers, SMEM, block shape, and other limits | TMEM columns plus compiled registers, SMEM, block/cluster shape, and other limits |

## Decide with a controlled comparison

Buffering is useful only when useful producer/consumer work overlaps enough to repay extra storage and coordination. For each concrete variant:

1. Keep shape, datatype, outputs, launch policy, warmup, and timing statistics identical.
2. Compare one versus multiple TMEM regions and the legal SMEM stage counts.
3. Record compiled registers/thread, spill traffic, static/dynamic SMEM, TMEM columns, barriers, threads/CTA, and cluster shape.
4. Use the CUDA occupancy APIs for the actual resource record; do not infer a universal CTA/SM count from an instruction tile.
5. Inspect profiler pipeline and barrier stalls. PTX specifies mbarrier semantics, not a fixed “few cycles” latency.
6. Inspect generated PTX/SASS and execute correctness tests that force region wraparound and pipeline-tail paths.

Useful negative tests deliberately delay the epilogue, swap a barrier phase, omit the final drain, or reuse a region early. A correct test should detect stale/overwritten output or time out under a watchdog rather than silently accepting the broken schedule.

## Primary references

- [PTX ISA 9.0 Tensor Memory](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensor-memory)
- [PTX ISA 9.0 tcgen05 allocation](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc)
- [PTX ISA 9.0 tcgen05 load and wait](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-ld)
- [PTX ISA 9.0 mbarrier](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- [CUDA 13.0.2 Blackwell Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-tuning-guide/index.html#occupancy)
- [Pinned combined SMEM/TMEM example](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v6.cu)
- [FlashInfer PR 2387 merged source](https://github.com/flashinfer-ai/flashinfer/blob/18804cd51734cccf807356d017733bc757677f15/include/flashinfer/mamba/selective_state_update.cuh)

## Related

- [Tensor Memory](../hardware/tmem.md) — allocation, addressing, and lifetime rules
- [pipeline stages](pipeline-stages.md) — SMEM pipeline construction and tail handling
- [epilogue fusion](epilogue-fusion.md) — work performed while draining an output tile
