---
id: migration-wgmma-to-tcgen05
title: "Migrating from wgmma to tcgen05"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tcgen05, wgmma, tmem]
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, blog-colfax-cutlass]
blackwell_relevance: "The fifth-generation Tensor Core path changes issue granularity, accumulator storage, descriptors, and completion handling."
confidence: source-reported
reproducibility: pseudocode
---

# Migrating from wgmma to tcgen05

This is a redesign rather than an opcode substitution. Hopper WGMMA is warpgroup-collective and keeps D in registers. The SM100 `tcgen05.mma` family has single-thread MMA issue and keeps D in Tensor Memory.

## Required changes

1. **Move D to TMEM.** Allocate columns with one fully active warp, distribute the resulting shared-memory address safely, and deallocate with a fully active warp before exit.
2. **Rebuild operand paths.** B is described in shared memory. A may use either a shared-memory descriptor or a TMEM address, depending on the selected instruction form.
3. **Build a real instruction descriptor.** The required 32-bit `idesc` encodes shapes, types, sparsity, and other details. Passing zero as an “unused scale descriptor” is invalid guidance.
4. **Change issue ownership.** One elected thread issues `tcgen05.mma`; this does not make allocation, TMEM load, or TMEM store single-thread operations.
5. **Use the documented completion path.** `tcgen05.commit` can track prior asynchronous MMA work through an mbarrier. Apply the required TMEM fences and waits before another thread or memory proxy consumes results.
6. **Rebuild the epilogue.** Load the TMEM D fragment into registers with the matching warp-collective copy before applying ordinary arithmetic or stores.
7. **Select layouts from constraints.** Make the shared-memory descriptor agree with the physical layout. Do not mechanically change every Hopper layout to a 128-byte swizzle.

## Allocation ownership sketch

```text
if warp_id == allocation_warp:       # all 32 lanes take the same branch
    tcgen05.alloc.sync.aligned(...)
synchronize_and_publish_tmem_address()
if elected_mma_thread:
    tcgen05.mma(..., idesc, enable_input_d)
wait_for_mma_completion_and_fence()
if warp_id == allocation_warp:       # fully active warp again
    tcgen05.dealloc.sync.aligned(...)
```

The same CTA-group qualifier must be used consistently by all `tcgen05` instructions in a kernel. For `cta_group::2`, allocation management is collective across one warp in each peer CTA and the peer must remain active for the paired operation.
