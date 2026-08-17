---
id: hw-tmem
title: "Tensor Memory (TMEM)"
type: hardware
architectures: [sm100, sm100a]
tags: [tmem, tcgen05]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tcgen05-mma, technique-double-buffering, pattern-register-pressure]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-colfax-cutlass]
aliases: [TMEM, "tensor memory", "Tensor Memory"]
---

# Tensor Memory (TMEM)

## Overview

Tensor Memory is an SM100 memory space used by the fifth-generation tensor-core instruction family. An SM provides 512 TMEM columns, each spanning 128 lanes of 32 bits, for 256 KiB total. `tcgen05.mma` writes its destination there; `tcgen05.ld` and `tcgen05.st` move documented fragments between TMEM and registers, and `tcgen05.cp` copies supported shared-memory shapes to TMEM.

“Lane” is the PTX layout dimension, not a permanent mapping of rows 0–127 to four particular warps. The register fragment delivered to a participating warp is defined by the selected `tcgen05.ld` shape and repetition/packing qualifiers. A logical matrix accumulator is likewise mapped according to its MMA kind and instruction descriptor, not as a generic row-major 128-by-N FP32 array.

## Allocation lifecycle

`tcgen05.alloc` reserves a power-of-two number of columns in the documented range and writes the allocation address to shared memory. `tcgen05.dealloc` releases that allocation; the base and column count must match. CTA-group-2 forms coordinate the paired CTAs according to the ISA rules.

```python
def tmem_lifecycle(column_count):
    base = one_warp_allocates_to_shared(column_count)
    cta_waits_for_published_address()
    use_tmem_with_documented_layout(base)
    wait_for_all_tmem_users_and_async_work()
    one_warp_deallocates(base, column_count)
```

This is dependency pseudocode. Exactly one warp performs a group-1 allocation or
deallocation; every thread in that participating warp must execute the same
`.sync.aligned` instruction with the same column count. Group 2 instead requires
one participating warp in each peer CTA. Allocation/deallocation are synchronous
instruction forms with strict participation and ownership requirements; use exact
inline PTX from the selected ISA or a CUTLASS/CuTe wrapper.

## Addressing and movement

A TMEM address identifies a column plus lane information as specified by the instruction family. For `tcgen05.ld` and `tcgen05.st`, the four warps of a warpgroup can all address every column, but warp IDs 0 through 3 within that warpgroup are restricted respectively to lanes 0–31, 32–63, 64–95, and 96–127. The valid load/store shapes include forms such as `32x32b`, with repetition and packing variants. Register counts and lane-to-element mappings come from the corresponding PTX fragment table.

For asynchronous TMEM load/store forms, use the documented `tcgen05.wait::ld` or `tcgen05.wait::st` completion operation before consuming registers or reusing sources. For MMA, use `tcgen05.commit` with an `mbarrier` and wait for that completion before dependent TMEM loads. Cross-thread ordering around an execution barrier may additionally require the paired `tcgen05.fence::before_thread_sync`/`::after_thread_sync` protocol.

## Resource accounting

TMEM allocation is measured in columns, but logical element count does not generally equal `128 * allocated_columns`. Datatype packing, instruction shape, and accumulator layout determine how many columns a tile uses. Therefore tables that infer “N columns for a 128-by-N FP32 tile” without naming the MMA layout are unsafe.

Multiple accumulator stages can share the 512-column budget when their exact column ranges do not overlap. That may enable MMA/epilogue overlap, but it reduces capacity available to other TMEM operands or scratch layouts and can alter the legal instruction configuration.

## What TMEM changes—and what it does not

Moving large MMA destinations out of general registers can reduce accumulator register pressure. It does not make all other register pressure disappear, guarantee higher occupancy, or make TMEM a general replacement for shared memory. Epilogues still load fragments into registers, and pipeline state, address arithmetic, and fused operations consume registers.

The cited official documents do not publish a general “420-cycle TMEM cache-miss latency,” nor does TMEM have the cache semantics implied by that phrase. Use target-specific microbenchmarks for latency or bandwidth claims and preserve their instruction shape, dependency chain, GPU, clocks, and toolchain.

## Failure modes

- Using a plausible but wrong fragment shape or register tuple.
- Reading MMA output after a fence but without waiting for MMA completion.
- Reusing or deallocating columns while an async operation still accesses them.
- Assuming one warp's register-fragment mapping applies to every warp and shape.
- Hand-computing column requirements instead of using the instruction layout or library trait.
