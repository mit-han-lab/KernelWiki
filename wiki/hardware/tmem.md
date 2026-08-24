---
id: hw-tmem
title: "Tensor Memory (TMEM)"
type: hardware
architectures: [sm100, sm100a]
tags: [tmem, tcgen05]
confidence: source-reported
related: [hw-tcgen05-mma, technique-double-buffering, pattern-register-pressure]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, blog-colfax-cutlass]
aliases: [TMEM, "tensor memory", "Tensor Memory"]
---

# Tensor Memory (TMEM)

Tensor Memory is dedicated on-chip storage used by fifth-generation Tensor Core operations. For `sm_100a`/`sm_100f`, the PTX ISA exposes a CTA-visible two-dimensional view with 512 columns, 128 lanes, and 32-bit cells. That view contains 262,144 bytes of cells, but the ISA defines it per CTA and does not specify a physical “bytes per SM” capacity.

A TMEM address encodes a lane index and a column index; it is not a generic-memory pointer. `tcgen05.mma` writes its D matrix to TMEM. Depending on the instruction form, A may come from shared memory or TMEM, while B comes from shared memory.

## Allocation contract

- Allocation and deallocation are in columns. The unit is 32 columns; the count must be a power of two in the inclusive range 32–512.
- `tcgen05.alloc` and `tcgen05.dealloc` are warp-collective `.sync.aligned` instructions. Every active lane in the selected warp must execute with the same operands; a single-lane `threadIdx.x == 0` guard is invalid.
- `tcgen05.alloc` writes the allocated address to a shared-memory destination. Code must make that weak shared-memory store visible before other threads consume it.
- Every allocation must be explicitly deallocated before kernel exit. Repeated allocations within a CTA must not request a larger column count than an earlier allocation.
- `tcgen05.ld` and `tcgen05.st` are also warp-collective; a warp accesses its permitted lane slice, so a warpgroup is required to cover all 128 lanes.

The exact fence, proxy, and completion sequence depends on the operation. Use the PTX ISA or the corresponding CUTLASS/CuTe abstraction rather than a generic inline-assembly template.

## Primary references

- [PTX ISA: Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)
- [PTX ISA §9.7.17.7: allocation and management](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
- [CUTLASS CuTe DSL documentation compilation](../../sources/docs/cutlass-cute-dsl.md)
