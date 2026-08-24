---
id: migration-register-to-tmem
title: Register Accumulators to TMEM
type: migration
from_arch: sm90
to_arch: sm100
tags: [tmem, tcgen05]
related: [hw-tmem, hw-tcgen05-mma, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, doc-cutlass-cute-dsl, blog-colfax-cutlass]
blackwell_relevance: Hopper WGMMA exposes register-resident accumulators; Blackwell tcgen05 uses Tensor Memory for its accumulator.
confidence: source-reported
reproducibility: pseudocode
---

# Register Accumulators to TMEM

## Migration boundary

Hopper WGMMA and Blackwell `tcgen05` do not expose interchangeable accumulator fragments. Hopper WGMMA accumulates into registers owned by participating threads. Blackwell `tcgen05` places the accumulator in Tensor Memory (TMEM), and epilogue threads explicitly copy the values they need from TMEM into registers.

That changes the kernel around the MMA, not just the instruction name:

| Concern | Hopper WGMMA | Blackwell `tcgen05` |
|---|---|---|
| Accumulator location | Registers | TMEM |
| MMA issuer | Warpgroup instruction model | A designated thread issues a CTA- or two-CTA-group operation |
| Common A/B path | Shared-memory descriptors for WGMMA | Shared-memory descriptors for `tcgen05`; supported forms may source A from TMEM |
| Result consumption | Register fragment already present | Explicit TMEM-to-register tiled copy |
| Lifetime | Register-fragment lifetime | Explicit TMEM allocation and deallocation |

## Practical porting sequence

1. Select the supported `tcgen05` operation and instruction shape for the input and accumulator types.
2. Build the corresponding shared-memory layouts and descriptors for A and B.
3. Create the TMEM accumulator layout and include all live accumulator stages in its column budget.
4. Allocate TMEM using the documented CTA-group protocol.
5. Replace WGMMA fence/commit/wait logic with the synchronization protocol required by the selected `tcgen05` and pipeline abstractions.
6. Partition a TMEM-to-register copy for the epilogue instead of treating C as a register fragment.
7. Deallocate TMEM only after MMA and epilogue users have completed.

CUTLASS/CuTe expresses these steps through a tiled MMA, `TmemAllocator`, TMEM fragment binding, pipeline barriers, and tiled-copy partitions. This is safer than translating an old WGMMA inline-PTX wrapper instruction by instruction.

## What must be re-tuned

Moving the accumulator out of registers can relieve one source of register pressure, but it does not prove a particular occupancy or speedup. CTA shape, TMEM columns, shared memory, epilogue registers, synchronization, and work scheduling remain kernel-specific constraints. Re-measure those resources and performance after the port.

## Primary references

- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [CUTLASS CuTe DSL `tcgen05` programming guide](../../sources/docs/cutlass-cute-dsl.md)
- [Tensor Memory](../hardware/tmem.md)
