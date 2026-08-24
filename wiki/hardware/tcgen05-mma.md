---
id: hw-tcgen05-mma
title: "tcgen05.mma — Fifth-Generation Tensor Core MMA"
type: hardware
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, mbarrier]
confidence: source-reported
related: [hw-tmem, hw-2sm-cooperative, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, blog-tcgen05-tutorial, blog-colfax-cutlass]
aliases: [UMMA, tcgen05, "tensor core gen 05"]
---

# tcgen05.mma

`tcgen05.mma` starts an asynchronous fifth-generation Tensor Core matrix multiply-accumulate. Its operation is `D = A × B + D`; an input predicate can instead select `D = A × B`.

## Operand and issue model

| Matrix | Permitted storage |
| --- | --- |
| A | Shared memory descriptor or TMEM address |
| B | Shared memory descriptor |
| D / accumulator | TMEM address |

For `cta_group::1`, one thread in the CTA issues the whole MMA. For `cta_group::2`, one thread from the CTA pair issues it while the peer CTA is active. This single-thread issue rule is specific to MMA-like operations; TMEM allocation, deallocation, loads, and stores have warp-collective rules.

The 32-bit `idesc` operand is mandatory. It encodes matrix dimensions, exact types, sparsity, and other operation details; it is not an optional scale descriptor. A representative unscaled form is:

```ptx
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc, enable_input_d;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];
```

The commit instruction provides an mbarrier-based completion path for earlier asynchronous `tcgen05.mma` operations issued by the same thread.

## Instruction kinds

The PTX ISA lists unscaled kinds `f16`, `tf32`, `f8f6f4`, and `i8`. Block-scaled forms use `mxf8f6f4`, `mxf4`, or `mxf4nvf4` with the `.block_scale` syntax and explicit TMEM scale-factor matrices. There is no standalone `kind::mxf8`. `mxf4nvf4` denotes support for MXFP4 and NVIDIA’s scaled 4-bit format; it does not prescribe MXFP4 for one operand and NVFP4 for the other.

## Shared-memory layouts

Matrix descriptors describe both addressing and swizzling. Supported layouts include non-swizzled and several swizzled forms, subject to each instruction’s shape and alignment constraints. Correctness requires the descriptor to match the physical shared-memory layout; 128-byte swizzling is a common high-throughput choice, not a universal requirement.

Exact shapes, packing, scale-vector sizes, sparse forms, and architecture targets vary by instruction kind and PTX version. Consult the PTX tables instead of extrapolating a single GEMM configuration.

## Performance evidence

The separate [tcgen05 tutorial source page](../../sources/blogs/tcgen05-tutorial.md) records one author’s staged B200 GEMM results. Those numbers characterize that implementation and benchmark, not the instruction in general.
