---
id: blog-colfax-cutlass
title: 'Colfax CUTLASS Tutorial: GEMM Kernels Using Tensor Memory for Blackwell'
author: Colfax Research
url: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tmem-for-nvidia-blackwell-gpus/
source_category: community-note
architectures:
- sm100
tags:
- tcgen05
- tmem
- cute-dsl
- warp-specialization
- 2sm-cooperative
retrieved_at: 2026-04-16
artifact_dir: artifacts/blogs/colfax-cutlass-blackwell/code
---

## Summary

Detailed tutorial on CUTLASS abstraction for Blackwell UMMA (tcgen05.mma) with sub-byte GEMM support.

## Key Content
- UMMA replaces WGMMA: register-free operation, single-thread launch, built-in block scaling
- TMEM: 512 columns × 128 rows of 32-bit cells (256KB/SM)
- 32-bit addressing: bits 31-16 = lane ID, bits 15-0 = column
- CUTLASS two-level abstraction: MMA_Atom (PTX wrapper) + MMA_Traits (CuTe layouts)
- Architectural progression: Volta → Hopper TMA → Blackwell TMEM+UMMA
- Sub-byte GEMM tutorial covering NVFP4, MXFP4, block scaling

## Verified implementation notes

The article is useful for understanding CUTLASS's `MMA_Atom`/`MMA_Traits`
layering and TMEM-backed accumulators, but abbreviated inline PTX is not a safe
substitute for the ISA grammar:

- `tcgen05.alloc` and `tcgen05.dealloc` are warp-collective for
  `cta_group::1`; they are not lane-0-only instructions.
- `tcgen05.mma` is issued by one thread, but its full operand list includes a
  disable-output-lane vector whose width depends on the CTA group.
- `tcgen05.ld` is warp-collective and asynchronous; use its documented shape,
  repetition, register mapping, and completion mechanism.
- CUTLASS supplies exact wrapper types and layouts for supported combinations;
  choose the atom from the version-pinned library rather than reconstructing a
  template signature from prose.

For executable references, use the [PTX ISA 9.0 tcgen05 sections](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma)
and the [CUTLASS 4.5.0 SM100 examples](https://github.com/NVIDIA/cutlass/tree/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/70_blackwell_gemm).
