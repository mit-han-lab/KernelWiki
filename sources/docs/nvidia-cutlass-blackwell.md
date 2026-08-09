---
id: doc-cutlass-blackwell
title: "NVIDIA CUTLASS 4.5.0 Blackwell Sources"
url: https://github.com/NVIDIA/cutlass/tree/v4.5.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, nvfp4, block-scale, cute-dsl]
version: "4.5.0"
retrieved_at: 2026-08-09
---

# NVIDIA CUTLASS 4.5.0 Blackwell Sources

## Evidence scope

This card routes version-sensitive CUTLASS claims to tag `v4.5.0`, commit `e406c186f510a15091cce01f782020ceb7ba8eb5`. CUTLASS provides both C++ template APIs and Python-native DSLs; this card does not rank one as the primary Blackwell interface. Rolling `latest` documentation can describe later APIs and is not evidence for an exact 4.5.0 symbol.

## Verified CuTe DSL loci

- `python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py` defines typed tcgen05 operations including `MmaF16BF16Op`, `CtaGroup`, operand sources, and traits.
- `python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py` defines TMEM load, store, and shared-to-TMEM copy operations.
- `python/CuTeDSL/cutlass/cute/arch/tmem.py` defines allocation, pointer retrieval, permit relinquishment, and deallocation helpers.
- `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` through `fp16_gemm_6.py` are the tagged progressive dense-GEMM tutorials.
- The tutorial constructs TMA atoms with `CopyBulkTensorTileG2SOp` plus `make_tiled_tma_atom_A/B`; it does not define `SM100_TMA_LOAD_2D`.
- The tutorial uses `TmemAllocator`, typed copy atoms, `make_tmem_copy`, and explicit pipeline state. Short snippets that omit those lifetimes are not standalone kernel recipes.

## Scope limits

The tag supplies layout algebra and Blackwell helper functions, but users still choose datatypes, instruction/CTA shapes, operand modes, tile layouts, alignment, swizzles, participant groups, and pipeline policy. The source tree alone does not establish a universal runtime-performance percentage.

## Direct links

- [CUTLASS v4.5.0 release](https://github.com/NVIDIA/cutlass/releases/tag/v4.5.0)
- [Pinned CuTe DSL tcgen05 package](https://github.com/NVIDIA/cutlass/tree/e406c186f510a15091cce01f782020ceb7ba8eb5/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05)
- [Pinned TMEM helpers](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/python/CuTeDSL/cutlass/cute/arch/tmem.py)
- [Pinned progressive GEMM tutorial](https://github.com/NVIDIA/cutlass/tree/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm)
