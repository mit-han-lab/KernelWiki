---
id: doc-cutlass-blackwell
title: NVIDIA CUTLASS 4.x Blackwell support
url: https://github.com/NVIDIA/cutlass/tree/v4.5.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, 2sm-cooperative, nvfp4, fp8, fp4, fp6, block-scale, cute-dsl]
retrieved_at: 2026-08-16
---

# CUTLASS 4.x Blackwell support

This record is pinned to CUTLASS 4.5.0, the latest stable release before the repository's 2026-05-20 refresh cutoff. CUTLASS exposes SM100 C++ and CuTe DSL building blocks for `tcgen05` MMA, TMEM, TMA, block-scaled datatypes, persistent scheduling, grouped GEMM, and attention.

## Source-backed boundaries

- `tcgen05` MMA writes its destination to TMEM, but supported A operand forms include shared memory and, for documented forms, TMEM. “Register-free” is therefore too broad a description of an entire kernel.
- One thread issues an MMA, while data preparation, collective loads/stores, and TMEM transfers still involve the participation required by their exact instruction or CUTLASS atom.
- One-CTA and two-CTA schedules are distinct configurations. Their legal instruction and tile shapes depend on datatype, MMA kind, and selected traits.
- NVFP4 uses E2M1 data with UE4M3 block scales; MXFP4 uses UE8M0. A generic E4M3 CUTLASS element alias does not by itself prove a valid NVFP4 scale layout.
- Warp roles, epilogue thread count, TMEM columns, pipeline stages, and swizzles are schedule-derived, not universal SM100 constants.

## Reproducibility

Use exact headers and examples from the pinned tag. The former local page included illustrative schedule/atom/epilogue type names that were not verified as public CUTLASS 4.5.0 APIs and has removed them.

Primary sources: [CUTLASS 4.5.0 tree](https://github.com/NVIDIA/cutlass/tree/v4.5.0), [4.5.0 changelog](https://github.com/NVIDIA/cutlass/blob/v4.5.0/CHANGELOG.md), and [CuTe DSL documentation](cutlass-cute-dsl.md).
