---
id: doc-triton-3.6-blackwell
title: "Triton 3.6 Release Notes — Blackwell Backend Work"
url: https://github.com/triton-lang/triton/releases/tag/v3.6.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, 2sm-cooperative, block-scale, nvfp4, warp-specialization]
retrieved_at: 2026-08-18
---

# Triton 3.6 release notes: Blackwell items

The official Triton 3.6 release notes enumerate substantial Blackwell backend and Gluon work.

## Items stated in the release notes

- TMEM encoding/layout construction and broadcasting changes (`#8136`, `#8148`, `#8202`).
- Generic `tcgen05.cp` lowering and distributed layouts for `tcgen05.ld/st` (`#8225`, `#8421`, `#8495`).
- Generalization and verification work around fifth-generation MMA operations.
- Warp-specialization compiler and documentation changes.
- Gluon multi-CTA support and initial two-CTA mode support (`#8644`, `#8653`), plus cluster-sizing emission (`#8645`).
- Gluon `tcgen05 mma scaled` support (`#8393`) and `dot_scaled` frontend fixes (`#8564`, `#8658`).

These entries establish that Triton 3.6 contains Blackwell/TMEM/`tcgen05` infrastructure. They do not establish that every ordinary `tl.dot` shape uses the same instruction form, or that a Triton kernel has performance parity with a hand-tuned CUDA/CuTe implementation.

## Downstream evidence retained in this repository

- `pr-vllm-34597` contains a real `@triton.jit` MLA decode-attention kernel with `tl.dot` operations and an SM100 architecture assignment grounded in its PR evidence.
- `pr-sglang-22079` contains an SM100/SM90 Triton attention path with `tl.dot` operations.
- `pr-sglang-21019` contains a Triton GatedDeltaNet data-rearrangement kernel; it demonstrates downstream SM100 use but not tensor-core lowering.

These downstream pages demonstrate adoption, while the release notes remain the authority for compiler capability. No blanket lowering or speedup inference is made from their combination.
