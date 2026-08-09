---
id: doc-triton-3.3-blackwell
title: "Triton v3.3.0 — Blackwell TCGen5/TMEM Boundary"
url: https://github.com/triton-lang/triton/compare/v3.2.0...v3.3.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem]
retrieved_at: 2026-08-08
---

# Triton v3.3.0 — Blackwell TCGen5/TMEM Boundary

## Exact comparison

The checked boundary is Triton v3.2.0 (`9641643da6c52000c807b5eeed05edaec4402a67`) to v3.3.0 (`819e9c8c29ad2ae96cbd93a1d3b8a3a0f4c8f09c`). The corresponding TCGen5 MMA, TMEM, and MMAv5-lowering symbols are absent from the v3.2.0 tree. The v3.3.0 tree adds:

- `TTNG_TCGen5MMAOp` and `TTNG_TCGen5MMAScaledOp`;
- TMEM allocation, load, store, and copy operations;
- MMAv5 lowering and tensor-memory allocation passes; and
- Blackwell conversion tests that require concrete `tcgen05.mma`, commit, TMEM, scaled-MMA, and `cta_group::2` output.

The pinned positive control is [`test/Conversion/tritongpu_to_llvm_blackwell.mlir`](https://github.com/triton-lang/triton/blob/819e9c8c29ad2ae96cbd93a1d3b8a3a0f4c8f09c/test/Conversion/tritongpu_to_llvm_blackwell.mlir). The operation definitions are in [`TritonNvidiaGPUOps.td`](https://github.com/triton-lang/triton/blob/819e9c8c29ad2ae96cbd93a1d3b8a3a0f4c8f09c/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td).

## Scope

This tag comparison establishes that native Blackwell TCGen5/TMEM compiler support enters between v3.2.0 and v3.3.0. It does not establish that every user-level `tl.dot` shape selects that path, or that every later frontend surface was already mature in v3.3.0.

Later releases extend the surface: v3.5.0 includes explicit Gluon TCGen5/TMEM and block-scaled matmul tutorials; v3.6.0 generalizes layouts and copies and advances warp-specialized and initial multi-CTA Gluon support; v3.7.0 continues 2-CTA, multicast, and TMA work. v3.7.1 is a regression-fix patch with no new API or feature.
