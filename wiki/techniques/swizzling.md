---
id: technique-swizzling
title: "Shared Memory Swizzling"
type: technique
architectures: [sm100, sm90]
tags: [swizzling, shared-memory-optimization, tma]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tma]
related: [hw-tma, technique-pipeline-stages, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, blog-tcgen05-tutorial]
blackwell_relevance: "The physical shared-memory layout and its matrix/TMA descriptor must carry the same swizzle; the best supported mode depends on the tile and element layout."
---

# Shared Memory Swizzling

A shared-memory swizzle permutes address bits so that a cooperative access pattern spreads requests across banks. It can reduce bank conflicts, but the correct mapping depends on element size, matrix major mode, instruction shape, and transfer pattern.

For descriptor-based operations, the encoded descriptor and the physical shared-memory layout must agree. PTX represents non-swizzled and multiple swizzled layouts. A 128-byte swizzle is therefore neither universally mandatory nor universally optimal for `tcgen05` inputs.

## CuTe layout coupling

The official CuTe DSL dense-GEMM documentation materializes a logical layout and its selected swizzle together. This excerpt is a fragment from the compiled `dense_gemm.py` documentation:

```python
sA = smem.allocate_tensor(
    a_smem_layout.outer, swizzle=a_smem_layout.inner, dtype=ab_dtype
)
sB = smem.allocate_tensor(
    b_smem_layout.outer, swizzle=b_smem_layout.inner, dtype=ab_dtype
)
```

The helper that produced `a_smem_layout` and `b_smem_layout` chooses among supported atoms using the actual major dimension and divisibility constraints. Hand-written PTX must enforce the equivalent agreement explicitly.

## Verification

- Validate the descriptor/layout pair against the instruction’s PTX layout tables.
- Check alignment and leading-stride constraints for the chosen mode.
- Measure bank conflicts and throughput for the actual access pattern; a larger swizzle span is not automatically faster.
- Scope benchmark conclusions. The `tcgen05 for dummies` article reports a large gain when changing several layout/transfer choices in one B200 GEMM, not a universal penalty for every other mode.
