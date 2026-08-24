---
id: kernel-grouped-gemm
title: Grouped GEMM for MoE
type: kernel
architectures: [sm100, sm90]
tags: [grouped-gemm, moe, gemm, fp8, nvfp4, persistent-kernel, tile-scheduling]
confidence: source-reported
reproducibility: snippet
kernel_types: [grouped-gemm, gemm, moe]
languages: [cuda-cpp]
related: [kernel-fused-moe, kernel-deepgemm, technique-persistent-kernels, technique-tile-scheduling]
sources: [contest-gpumode-p4, blog-deepgemm, pr-DeepGEMM-304, blog-gpu-mode-reward-hack]
performance_claims: []
blackwell_relevance: SM100 implementations can use native block-scaled MMA and either static persistence or CLC-assisted work acquisition; the scheduler is an implementation choice.
---

# Grouped GEMM for MoE

Grouped GEMM executes a collection of matrix products in one dispatch. In an
MoE M-grouped layout, the expert dimension identifies the group and each expert
may have a different valid row count while N and K are shared by the model
configuration. This avoids launching one kernel per expert, but it does not
remove small-M inefficiency or unequal expert loads.

## Retained implementation boundary

DeepGEMM PR 304 exposes separate contiguous and masked M-grouped entry points.
Its host dispatcher selects an architecture-specific implementation. This is a
contiguous excerpt from the retained `csrc/apis/gemm.hpp`:

```cpp
if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
    sm90_m_grouped_fp8_gemm_contiguous_1d2d(
        a.first, sfa, b.first, sfb, d, grouped_layout, num_groups, m, n, k,
        major_a, major_b, major_sfb, compiled_dims, use_psum_layout,
        expected_m_for_psum_layout);
} else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
    sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(
        a.first, sfa, b.first, sfb, d, grouped_layout, num_groups, m, n, k,
        gran_k_a, gran_k_b, major_a, major_b, compiled_dims,
        use_psum_layout, expected_m_for_psum_layout);
}
```

The excerpt is a dispatcher, not the device mainloop. It establishes that the
retained code has distinct SM90 and SM100 paths and different scale
representations; it does not justify a hand-written generic tcgen instruction.

## Layout and scheduling choices

- A contiguous layout packs valid rows and carries a grouping layout that maps
  rows back to experts.
- A masked layout keeps a fixed allocation and supplies each expert's valid M,
  which is useful when stable shapes matter.
- A static persistent scheduler computes a deterministic tile sequence.
- A CLC scheduler on Blackwell may acquire IDs of same-grid clusters canceled
  before launch. CLC is optional and does not guarantee that variable expert
  work becomes balanced.

Correctness tests should cover empty experts, nonuniform row counts, padding,
scale layouts, output offsets, and both architecture dispatch branches. Measure
the whole routed workload; a grouped kernel can reduce launch overhead while
remaining dominated by thin expert GEMMs.

## Benchmark boundary

The GPU Mode problem-4 reward-hack result is not a valid kernel benchmark: the
submission exploited object reuse between correctness and timing. This page
therefore carries no performance claim from that result.
