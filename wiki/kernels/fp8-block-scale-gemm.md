---
id: kernel-fp8-block-scale-gemm
title: FP8 block-scale GEMM
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05, wgmma]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp]
related: [kernel-deepgemm, kernel-nvfp4-gemm, technique-fine-grained-quantization, hw-tcgen05-mma]
sources: [blog-deepgemm, pr-cutlass-2139, doc-cutlass-changelog-sm100]
performance_claims:
  - gpu: H800
    dtype: fp8
    shape: best reported benchmark; shape not specified in README news entry
    metric: TFLOPS
    value: 1550
    source_id: blog-deepgemm
    source_locator: https://github.com/deepseek-ai/DeepGEMM#news (2025-04-18 entry)
blackwell_relevance: CUTLASS and DeepGEMM provide SM100 block-scaled paths, but their scale formats and layouts are part of the API contract.
---

# FP8 block-scale GEMM

An FP8 block-scale GEMM multiplies low-precision operands while applying scale
metadata at a granularity finer than the full tensor. The exact granularity,
scale type, layout, promotion policy, and accumulator behavior belong to the
selected implementation; they must not be combined from different libraries.

CUTLASS PR 2139's Blackwell example wires scale layouts into its collective
builder. This contiguous excerpt is a reproducible configuration fragment:

```cpp
using ScaleConfig = decltype(
    cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100
>::CollectiveOp;
```

The full retained PR artifact is the executable source. The former raw-PTX block
was removed because it omitted the required instruction descriptor and did not
match the official block-scale operand form.

DeepGEMM's README reports up to 1550 TFLOPS on H800 without naming the shape in
that news entry. It remains an attributed maximum, not a portable expectation
for this kernel class.
