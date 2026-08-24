---
id: technique-fine-grained-quantization
title: Fine-grained FP8/FP4 scaling
type: technique
architectures: [sm100, sm90]
tags: [fine-grained-quantization, fp8, fp4, nvfp4, block-scale]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-nvfp4]
related: [hw-nvfp4, kernel-deepgemm, kernel-fp8-block-scale-gemm]
sources: [blog-deepgemm, doc-ptx-isa-sm100, pr-DeepGEMM-304]
blackwell_relevance: SM100 block-scaled MMA consumes scale metadata through instruction- and layout-specific descriptors; DeepGEMM's SM100 API uses packed UE8M0 scales.
---

# Fine-grained FP8/FP4 scaling

Fine-grained quantization associates scales with smaller regions than an entire
tensor. This can limit the influence of an outlier, at the cost of additional
scale storage, layout constraints, conversion, and loads. “Block scale” is not
one universal format: DeepGEMM's SM100 FP8 path uses packed UE8M0, while the
retained NVFP4 contest uses per-16 FP8 scales. The contest task prose calls
them E4M3FNUZ, while its executable reference constructs them as
`torch.float8_e4m3fn`; the source artifacts disagree on the encoding name.

## Retained implementation anchor

DeepGEMM's captured SM100 kernel constructs a block-scaled descriptor and
selects runtime scale-factor IDs before issuing its typed MMA wrapper:

```cpp
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
    UMMA_M, UMMA_N, kMajorA, kMajorB>();
const auto& runtime_instr_desc =
    make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);
mma_t::fma(a_desc, b_desc,
           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
           k_block_idx > 0 or k > 0, runtime_instr_desc,
           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
           kTmemStartColOfSFB);
```

This contiguous excerpt still depends on the surrounding template; use the full
pinned artifact for executable code. It shows the required relationship among
type, layout, runtime instruction descriptor, and scale locations without
inventing an inline-PTX signature.

## Porting rules

- Preserve the source format and granularity; UE8M0, E4M3, and FP32 scales are
  not interchangeable.
- Use the library's layout transformer/query rather than deriving scale strides
  from prose.
- Treat promotion intervals on SM90 as kernel-specific numerical policy, not a
  universal Hopper requirement.
- Validate error and performance for the complete quantize–GEMM–dequantize
  path, including tails and exceptional values.
