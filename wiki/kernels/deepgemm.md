---
id: kernel-deepgemm
title: DeepGEMM — runtime-JIT tensor-core kernels
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, fp4, fine-grained-quantization, block-scale, grouped-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, grouped-gemm]
languages: [cuda-cpp]
related: [technique-fine-grained-quantization, hw-tcgen05-mma, hw-nvfp4, kernel-grouped-gemm]
sources: [blog-deepgemm, pr-DeepGEMM-304]
performance_claims:
  - gpu: H800
    dtype: fp8
    shape: best reported benchmark; shape not specified in README news entry
    metric: TFLOPS
    value: 1550
    source_id: blog-deepgemm
    source_locator: https://github.com/deepseek-ai/DeepGEMM#news (2025-04-18 entry)
blackwell_relevance: The retained SM100 kernel builds a block-scaled UMMA descriptor, accumulates in TMEM, and uses packed UE8M0 scale layouts.
artifact_dir: artifacts/kernels/deepgemm
---

# DeepGEMM — runtime-JIT tensor-core kernels

DeepGEMM supplies dense and grouped tensor-core kernels for SM90 and SM100. Its
current README documents different layout and scale contracts on the two
architectures: SM90 uses FP32 scales and NT inputs, while SM100 uses packed
UE8M0 scales and supports four A/B layout combinations.

## Verbatim SM100 anchor

The retained artifact is pinned to DeepGEMM commit
`891d57b4db1071624b5c8fa0d1e51cb317fa709f`. This contiguous excerpt from
`sm100_fp8_gemm_1d1d.cuh` shows that the implementation builds a block-scaled
instruction descriptor and passes it, scale-factor TMEM columns, and operand
descriptors through a typed MMA wrapper:

```cpp
const auto& runtime_instr_desc =
    make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);
b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
    b_desc_base_lo, 0, k * UMMA_K);
mma_t::fma(a_desc, b_desc,
           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
           k_block_idx > 0 or k > 0, runtime_instr_desc,
           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
           kTmemStartColOfSFB);
```

This is not a standalone kernel. It deliberately replaces the former invalid
inline-PTX sketch, which omitted required descriptors and used the wrong MMA
kind for block scaling.

## Grouped layouts and JIT boundary

The public API distinguishes contiguous and masked M-grouped layouts plus a
K-grouped weight-gradient API. Those layouts have different shape/alignment
contracts; “grouped GEMM” is not one generic pointer-array ABI. Current
DeepGEMM compiles kernels at runtime through its JIT module, with NVCC as the
documented default and NVRTC as an opt-in path.

## Performance boundary

The project README reports up to 1550 TFLOPS on H800 in its 2025-04-18 news
entry. Because that line does not specify a matrix shape, this page preserves it
only as a source-reported maximum with an explicit unknown-shape boundary.

The exact retained files and hashes are recorded in
`artifacts/kernels/deepgemm/full/PROVENANCE.yaml`. The former derived
“Nc=128” teaching file was removed because it presented illustrative code as if
it were the library implementation.
