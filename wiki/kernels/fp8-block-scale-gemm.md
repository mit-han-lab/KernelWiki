---
id: kernel-fp8-block-scale-gemm
title: FP8 Fine-Grained-Scale GEMM
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05, wgmma]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, cute-dsl, python]
related: [kernel-deepgemm, kernel-nvfp4-gemm, technique-fine-grained-quantization, hw-tcgen05-mma]
sources: [blog-deepgemm, doc-deepseek-v3-fp8, doc-ptx-isa-sm100]
performance_claims: []
blackwell_relevance: SM100 accepts packed UE8M0 factors in block-scaled UMMA,
  replacing DeepGEMM's SM90 FP32-scale CUDA-core promotion path.
---

# FP8 Fine-Grained-Scale GEMM

## Scope

The DeepSeek-V3 training recipe quantizes forward activations per `1x128` tile and weights per `128x128` block. The smaller groups let each scale adapt to local outliers. This is a **logical quantization format**; a kernel is compatible only when its element type, scale granularity, scale representation, matrix layout, and architecture-specific ABI all match.

DeepGEMM commit [`891d57b4`](https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f) implements related FP8 GEMMs on SM90 and SM100, but the two generations handle scale factors differently.

## Logical Scale Shapes

For `A[M,K] @ B[N,K].T` with the DeepSeek-V3 forward recipe, the logical scale arrays have shapes `A_sf[M,K/128]` and `B_sf[N/128,K/128]`. This small check documents only that grouping; it does not encode DeepGEMM's required TMA-transformed layouts.

```python
# KernelWiki-derived format check; not upstream DeepGEMM code.
def deepseek_v3_scale_shapes(m: int, n: int, k: int):
    assert m % 128 == 0 and n % 128 == 0 and k % 128 == 0
    activation_scales = (m, k // 128)
    weight_scales = (n // 128, k // 128)
    return activation_scales, weight_scales

assert deepseek_v3_scale_shapes(4096, 4096, 4096) == ((4096, 32), (32, 32))
```

The paper documents phase-specific exceptions, including `128x1` activation grouping in backward paths. Do not treat the forward layout above as a universal FP8 tensor ABI.

## SM90: WGMMA Followed by CUDA-Core Promotion

The pinned SM90 1D1D kernel requires FP32 scale factors and fixes `BLOCK_K == 128`. Within each K block it issues the selected WGMMA operations into a register `accum` array; after the WGMMA batch completes, it multiplies those partials by the corresponding A/B scales and adds them into a separate FP32 `final_accum` array on CUDA cores.

For the DeepSeek-V3 description, a 128-element K interval corresponds to four WGMMAs. The paper describes Hopper's relevant internal addition/alignment precision as 14 bits, not as a generic “FP22 accumulator.” Promotion improves numerical behavior but still adds scale loads and CUDA-core work, so its overhead must be measured rather than assumed away.

## SM100: Packed UE8M0 and Block-Scaled UMMA

The pinned SM100 interface requires scale factors packed as four UE8M0 values per 32-bit `torch.int`. The 1D1D kernel TMA-loads scale-factor blocks, copies them into TMEM, builds a `make_instr_desc_block_scaled<..., float_ue8m0_t, ...>` descriptor, and accumulates with the selected block-scaled UMMA path in TMEM.

This is hardware-integrated scale consumption, not the SM90 `final_accum += scale_a * scale_b * accum` loop. Exact `tcgen05.mma` instruction spelling and descriptor restrictions are PTX-version-sensitive; use the pinned source wrapper or the NVIDIA PTX ISA rather than a hand-written approximate inline-assembly string.

The current DeepGEMM snapshot supports more than the original 128-granularity training recipe on SM100, including selected 32-element recipes. Treat the API's transformed/padded scale layout as authoritative for the chosen recipe.

## Source-Reported Performance

The pinned README says DeepGEMM achieved **up to 1550 TFLOPS on H800** in its 2025-04-18 news item. That sentence does not identify a matrix shape, utilization percentage, timing protocol, sample count, variance, or a single optimization responsible for the maximum. The result was not reproduced in this audit and is therefore kept out of structured `performance_claims`.

CUTLASS also ships SM100 block-scaled GEMM schedules, but no matched CUTLASS-versus-DeepGEMM shape/environment record is established here.

## Selection Checklist

- Confirm the producer's scale grouping; FP8 element type alone is insufficient for compatibility.
- On SM90, provide the FP32 scale layout expected by the selected DeepGEMM kernel.
- On SM100, provide correctly packed, TMA-aligned UE8M0 factors for the selected granularity and layout.
- Include quantization, scale-layout transformation, and epilogue costs when measuring an end-to-end path.
- Validate numerical error against an FP32/BF16 reference on the actual activation/weight distribution.

## Sources

- [DeepSeek-V3 Technical Report v2, FP8 training](https://arxiv.org/html/2412.19437v2)
- [DeepGEMM at audited commit `891d57b4`](https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
