---
id: technique-fine-grained-quantization
title: "Fine-Grained FP8/FP4 Quantization"
type: technique
architectures: [sm100, sm90]
tags: [fine-grained-quantization, fp8, fp4, nvfp4, block-scale]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-nvfp4]
related: [hw-nvfp4, kernel-deepgemm]
sources: [blog-deepgemm, doc-ptx-isa-sm100, pr-vllm-23696]
blackwell_relevance: "SM100 tcgen05 supports multiple block-scaled MMA kinds: MX formats use UE8M0 scales, while NVFP4 uses UE4M3 scales with its documented vector size and layout."
---

# Fine-Grained FP8/FP4 Quantization

Fine-grained quantization partitions an operand into scale groups instead of applying one scale to an entire tensor. A smaller group can track local ranges more closely, but it consumes more scale storage and bandwidth and imposes a more detailed layout contract. It does not guarantee lower error for every distribution or higher end-to-end performance.

```python
def dequantized_dot(a_codes, b_codes, a_scales, b_scales, vector_size):
    """Logical reference only; physical scale layouts are kernel-specific."""
    total = 0.0
    for k, (a_code, b_code) in enumerate(zip(a_codes, b_codes)):
        group = k // vector_size
        a = decode(a_code) * decode_scale(a_scales[group])
        b = decode(b_code) * decode_scale(b_scales[group])
        total += a * b
    return total
```

## Keep the format contracts separate

| Contract | Data and scale format | Scale granularity established by the cited source |
|---|---|---|
| SM100 NVFP4 MMA | E2M1 data with UE4M3 scales | 16 dense K elements per scale; 32 for the documented sparse form |
| SM100 MXFP4 MMA | E2M1 data with UE8M0 scales | 32 K elements per scale |
| Current DeepGEMM SM100 FP8 path | FP8 data with packed UE8M0 scales | library-specific TMA-aligned/transposed layout |
| Current DeepGEMM SM90 FP8 path | FP8 data with FP32 scales | checked kernel uses 128-element K blocks and applies scales during CUDA-core promotion |

PTX represents the SM100 scale-factor matrices through prescribed TMEM layouts and descriptors. They are not arbitrary pointers to row-major scale arrays. CUTLASS and DeepGEMM provide layout/packing helpers; callers should use the interface for the exact kernel version.

## DeepGEMM boundary

The checked DeepGEMM README says SM90 requires FP32 scale factors while SM100 requires four packed UE8M0 values per `torch.int`. The pinned SM90 implementation statically requires `BLOCK_K == 128`, performs its WGMMA work for that block, then multiplies by the A/B FP32 scales while accumulating into `final_accum`. This supports a per-128-K implementation statement; it does not support the former universal “FP22,” 0.1%-error, or fixed four-instruction justification.

The pinned SM100 implementation constructs a block-scaled UMMA instruction descriptor with `cutlass::float_ue8m0_t`. That is evidence for this DeepGEMM FP8 path—not evidence that every SM100 block-scaled operation uses UE8M0. In particular, PTX assigns UE4M3 to NVFP4.

## Application-level scaling

Some quantization recipes add a tensor-level FP32 multiplier outside the MMA's native block-scale representation. Keep that software-level factor separate in the numerical reference and apply it at the location required by the calling framework. Do not infer such a factor from `tcgen05.mma` itself.

## Verification checklist

- Record dtype, scale type, vector size, physical scale layout, and K-tail constraints.
- Compare against a dequantize-then-matmul reference with the same rounding and output dtype.
- Test zero scales, extreme scale exponents, clipping, nonmultiple tails, and layout transforms.
- Attribute performance only to a pinned kernel, shape, software revision, and measurement method.

No standalone device pseudo-kernel is provided here: the exact instruction signature and scale layout are configuration-dependent, and the previous examples mixed NVFP4, MXFP4, and DeepGEMM contracts.
