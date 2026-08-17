---
id: hw-nvfp4
title: "NVFP4 and Block-Scaled Narrow Precision"
type: hardware
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, block-scale, fp8, fp6]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
related: [technique-fine-grained-quantization, kernel-nvfp4-gemm, kernel-nvfp4-gemv, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, doc-cutlass-blackwell, doc-nvidia-tuning-guide, contest-gpumode-p1, contest-gpumode-p2, blog-yue-nvfp4, pr-cutlass-2139]
aliases: [NVFP4, E2M1, "FP4 E2M1", "nv_float4"]
---

# NVFP4 and Block-Scaled Narrow Precision

## Format

The SM100 NVFP4 operand pairs signed E2M1 data with an unsigned E4M3 scale-factor type (`ue4m3` in PTX/CUTLASS). One scale applies to 16 consecutive dense K elements (32 for the documented sparse case). E2M1 represents zero and signed magnitudes 0.5, 1, 1.5, 2, 3, 4, and 6; it has no infinity or NaN encoding.

Applications often add a higher-level FP32 tensor/global scale as part of their quantization recipe. That extra scale is not itself part of the `tcgen05` NVFP4 instruction format.

## Hardware operation

The block-scaled MMA computes the equivalent of:

```text
D[i,j] = C[i,j] + sum_k
    (decode(A[i,k]) * SFA[i,floor(k/16)]) *
    (decode(B[j,k]) * SFB[j,floor(k/16)])
```

Scale tensors use a prescribed swizzled/basic-block physical layout. Use CUTLASS layout builders or the PTX tables rather than assuming a row-major matrix.

The associated PTX kind is `mxf4nvf4` (some CUTLASS text/API generations use `nvf4mxf4`). CUTLASS characterizes its peak tensor-core class as 4x Hopper FP8 Tensor Core throughput; this is a theoretical instruction-class comparison, not a guarantee that an application is 4x faster.

## NVFP4 versus MXFP4

| Property | NVFP4 | MXFP4 |
|---|---|---|
| data | E2M1 | E2M1 |
| scale | UE4M3 | UE8M0 |
| dense K elements per scale | 16 | 32 |
| OCP MX compliant | no | yes |

The finer, fractional NVFP4 scale can reduce quantization error for many distributions, but “always lower error” is not an architecture fact; clipping, scale selection, and data distribution determine the result.

## Conversion boundary

PTX supports packed E2M1 conversion forms, with saturation/rounding qualifiers determined by direction and destination. Consult the exact `cvt` signature; a generic two-operand mnemonic without required qualifiers is not a portable code sample.
