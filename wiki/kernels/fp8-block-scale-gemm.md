---
id: kernel-fp8-block-scale-gemm
title: "FP8 Block-Scale GEMM"
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05, wgmma]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, cute-dsl]
related: [kernel-deepgemm, kernel-nvfp4-gemm, technique-fine-grained-quantization, hw-tcgen05-mma]
sources: [blog-deepgemm, doc-cutlass-blackwell, doc-cutlass-changelog-sm100, doc-ptx-isa-sm100, pr-cutlass-2139]
performance_claims:
  - gpu: H800
    software: "DeepGEMM README news item dated 2025-04-18; CUDA and library versions not stated"
    dtype: fp8
    shape: unspecified peak case in the 2025-04-18 DeepGEMM README news item
    workload: "FP8 GEMM peak headline; exact operation variant not stated"
    metric: TFLOPS
    value: 1550
    measurement_method: "source-reported project benchmark; method not stated in the headline"
    baseline: "none; absolute throughput headline"
    limitations: "shape, clocks, CUDA version, and measurement method are not stated in the retained headline"
    source_id: blog-deepgemm
blackwell_relevance: "SM100 block-scaled tcgen05 paths use layout-defined scale factors and TMEM accumulation; SM90 libraries implement fine-grained scaling with different MMA and scale handling."
---

# FP8 Block-Scale GEMM

## Operation

A block-scaled GEMM associates scale tensors with groups of quantized operand elements. A simple logical form is:

```python
def scaled_product(a_q, a_sf, b_q, b_sf, i, j, k_size, vec_size):
    total = 0.0
    for k in range(k_size):
        a = decode_fp8(a_q[i, k]) * a_sf[i, k // vec_size]
        b = decode_fp8(b_q[j, k]) * b_sf[j, k // vec_size]
        total += a * b
    return total
```

DeepGEMM's historically described recipe uses 1-by-128 activation and 128-by-128 weight granularity. CUTLASS supports blockwise/groupwise configurations beyond that pair, so the exact scale granularity is an API/kernel property.

## SM90 and SM100 are different implementations

SM90 WGMMA paths use register accumulator fragments and library-defined scale/promotion strategies. DeepGEMM documents CUDA-core promotion for its Hopper FP8 implementation; that is not a universal WGMMA instruction rule.

SM100 block-scaled paths use `tcgen05.mma` kinds such as `mxf8f6f4`, TMEM scale-factor layouts/IDs, and TMEM destinations. A syntactically shortened MMA with raw “scale descriptors” is not sufficient: the instruction descriptor encodes shapes, types, major orders, scale IDs/vector size, and accumulation behavior.

## Layout

Optimized scale tensors use prescribed physical layouts and alignment. DeepGEMM's current SM100 FP8 interface requires packed UE8M0 scales and provides layout-transform utilities; CUTLASS provides its own scale-layout builders. “FP32 or E4M3 or UE8M0” cannot be interchanged without selecting a compatible kernel kind.

## Precision

TMEM is the destination storage, not a statement that every internal accumulation step has unrestricted FP32 precision. Numerical behavior follows the selected MMA kind/accumulator type and any promotion scheme. Validate error against the intended reference and distribution.

## Performance boundary

DeepGEMM's README reports a historical peak of up to 1,550 TFLOP/s on H800. It does not tie that headline to 4096-by-4096-by-4096 or a universal 90% utilization value. CUTLASS SM100 performance must be reported for a named kernel, shape, build, and GPU rather than as a “similar ratio.”
