---
id: kernel-gated-dual-gemm
title: Gated Dual GEMM (Gate-Up + Activation)
type: kernel
architectures: [sm100, sm90]
tags: [gated-dual-gemm, gemm, fused-kernel, kernel-fusion, nvfp4, tmem]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, cute-dsl]
related: [kernel-nvfp4-gemm, kernel-fused-moe, technique-kernel-fusion, technique-epilogue-fusion]
sources: [contest-gpumode-p3, blog-deepgemm, blog-tflops-gap-fp4-moe, pr-TensorRT-LLM-11897]
performance_claims: []
blackwell_relevance: Blackwell kernels can hold matrix accumulators in TMEM and fuse their register-side epilogue, subject to the selected tile and TMEM budget.
---

# Gated Dual GEMM

A gated dual GEMM evaluates two projections of the same input and combines them through a gated activation, for example `SiLU(X·W_gate) * (X·W_up)`. Reusing the input tile and fusing the elementwise combination can avoid materializing both projections in global memory.

## Implementation questions

- whether the two projections use one combined output tile or separate accumulator regions;
- how A and both weight tiles are staged and synchronized;
- whether block scales are shared, independent, or applied in the epilogue;
- how the epilogue drains accumulator fragments without exceeding register or TMEM capacity;
- whether fusing both projections improves traffic enough to offset added live state.

## Evidence boundary

GPU Mode problem 3 defines the workload and its live organizer page records the benchmark context. The official evidence retained here does not support the earlier local latency or standalone speedup claims, so `performance_claims` is empty.

The previous artifact bundle was removed: its anchor was a vLLM test-only PR excluded by the kernel-source policy, and its remaining files were a blog extract and a synthetic skeleton rather than one upstream implementation.

TensorRT-LLM PR 11897 supplies a concrete retained implementation boundary: an
NVFP4 dense GEMM fused with SwiGLU. The following contiguous added-lines excerpt
is the BF16-output call site in
`tensorrt_llm/_torch/modules/gated_mlp.py`, not a standalone kernel:

```python
output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
    act_fp4, module.weight, act_sf, module.weight_scale, alpha,
    module.dtype)
```
