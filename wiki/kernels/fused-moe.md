---
id: kernel-fused-moe
title: Fused MoE — Expert GEMM and Adjacent Operations
type: kernel
architectures: [sm100, sm100a, sm90]
tags: [moe, fused-kernel, fp8, block-scale, kernel-fusion, grouped-gemm, gated-dual-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, fused-kernel, grouped-gemm, gated-dual-gemm]
languages: [cuda-cpp, cute-dsl, triton]
related: [kernel-grouped-gemm, kernel-deepgemm, technique-fine-grained-quantization, technique-tile-scheduling]
sources: [contest-flashinfer-track-a, blog-deepgemm, pr-TensorRT-LLM-11897]
performance_claims: []
blackwell_relevance: Blackwell block-scaled tensor-core operations and TMEM are implementation tools for expert GEMMs; the useful fusion boundary remains workload-specific.
---

# Fused MoE

“Fused MoE” covers kernels that combine an expert GEMM with adjacent data preparation, activation, quantization, or result-combination work. It does not imply that routing, dispatch, both expert projections, and combine are always one device kernel.

## Kernel boundary

A concrete implementation should state which of these operations are inside the launch:

- routed-row preparation or permutation;
- grouped gate/up expert GEMM;
- gated activation;
- intermediate quantization;
- grouped down-projection GEMM;
- weighted output combination.

The profitable boundary depends on batch/routed-row distribution, data type, intermediate traffic, code size, and the available backend. Correctness must cover empty experts, repeated expert IDs, padding, routing weights, and quantization scales.

## Evidence boundary

The FlashInfer contest page defines a fused-MoE track but does not publish the earlier local performance table. DeepGEMM documents grouped expert GEMM primitives, not a universal end-to-end fusion speedup. Consequently this page carries no numeric performance claim.

The previous artifact directory mixed a vLLM test-only PR, an SGLang dispatcher file, a blog extract, and a synthetic routing skeleton. It was removed because it was not a coherent fused-MoE kernel implementation and linked to excluded source PRs.

One retained, narrower example is TensorRT-LLM PR 11897's shared-expert path.
Its BF16-output call site invokes a dense NVFP4 GEMM fused with SwiGLU:

```python
output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
    act_fp4, module.weight, act_sf, module.weight_scale, alpha,
    module.dtype)
```

This contiguous upstream excerpt demonstrates one expert-projection fusion. It
does not imply that routing, dispatch, both projections, and combine are in the
same device kernel.
