---
id: kernel-tensorrt-llm-blackwell-indexer
title: TensorRT-LLM Blackwell FP4 DSA Indexer
type: kernel
architectures:
- sm100
tags:
- attention
- gemm
- fp4
- kernel-fusion
- top-k-selection
- vectorized-loads
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- gemm
- topk
languages:
- cuda-cpp
- python
related:
- kernel-fused-moe
- kernel-fp8-block-scale-gemm
- technique-vectorized-loads
- technique-fine-grained-quantization
sources:
- pr-TensorRT-LLM-13340
performance_claims: []
blackwell_relevance: TensorRT-LLM's Blackwell DSA indexer PR is a current upstream implementation reference for FP4/FP8 cache indexer paths and fused quantized gather/scatter kernels.
---

## Shape

TensorRT-LLM PR 13340 integrates an FP4 indexer path for DSA on Blackwell and
lands CUDA kernels for K-cache gather/scatter and fused FP4 concatenation. Use it
as implementation evidence for sparse indexer memory movement and quantized
cache layout, not as a drop-in answer for FlashInfer-Bench.

The captured PR's gather wrapper distinguishes FP8 and packed-FP4 byte widths.
This contiguous added-lines excerpt is an input/launch contract, not the top-k
algorithm:

```cpp
constexpr int32_t VEC_SIZE = 4;
TLLM_CHECK_WITH_INFO(head_dim == 128 || head_dim == 64,
    "head_dim must be 128 (FP8) or 64 (FP4 packed) for the indexer cache (got %d)", head_dim);
TLLM_CHECK_WITH_INFO(scale_size == 4,
    "scale_size must equal 4 bytes (packed UE8M0 x4 for FP4, 1 float32 for FP8, got %d)", scale_size);
int32_t const threads_per_block = head_dim / VEC_SIZE;
```

## Transfer Notes

- Keep FP4 packing, scale placement, and invalid-token handling explicit in the
  correctness reference.
- Treat gather/scatter traffic as a separate NCU profile target.
- Avoid merging this with top-k selection until the memory path is understood.

The former `candidate_indexer_probe` block was removed because it was locally
invented and did not represent any kernel in PR 13340.
