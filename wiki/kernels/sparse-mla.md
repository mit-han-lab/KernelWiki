---
id: kernel-sparse-mla
title: "Sparse MLA (DeepSeek V3.2)"
type: kernel
architectures: [sm100, sm90]
tags: [sparse-attention, mla, fp8, attention, decode, prefill]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, mla, attention, decode, prefill]
languages: [cuda-cpp, cute-dsl]
related: [kernel-flashmla, kernel-nsa, hw-tcgen05-mma]
sources: [blog-flashmla, blog-vllm-deepseek-v3-sparse, blog-nsa]
performance_claims:
  - gpu: B200
    software: "FlashMLA README sparse MLA prefill path with CUDA 12.9; exact library revision not stated"
    dtype: not stated in README
    shape: unspecified peak case for sparse MLA prefill; CUDA 12.9
    workload: "sparse MLA prefill"
    metric: TFLOPS
    value: 1450
    measurement_method: "source-reported upstream README headline; method not stated"
    baseline: "none; absolute throughput headline"
    limitations: "shape, dtype, clocks, and exact library revision are not stated in the headline"
    source_id: blog-flashmla
blackwell_relevance: "FlashMLA supplies SM100 sparse MLA prefill/decode kernels; an external indexer provides token indices, and attention executes only the selected positions."
---

# Sparse MLA

## Interface boundary

DeepSeek V3.2's sparse-attention system has an index-selection stage and a sparse MLA stage. FlashMLA's sparse kernels consume an `indices` tensor; they do not, by themselves, implement the Lightning Indexer/top-k scorer described in serving-system sources.

For decoding, `indices[batch, seq_q, topk]` identifies token positions in the paged cache. Invalid entries are `-1`. For sparse prefill, the documented interface uses `indices[s_q, h_kv, topk]`, requires `h_kv = 1` in the equivalent reference, and accepts `-1` or values `>= s_kv` as invalid. The current README says this prefill interface has no batch dimension.

## Logical reference

```python
def sparse_mla_reference(q, kv, indices, sm_scale):
    selected = gather_valid_tokens(kv, indices)
    logits = matmul(q, transpose_last_two(selected)) * sm_scale
    probabilities = softmax(logits, axis=-1)
    return matmul(probabilities, selected)
```

The production implementation uses tiled online-softmax and layout-specific gathers. It must preserve invalid-index masking and return the documented output/LSE (and, for prefill, max-logit) semantics.

## FP8 decode cache

FlashMLA documents a 656-byte per-token cache format for the FP8 decode path:

- 512 E4M3 values (512 bytes) for the quantized NoPE part;
- four FP32 scales (16 bytes), each covering 128 values;
- 64 BF16 RoPE values (128 bytes).

The kernel dequantizes this cache to BF16 and performs attention computation in BF16. Thus describing its 350-TFLOP/s B200 sparse-decode result as “FP8 compute” is misleading.

## Performance boundary

The upstream FlashMLA README reports peaks of 410 TFLOP/s for H800 sparse decode, up to 350 TFLOP/s for a then-unoptimized B200 sparse decode, 640 TFLOP/s for H800/CUDA 12.8 sparse prefill, and 1,450 TFLOP/s for B200/CUDA 12.9 sparse prefill. The sparse-prefill headline states neither its dtype nor its shape, so this page does not attach BF16, `seqlen=32k`, or `topk=2048` to the 1,450 figure.

Sparse attention reduces arithmetic only relative to the selected set and can worsen gather locality. Benchmark index generation, gather, attention, and end-to-end serving separately.
