---
id: kernel-flashmla
title: FlashMLA attention kernels
type: kernel
architectures: [sm100, sm90]
tags: [mla, attention, decode, prefill, fp8, sparse-attention]
confidence: source-reported
reproducibility: snippet
kernel_types: [mla, attention, decode, prefill, sparse-attention]
languages: [cuda-cpp, python]
related: [kernel-sparse-mla, kernel-nsa]
sources: [blog-flashmla]
performance_claims:
  - gpu: B200
    dtype: not stated in upstream README
    shape: dense MHA prefill; upstream benchmark suite
    metric: TFLOPS
    value: 1460
    source_id: blog-flashmla
    source_locator: https://github.com/deepseek-ai/FlashMLA#test--benchmark-mha-prefill-dense
  - gpu: B200
    dtype: bf16
    shape: sparse MLA prefill; upstream benchmark suite
    metric: TFLOPS
    value: 1450
    source_id: blog-flashmla
    source_locator: "FlashMLA README, Test & benchmark MLA prefill (Sparse), plus flash_mla/flash_mla_interface.py::flash_mla_sparse_fwd dtype contract"
blackwell_relevance: FlashMLA's support matrix includes SM100 sparse decode, dense MHA prefill, and sparse MLA prefill with CUDA 12.9 or newer for SM100.
---

# FlashMLA attention kernels

FlashMLA is DeepSeek's upstream library for dense and token-sparse attention
kernels used by DeepSeek-V3 and V3.2. “MLA” in the repository's support matrix
has different modes: decode uses MQA-shaped latent attention, while its dense
SM100 prefill entry is MHA. Preserve that distinction when comparing kernels.

## Reproducible API boundary

The pinned upstream README gives this decode flow:

```python
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk
)
o_i, lse_i = flash_mla_with_kvcache(
    q_i, kvcache_i, block_table, cache_seqlens, dv,
    tile_scheduler_metadata, num_splits,
    is_causal, is_fp8_kvcache, indices,
)
```

This is an upstream usage excerpt, not a complete benchmark. Use the repository
tests named by the README and the matching CUDA/toolkit requirements.

## Layout and benchmark boundaries

The documented FP8 decode cache is 656 bytes per token: 512 E4M3 values, four
FP32 scales, and 64 BF16 RoPE values. Sparse decode's indices encode the page
block and within-page offset; invalid values are `-1`.

The retained B200 prefill maxima are 1460 TFLOPS for dense MHA forward and 1450
TFLOPS for sparse MLA forward. They come from different benchmark suites and
must not be interpreted as a controlled dense-versus-sparse comparison.

The former `artifacts/kernels/flashmla` bundle was removed because its files
were sourced from CUTLASS and FlashInfer PRs rather than the FlashMLA repository.
