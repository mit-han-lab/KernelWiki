---
id: kernel-flashmla
title: FlashMLA — Multi-head Latent Attention
type: kernel
architectures: [sm100, sm90]
tags: [mla, attention, decode, prefill, fp8, sparse-attention]
confidence: source-reported
reproducibility: snippet
kernel_types: [mla, attention, decode, prefill, sparse-attention]
languages: [cuda-cpp]
related: [hw-tcgen05-mma, hw-tmem, kernel-sparse-mla, kernel-nsa]
sources: [blog-flashmla, pr-flashinfer-1117, pr-vllm-39752]
performance_claims:
  - gpu: B200
    software: "FlashMLA README as retrieved 2026-08-16; exact library/CUDA version not stated for this dense headline"
    dtype: not stated in README
    shape: unspecified peak case for dense MHA prefill forward; NVIDIA-reported
    workload: "dense MHA prefill forward"
    metric: TFLOPS
    value: 1460
    measurement_method: "NVIDIA-reported upstream README headline; method not stated"
    baseline: "none; absolute throughput headline"
    limitations: "shape, dtype, clocks, and software environment are not stated in the headline"
    source_id: blog-flashmla
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
blackwell_relevance: "FlashMLA supplies SM100 dense- and sparse-prefill kernels; its published README reports peak B200 results without enough shape detail to infer utilization."
artifact_dir: artifacts/kernels/flashmla
---

# FlashMLA

FlashMLA is DeepSeek's CUDA kernel library for Multi-head Latent Attention. The upstream project exposes separate dense-decode, sparse-decode, dense-prefill, and sparse-prefill paths; architecture, datatype, and cache-format support differ between those paths.

## Interface boundaries

- Dense decode uses a paged KV cache and is documented for Hopper.
- Sparse decode and sparse prefill consume externally produced token indices. FlashMLA does not make the indexer part of the attention kernel.
- Dense MHA prefill is the SM100-specific dense-prefill path in the current project description; it is not labeled dense MLA.
- The 656-byte token layout applies to the FP8 sparse-decode cache: 512 E4M3 values, four FP32 scales, and 64 BF16 RoPE values. It is not a universal MLA cache-size claim.

The sparse interfaces and invalid-index semantics are summarized in [Sparse MLA](sparse-mla.md).

```python
def sparse_attention_contract(query, paged_cache, indices):
    selected = gather_valid_entries(paged_cache, indices)
    probabilities = softmax(query @ selected.keys.T, axis=-1)
    return probabilities @ selected.values
```

This is a logical contract, not copied production code; the pinned artifact supplies the implementation evidence.

## Published performance boundary

The upstream README reports these peak headlines:

| Path | GPU | Reported peak |
|---|---|---:|
| Dense decode | H800 | 3,000 GB/s and 660 TFLOP/s |
| Sparse decode | H800 / B200 | 410 TFLOP/s / up to 350 TFLOP/s on the then-unoptimized B200 path |
| Dense MHA prefill forward / backward | B200 | 1,460 / 1,000 TFLOP/s, as reported by NVIDIA |
| Sparse MLA prefill | H800 CUDA 12.8 / B200 CUDA 12.9 | 640 / 1,450 TFLOP/s |

The headline paragraph does not give the shapes needed to derive utilization and does not state a prefill compute dtype, so this page does not attach percentages, sequence lengths, or a dtype to those figures. The sparse decode cache is stored in FP8 but is dequantized for BF16 attention computation; that decode contract must not be projected onto the separately reported prefill headlines.

## Reproduction

Pinned upstream files and patches are in [`artifacts/kernels/flashmla/full/`](../../artifacts/kernels/flashmla/full/). The former pseudo-kernel was removed rather than retained as implementation evidence.
