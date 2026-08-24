---
id: blog-vllm-deepseek-v3-sparse
title: "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action"
author: vLLM Team
url: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
source_category: community-note
architectures: [sm90, sm100]
tags: [sparse-attention, mla, attention, decode, prefill, fp8, quantization, kernel-fusion]
retrieved_at: 2026-08-18
---

# DeepSeek-V3.2-Exp sparse attention in vLLM

This release post describes vLLM's initial support for DeepSeek Sparse
Attention. A lightning indexer computes relevance logits and selects up to 2048
context positions for each query; FlashMLA performs the following sparse
attention operation. Prefill and decode require different batching treatment.

The post documents a separate indexer-key cache and an FP8 MLA cache occupying
656 bytes per token: 512 bytes of E4M3 NoPE values, 16 bytes of FP32 scales, and
128 bytes of unquantized BF16 RoPE values. The indexer cache uses a per-block
layout, and this release supports a block size of 64; the article also says the
FlashMLA path is tailored to that size.

For the initial release, the authors give launch configurations of 16 H100s,
eight H200s, or eight B200s and a tensor-parallel launch command. They explicitly
say verification against the official accuracy results was still in progress
and that expert parallelism had a bug being fixed. Those are historical release
conditions, not current minimum hardware requirements.

The article names DeepGEMM indexer kernels, a TileLang top-k reference,
quantization fused into page-table writes, and out-of-the-box B200/GB200
support. It reports no latency, throughput, or API-cost result.
