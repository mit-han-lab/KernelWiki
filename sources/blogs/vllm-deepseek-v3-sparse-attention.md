---
id: blog-vllm-deepseek-v3-sparse
title: 'DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action'
author: vLLM Team
url: https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html
source_category: community-note
architectures: [sm100, sm100a]
tags: [sparse-attention, mla, attention, flash-attention, decode, prefill, fp8, quantization, kernel-fusion]
retrieved_at: 2026-08-16
---

# DeepSeek-V3.2-Exp in vLLM

The vLLM launch post describes a two-stage serving path: DeepGEMM computes weighted MQA relevance logits for a Lightning Indexer, row-wise top-k produces up to 2,048 indices, and FlashMLA attends to those selected tokens. Prefill and decode need different batching/cache handling.

The post documents the 656-byte FP8 MLA cache layout: 512 E4M3 values, four FP32 scales, and 64 BF16 RoPE values. It says the indexer cache uses a different per-block layout and that the initial release supports block size 64 in part because FlashMLA is tailored to it.

At publication, the usage section listed 16×H100, 8×H200, or 8×B200 and used tensor parallelism because expert parallelism had a bug being fixed. These are release-time deployment notes, not permanent minimum hardware requirements. The post also says accuracy verification was still in progress, had matched expected GSM8K/GPQA-Diamond on a previous weight version, and had removed Hadamard transforms after observing no accuracy effect.

The former local page called configurations recommendations, claimed 50% cost reduction, and inferred an indexer benefit from B200 bandwidth. Those statements are not supported by this post and are removed.
