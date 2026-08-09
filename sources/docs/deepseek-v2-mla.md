---
id: doc-deepseek-v2-mla
title: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model'
author: DeepSeek-AI
url: https://arxiv.org/abs/2405.04434v5
source_category: paper
architectures: []
tags:
- mla
- attention
retrieved_at: 2026-08-08
---

## MLA Cache Formula

Section 2.1.1 defines Multi-head Latent Attention and Table 1 compares per-layer, per-token KV-cache elements. Standard MHA caches `2 * n_h * d_h` elements. MLA caches `d_c + d_h^R` elements; for the paper's `d_c=4*d_h` and decoupled-RoPE dimension `d_h^R=d_h/2`, this is approximately `4.5*d_h` elements.

The comparison is stated in elements, so byte totals additionally depend on storage dtype, layer count, and any quantization metadata. This paper record supports the model-level mechanism; FlashMLA's implementation-specific 656-byte FP8 sparse-decode layout is documented separately by its repository.
