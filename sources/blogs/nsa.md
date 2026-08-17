---
id: blog-nsa
title: Native Sparse Attention
author: DeepSeek AI
url: https://arxiv.org/abs/2502.11089
source_category: benchmark-blog
architectures: [sm80]
tags: [sparse-attention, attention, triton, chunk-parallelism]
retrieved_at: 2026-08-16
---

# Native Sparse Attention

NSA combines compressed coarse-grained tokens, selected fine-grained blocks, and a local sliding window. The selected path is block-structured and shares selections across grouped query heads to align the sparsity pattern with GPU data access.

Section 5 of the paper states that the efficiency experiments use an eight-GPU A100 system. Figure 6 compares the authors' Triton NSA kernel with a Triton FlashAttention-2 baseline. At 64K context it reports up to 9.0× forward and 6.0× backward speedup. Its decoding analysis reports up to 11.6× at 64K context, tied to reduced KV-cache loading.

These are paper-reported component benchmarks, not a universal end-to-end serving gain and not an H100 or SM100 result. The model mechanism includes the compression and selection machinery as well as the sparse attention kernel.
