---
id: doc-nsa
title: "Native Sparse Attention (NSA)"
author: Jingyang Yuan et al. (DeepSeek-AI, Peking University, University of Washington)
url: https://arxiv.org/abs/2502.11089
source_category: paper
architectures: [sm80]
tags: [sparse-attention, attention, triton, chunk-parallelism]
retrieved_at: 2026-04-16
---

## Summary

The DeepSeek-led paper presents natively trainable sparse attention with compressed,
selected, and sliding-window branches. Its kernel comparisons use NVIDIA A100
GPUs; this source does not establish an H100 or B200 benchmark.

## Architecture
1. Token compression via learnable MLP (coarse-grained)
2. Token selection using blockwise importance scores (top-n fine-grained blocks)
3. Sliding window (w=512) for local context

## Key Techniques
- Hardware-aligned blockwise memory access
- Group-centric loading: shares sparse KV blocks across GQA group heads
- Triton kernel: grid-based loop scheduling
- Up to 9.0x forward and 6.0x backward speedup at 64K context versus the
  paper's Triton FlashAttention-2 comparison, on A100
- Table 4 derives an expected decoding speedup of up to 11.6x at 64K context
  from per-operation memory-access volume; this is not measured decoding latency
