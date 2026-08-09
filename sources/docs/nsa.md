---
id: blog-nsa
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
author: DeepSeek AI
url: https://aclanthology.org/2025.acl-long.1126/
source_category: paper
architectures: []
tags: [sparse-attention, attention, triton]
retrieved_at: 2026-08-08
---

## Source identity and scope

This is the ACL 2025 proceedings paper for Native Sparse Attention. Its efficiency experiments use an eight-GPU A100 system. It does not establish SM90/SM100 compatibility or Blackwell performance for the paper's Triton implementation.

## Architecture

NSA combines three attention branches with learned input-dependent sigmoid gates:

1. overlapping KV blocks compressed by learned MLPs with intra-block position encoding;
2. fine-grained blocks selected by aggregating and reusing compression-attention scores; and
3. a direct sliding window over recent tokens.

The experimental settings are compression block length 32/stride 16, selected block length 64 with 16 blocks, and a 512-token window.

## Hardware-aligned selected attention

For selected attention, the paper describes a Triton training/prefill kernel that loads all query heads in a GQA/MQA group together, shares their sparse KV indices, consumes contiguous KV blocks, and maps nearly constant query/output loops to grid parallelism. It does not publish source code or an exact three-dimensional grid declaration.

## Source-reported performance

- Figure 5 reports 9.0x forward and 6.0x backward speedup at 64K against the authors' Triton FlashAttention-2 baseline.
- Table 4 gives 11.6x as an **expected** 64K decoding speedup derived from memory-access volume, not a matched timing measurement.

The record does not provide dtype, software versions, batch details, raw samples, or variance for a fully reproducible benchmark tuple.

## Deployment boundary

DeepSeek-V3.2-Exp later released DeepSeek Sparse Attention (DSA), whose learned indexer selects token positions and whose sparse kernels are provided through FlashMLA. That later DSA mechanism is not the paper's gated three-branch NSA architecture.
