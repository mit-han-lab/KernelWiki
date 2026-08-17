---
id: doc-tfla
title: Tiled Flash Linear Attention (TFLA)
url: https://arxiv.org/abs/2503.14376
source_category: paper
architectures: [sm90, sm100]
tags: [linear-attention, gated-delta-net, chunk-parallelism]
retrieved_at: 2026-08-16
---

# Tiled Flash Linear Attention

TFLA adds a second level of sequence tiling to chunkwise linear attention. This permits much larger logical chunks than the SRAM-bound FLA tile, raises arithmetic intensity, and reduces how many recurrent states must be materialized in HBM. The paper applies the algorithm to mLSTM variants and explains how the formulation can extend to other linear RNNs.

The chunk size remains a runtime/memory/FLOP tradeoff. The paper reports measured optima around 128–256 for some H100 configurations and emphasizes that higher FLOP/s need not mean lower total runtime because larger chunks also increase work.

The former local record claimed the authors emitted inline WGMMA on Hopper and `tcgen05` on Blackwell. The paper describes Triton kernels and does not contain those instruction claims. It provides a hardware-agnostic algorithmic discussion of newer accelerators; it is not primary evidence for an SM100 instruction lowering.
