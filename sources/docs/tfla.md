---
id: doc-tfla
title: "Tiled Flash Linear Attention (TFLA)"
url: https://arxiv.org/abs/2503.14376
source_category: paper
architectures: [sm90]
tags: [linear-attention, chunk-parallelism, triton]
retrieved_at: 2026-08-18
---

# Tiled Flash Linear Attention

The TFLA paper presents a tiled algorithm for linear recurrent models. It adds
sequence parallelism within a chunk so that chunk size is not restricted by the
intermediate-state materialization used by earlier chunkwise formulations. The
paper evaluates the method for the matrix-LSTM (mLSTM), including training
forward and backward passes.

The authors' companion implementation is
[`NX-AI/mlstm_kernels`](https://github.com/NX-AI/mlstm_kernels/tree/5b98ff8e2bec189b3d3c249405bab5149564d6f8).
Its README identifies `chunkwise--triton_xl_chunk` as the TFLA mLSTM kernel and
reports H100 benchmarks. This source does not establish Gated DeltaNet support,
a Blackwell-specific implementation, or the use of WGMMA or `tcgen05` inline
PTX. Those claims have therefore been removed from the local source map.
