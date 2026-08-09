---
id: doc-tfla
title: Tiled Flash Linear Attention (TFLA)
url: https://arxiv.org/abs/2503.14376v3
source_category: paper
architectures:
- sm90
tags:
- linear-attention
- chunk-parallelism
- triton
retrieved_at: 2026-08-08
---

## Verified scope

Tiled Flash Linear Attention adds a second level of sequence parallelization within each chunk. The paper states that this enables arbitrarily large chunks, raises arithmetic intensity, and reduces the need to materialize intermediate recurrent states.

The paper applies TFLA to mLSTM. Its official `NX-AI/mlstm_kernels` repository at commit `5b98ff8e2bec189b3d3c249405bab5149564d6f8` provides PyTorch, JAX, and Triton mLSTM implementations and reports H100 benchmarks.

This source does not establish a Gated DeltaNet implementation, a Blackwell implementation, or inline WGMMA/tcgen05 assembly. Those claims are outside the cited paper and code revision.

## Primary sources

- [Paper revision 3](https://arxiv.org/abs/2503.14376v3)
- [Official code at `5b98ff8`](https://github.com/NX-AI/mlstm_kernels/tree/5b98ff8e2bec189b3d3c249405bab5149564d6f8)
