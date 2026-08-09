---
id: blog-gated-delta-net
title: Gated Delta Networks
author: NVlabs
url: https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921
source_category: benchmark-blog
architectures: []
tags:
- gated-delta-net
- linear-attention
- attention
- triton
- chunk-parallelism
retrieved_at: 2026-08-08
---

## Scope

This source capture summarizes the official NVlabs repository at commit `b53d6d3a161267432a79c1c04af69fa52bddc921` and the associated ICLR 2025 paper. It does not preserve a hardware performance record.

## Mechanism

- Gated DeltaNet combines an independent exponential decay gate with the delta rule's targeted memory correction.
- In the repository implementation, the recurrent state is shaped `[batch, heads, head_qk_dim, head_v_dim]`; its dimensions are model parameters rather than a universal `128x128` constant.
- The layer also owns learned Q/K/V, decay, update, convolution, output-gate, normalization, and output-projection parameters.
- The training path uses chunkwise Triton kernels and a WY representation rather than the additive update shown in the former local snippets.

## Implementations and adoption

The authors' FAQ says that FLA kernels are faster, support variable-length training, and are strongly recommended for better performance. The repository's dated updates record integration into Qwen3-Next and Qwen3.5.

No local code block is extracted from this summary. Consult the pinned upstream implementation for executable kernels.
