---
id: blog-gated-delta-net
title: Gated Delta Networks reference repository
author: NVlabs
url: https://github.com/NVlabs/GatedDeltaNet
source_category: benchmark-blog
architectures: []
tags: [gated-delta-net, linear-attention, attention, triton, chunk-parallelism]
retrieved_at: 2026-08-18
source_commit: b53d6d3a161267432a79c1c04af69fa52bddc921
---

# Gated Delta Networks reference repository

The NVlabs repository is the official PyTorch implementation associated with
the ICLR 2025 Gated Delta Networks paper. Its README points users needing
variable-length training or inference to the Flash Linear Attention (FLA)
implementation and explicitly recommends that path for performance.

The README records integration in Qwen3-Next, Qwen3.5, and OLMo Hybrid. Those
adoption notes do not by themselves establish a layer ratio, expert count,
state shape, GPU target, or throughput figure, so this source map does not infer
those properties.

The former local “reference” recurrence and Triton decode blocks were removed:
they were synthesized, and their updates did not faithfully implement the
paper's gated delta rule. Use the pinned upstream repository, paper, or the
linked FLA implementation for the mathematical and executable definitions.
