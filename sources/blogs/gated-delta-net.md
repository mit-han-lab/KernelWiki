---
id: blog-gated-delta-net
title: Gated Delta Networks
author: NVlabs
url: https://github.com/NVlabs/GatedDeltaNet
source_category: benchmark-blog
architectures: [sm90, sm100]
tags: [gated-delta-net, linear-attention, attention, triton, chunk-parallelism]
retrieved_at: 2026-08-16
---

# Gated Delta Networks

This repository is the official PyTorch implementation accompanying “Gated Delta Networks: Improving Mamba2 with Delta Rule” (ICLR 2025). Its README points users to Flash Linear Attention for variable-length support and states that the FLA kernels are faster than the repository's reference implementation.

The README records later incorporation into Qwen3-Next, Qwen3.5, and OLMo Hybrid. Those adoption notes do not establish a particular kernel shape, state size, Blackwell lowering, or speedup. The former local record attached a `10×` cross-model serving claim to an H100 kernel shape and included incomplete mathematical/Triton sketches as “key code”; both are removed.

For reproducible implementation details, use the upstream repository/FLA code or the separately pinned SGLang file in `artifacts/kernels/gated-delta-net/full/`.
