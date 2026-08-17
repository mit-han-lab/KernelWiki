---
id: blog-tilus-nvidia
title: Tilus — A Tile-Level GPU Kernel Programming Language
author: NVIDIA
url: https://github.com/NVIDIA/tilus
source_category: community-note
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, fp6, fp8, gemm, swizzling, pipeline-stages, ptx]
retrieved_at: 2026-08-16
---

# Tilus

Tilus is NVIDIA's research DSL with thread-block-level tensor operations, explicit shared-memory and register tensors, automatic tuning/caching, and low-precision types with bit widths from 1 through 8. The project says v0.2.0 added Hopper and Blackwell support and links a step-by-step B200 matmul tutorial; v0.1.0 targeted Ampere.

The repository describes automatic layout inference as inherited in part from Hexcute and Hidet as its lower-level target/runtime. The former local summary additionally claimed first-class TMEM storage, cluster synchronization, native FP4/FP6 instruction emission, and specific TMA behavior without anchoring those statements to a versioned API. Those claims are omitted here; consult the installed Tilus version and its tutorials for actual lowering support.

Primary sources: [NVIDIA/Tilus](https://github.com/NVIDIA/tilus) and the [Tilus paper](https://arxiv.org/abs/2504.12984).
