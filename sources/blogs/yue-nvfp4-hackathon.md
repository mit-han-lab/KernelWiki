---
id: blog-yue-nvfp4
title: Blackwell NVFP4 Kernel Hackathon Journey
author: Yue Zhang
url: https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html
source_category: community-note
architectures: [sm100, sm100a]
tags: [nvfp4, gemv, fp4, block-scale, batched-gemv]
techniques: [vectorized-loads, register-reuse, loop-unrolling]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx, cute-dsl]
retrieved_at: 2026-08-16
---

# Blackwell NVFP4 Kernel Hackathon Journey

Yue Zhang reports a final leaderboard submission of 22.392 microseconds for GPU Mode Problem 1. The post describes two paths:

- a CuTe DSL template near 100 microseconds, improved to about 33 microseconds; and
- a raw CUDA path that began near 2,000 microseconds, reached 443 microseconds after coalescing/thread collaboration, 39 microseconds after removing shared-memory overhead plus vectorized loads and hardware conversions, about 27 microseconds with a PTX decode block, about 26 microseconds after parameter tuning, and about 22.9 microseconds with two-tile instruction-level parallelism.

The post includes real source listings for its particular kernel. The former local record replaced them with short synthesized fragments and added cache-hint claims absent from the article. Those fragments are removed; the source measurements are retained as one author's optimization trajectory, not isolated causal benchmarks.

Primary source: [Yue Zhang, “Blackwell NVFP4 Kernel Hackathon Journey”](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html).
