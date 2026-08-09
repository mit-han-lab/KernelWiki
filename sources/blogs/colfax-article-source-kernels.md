---
id: blog-colfax-article-source-kernels
title: Colfax Article Source Kernels
author: Colfax Research
url: https://github.com/ColfaxResearch/cfx-article-src
source_category: community-note
architectures:
- sm90a
tags:
- cuda-cpp
- cute-dsl
- gemm
- tma
- wgmma
- pipeline-stages
- persistent-kernel
- swizzling
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for TMA, pipelined GEMM, Stream-K, and CuTe transpose examples.
---

At commit `fbecfed88de2e4246f104a023188ba722937c5fc`, the relevant Hopper
examples compile for SM90a. Inspect `tma/tma_copy.h`, `pipeline-gemm/`,
`streamk/tile_scheduler.hpp`, and `transpose-cute/` for exact TMA, pipelined
GEMM, persistent/Stream-K scheduler, and transpose implementations. This pinned
tree contains no demonstrated SM100 target.
