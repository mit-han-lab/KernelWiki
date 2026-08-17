---
id: blog-colfax-article-source-kernels
title: Colfax Article Source Kernels
author: Colfax Research
url: https://github.com/ColfaxResearch/cfx-article-src
source_category: community-note
architectures:
- sm90
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
description: Source-map entry for Hopper TMA, pipelined GEMM, Stream-K, and CuTe transpose examples.
---

This repository collects source files for Colfax Research articles. Its checked
TMA, pipeline-GEMM, Stream-K, transpose-CuTe, and CUTLASS-GEMM build files target
`sm_90a`. The algorithms may inform later ports, but the repository is not direct
SM100 implementation evidence.
