---
id: blog-colfax-cutlass-kernels
title: Colfax CUTLASS Kernels
author: Colfax Research
url: https://github.com/ColfaxResearch/cutlass-kernels
source_category: community-note
architectures:
- sm90
tags:
- cuda-cpp
- gemm
- tma
- wgmma
- persistent-kernel
- tile-scheduling
- pipeline-stages
retrieved_at: '2026-05-20'
description: Source-map entry for Hopper CUTLASS GEMM, FMHA, and scheduling examples.
---

The checked build scripts target `sm_90a` with CUDA 12.2/12.3-era CUTLASS. Use
these kernels as Hopper implementation evidence for GEMM/FMHA and TMA-GMMA
pipelines. Porting an idea to SM100 requires a separate Blackwell source because
WGMMA code is not a `tcgen05` implementation.
