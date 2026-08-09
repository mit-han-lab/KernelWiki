---
id: blog-colfax-cutlass-kernels
title: Colfax CUTLASS Kernels
author: Colfax Research
url: https://github.com/ColfaxResearch/cutlass-kernels
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
- warp-specialization
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for CUTLASS GEMM kernel examples and scheduling patterns.
---

At commit `84f0802e2b4a1bf068ac70359f20ffdb368c8f6a`, this repository contains
Hopper SM90a CUTLASS/CuTe GEMM and FMHA examples. The FMHA READMEs and compile
scripts pin CUDA 12.2/12.3, CUTLASS 3.3/3.4, and SM90a; `src/fmha-pipeline/`
implements TMA pipelining with optional warp specialization. The pinned tree is
not evidence for SM100, a persistent scheduler, or a tile scheduler.
