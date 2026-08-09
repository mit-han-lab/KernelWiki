---
id: blog-simveit-effective-transpose
title: simveit effective_transpose
author: Simon Veitner
url: https://github.com/simveit/effective_transpose
source_category: community-note
architectures:
- sm90a
tags:
- cuda-cpp
- tma
- swizzling
- shared-memory-optimization
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for CuTe transpose, swizzle, and memory-layout examples.
---

At commit `994b2b5acaa67f80e411df3e8274b6ae13fd1949`, this Hopper repository
compiles for SM90a and contains raw CUDA C++ TMA transpose variants.
`transpose_swizzle_batched.cu` encodes 128-byte tensor-map swizzling. It is not
a CuTe DSL, GEMM, SM100, or explicit vector-load-width source.
