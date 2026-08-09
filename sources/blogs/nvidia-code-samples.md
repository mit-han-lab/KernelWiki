---
id: blog-nvidia-code-samples
title: NVIDIA Developer Code Samples
author: NVIDIA Developer Blog
url: https://github.com/NVIDIA-developer-blog/code-samples
source_category: community-note
architectures: []
tags:
- cuda-cpp
- gemm
- shared-memory-optimization
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for CUDA sample kernels and memory-system examples.
---

At commit `3350d216083a902ccbf5b31665e3b82096a75b55`, this cross-generation
repository contains small CUDA examples including
`series/cuda-cpp/coalescing-global/coalescing.cu`,
`series/cuda-cpp/shared-memory/shared-memory.cu`,
`series/cuda-cpp/transpose/transpose.cu`, and
`posts/tensor-cores/simpleTensorCoreGEMM.cu`. The source card makes no SM90 or
SM100 compatibility claim; inspect the selected example's build requirements
before reuse.
