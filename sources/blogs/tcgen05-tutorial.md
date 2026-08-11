---
id: blog-tcgen05-tutorial
title: tcgen05 for dummies
author: Gau Nernst
url: https://gau-nernst.github.io/tcgen05/
source_category: community-note
architectures:
- sm100
tags:
- tcgen05
- tmem
- swizzling
- pipeline-stages
- persistent-kernel
- warp-specialization
- mbarrier
- cuda-cpp
- ptx
retrieved_at: 2026-04-16
artifact_dir: artifacts/blogs/tcgen05-tutorial/code
---

# tcgen05 for dummies

## Scope and provenance

Gau Nernst's tutorial develops a plain CUDA C++/PTX GEMM on a Modal B200. The article is dated 2025-12-21, and the relevant code is in `02e_matmul_sm100` at repository commit `3b90ac9b3f624bdf1f6f78d02dcd533675d36573`. The disclosed benchmark uses M=N=K=4096 and compares against PyTorch 2.9.1 with CUDA 13 cuBLAS.

Use the linked pinned source for complete assembly operands and synchronization. Earlier local excerpts omitted required tcgen05 operands and were not valid standalone code.

## Source-reported progression

| Version | Author's description | TFLOP/s |
|---|---|---:|
| cuBLAS | PyTorch 2.9.1 + CUDA 13 | 1506.74 |
| v1a | basic tcgen05 + 2D 16B TMA | 254.62 |
| v1b | 3D 16B TMA | 252.81 |
| v2a | 2D 128B TMA | 681.20 |
| v2b | 3D 128B TMA | 695.43 |
| v3 | pipelining | 939.61 |
| v4 | warp specialization | 1208.83 |
| v5 | 2-SM MMA | 1302.29 |
| v6 | persistent with static scheduling | 1475.93 |

These values are reports for the author's exact environment; they are not portable performance guarantees. The final v6 kernel uses static scheduling. The article explicitly says threadblock swizzling and Cluster Launch Control were not added.

## Supported takeaways

- The tutorial demonstrates TMEM allocation/lifecycle, tcgen05 MMA, TMA transfers, mbarrier-driven pipelines, 128-byte swizzling as a performance optimization, warp specialization, 2-SM MMA, and persistence.
- Its 128-byte swizzle stage substantially improves this benchmark, but PTX also defines other valid shared-memory descriptor modes.
- The article's completion and pipeline code should be read in full; short fragments can omit the mbarrier, fence, descriptor, and operand-lifetime context required for correctness.

## References

- [Article](https://gau-nernst.github.io/tcgen05/)
- [Pinned source tree](https://github.com/gau-nernst/learn-cuda/tree/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100)
- [PTX ISA 9.0 tcgen05 reference](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma)
