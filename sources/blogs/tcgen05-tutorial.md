---
id: blog-tcgen05-tutorial
title: tcgen05 for dummies
author: Gau Nernst
url: https://gau-nernst.github.io/tcgen05/
source_category: community-note
architectures: [sm100]
tags: [tcgen05, tmem, swizzling, pipeline-stages, persistent-kernel, warp-specialization, mbarrier, cuda-cpp, ptx]
retrieved_at: 2026-08-16
---

# tcgen05 for dummies

Gau Nernst's tutorial develops one BF16 B200 GEMM for `M=N=K=4096` and compares it with PyTorch 2.9.1/CUDA 13 cuBLAS at 1,506.74 TFLOP/s. Its reported progression is:

| Variant | Reported TFLOP/s |
|---|---:|
| basic `tcgen05`, 2D 16-byte TMA | 254.62 |
| basic `tcgen05`, 3D 16-byte TMA | 252.81 |
| 2D 128-byte-swizzled TMA | 681.20 |
| 3D 128-byte-swizzled TMA | 695.43 |
| pipelining | 939.61 |
| warp specialization | 1,208.83 |
| 2-SM MMA | 1,302.29 |
| persistent kernel with static scheduling | 1,475.93 |

The article is valuable for its descriptor/layout derivation and staged measurement. Its 128-byte swizzle is a choice for this K-major BF16 kernel, not a universal `tcgen05` requirement. The final measured kernel is persistent with static scheduling; Cluster Launch Control is left as an exercise rather than implemented in that result. Likewise, each row changes more than one generated-code detail and should not be used as an architecture-wide speedup guarantee.

The former local source embedded abbreviated pseudo-CUDA with invalid `tcgen05` allocation and synchronization syntax. It has been removed in favor of the actual article and linked source files.

Primary source: [Gau Nernst, “tcgen05 for dummies”](https://gau-nernst.github.io/tcgen05/).
