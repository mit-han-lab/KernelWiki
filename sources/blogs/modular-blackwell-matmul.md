---
id: blog-modular-blackwell
title: "Modular: Matrix Multiplication on Blackwell, Part 3"
author: Ali Taha, Jiexiang Liu, Hengjie Wang, and Abdul Dakkak (Modular)
url: https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance
source_category: community-note
architectures: [sm100]
tags: [gemm, tcgen05, tmem, tma, 2sm-cooperative, pipeline-stages, tma-multicast]
techniques: [pipeline-stages, tma-multicast, warp-specialization, double-buffering]
hardware_features: [tcgen05, tmem, tma, 2sm-cooperative]
kernel_types: [gemm]
languages: [cuda-cpp]
retrieved_at: 2026-08-18
---

# Modular Blackwell matmul, part 3

This article in Modular's Mojo matmul series combines CTA-cluster TMA
multicast, two-SM MMA, a staged TMA/MMA pipeline, warp specialization, and a
double-buffered write-out.

For the article's kernel, two CTAs partition the MMA M dimension and the leader
issues the collective operation. TMA multicast reduces redundant tile loads
within the cluster. The initial multicast/two-SM version is reported at
360.2 TFLOP/s, about 20% of the authors' SOTA reference.

The next version allocates five shared-memory stages for A and B and assigns
different warps to issue TMA and MMA while barriers protect buffer reuse. Five
is the depth chosen for this implementation after the article's shared-memory
calculation; it is not a general Blackwell default.

The final step in this article double-buffers smaller output slices so that TMA
stores can overlap later TMEM-load and `stmatrix` work. The authors report an
additional 64 TFLOP/s and 85% of their SOTA reference. They identify persistent
scheduling with cluster launch control as future work in the next article, so
this page does not attribute CLC or persistence to the measured Part 3 kernel.
