---
id: blog-modular-blackwell
title: 'Modular: Matrix Multiplication on Blackwell'
author: Ali Taha, Jiexiang Liu, Hengjie Wang, and Abdul Dakkak
url: https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction
source_category: community-note
architectures: [sm100, sm100a]
tags: [gemm, tcgen05, tmem, tma, 2sm-cooperative, pipeline-stages, tma-multicast]
techniques: [pipeline-stages, tma-multicast, warp-specialization, double-buffering]
hardware_features: [tcgen05, tmem, tma, 2sm-cooperative]
kernel_types: [gemm]
languages: [mojo]
retrieved_at: 2026-08-16
---

# Modular Blackwell matmul series

Modular's series implements Blackwell matrix multiplication in Mojo. Part 3 combines CTA clustering, TMA multicast, two-CTA MMA, a circular shared-memory pipeline, warp specialization, and double-buffered write-out. It reports:

- the multicast/two-SM intermediate at 360.2 TFLOP/s, about 20% of its SOTA target;
- the pipelined/warp-specialized version at 1,429 TFLOP/s, about 81%; and
- double-buffered output at about 85%.

The article says the combined 2-SM and pipelining work is roughly a 5× improvement over the preceding point in its series. Those are cumulative variants, not isolated per-technique speedups.

TMA multicast partitions/copies cluster tiles according to the selected cluster layout; it is not a universally “free N× reduction.” Likewise, the article's pipeline depth and warp allocation are its kernel configuration, not a B200 optimum. The former local summary invented a five-row 20→45→60→70→85% progression and attributed the remaining gap to a specific list; those claims are removed.

The series mentions CLC as possible future work rather than an implemented part
of the reported kernel, so CLC is not classified as evidence for this record.

Primary source: [Part 3, “The Optimizations Behind 85% of SOTA Performance”](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance).
