---
id: blog-modular-blackwell
title: "Modular: Matrix Multiplication on Blackwell"
author: Modular
url: https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance
source_category: community-note
architectures: [sm100, sm100a]
tags: [gemm, tcgen05, tmem, tma, 2sm-cooperative, pipeline-stages, tma-multicast, warp-specialization, double-buffering]
techniques: [pipeline-stages, tma-multicast, warp-specialization, double-buffering]
hardware_features: [tcgen05, tmem, tma, 2sm-cooperative]
kernel_types: [gemm]
languages: [mojo]
retrieved_at: 2026-08-09
---

# Modular Blackwell matmul series

## Evidence scope

Part 3 describes a sequence of Blackwell matmul changes rather than isolated universal prescriptions. The implementation uses 2-SM cooperative MMA and TMA multicast, then adds a five-stage circular buffer for A/B shared-memory tiles. Separate warp roles issue TMA and MMA so different stages can progress concurrently.

The article later double-buffers output writeback in shared memory: TMEM partitions are loaded/converted/stored to alternating output buffers while TMA stores progress. It says the reduced output-buffer footprint frees shared memory for deeper input pipelining, reports an additional 64 TFLOP/s for that final change, and labels the resulting endpoint 85% of the article's SOTA reference.

Those are author-reported results for the series' code and benchmark context. The article does not establish that five stages are universally optimal, a fixed B200 HBM latency, a portable percentage for each earlier step, or an architecture-wide optimization order.

## References

- [Part 1: introduction](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
- [Part 3: optimizations behind the 85% endpoint](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)
- [Part 3 linked kernel 6](https://github.com/modular/modular/blob/main/max/kernels/test/gpu/linalg/matmul_blackwell_iterative/6_2sm_pipelined.mojo) — rolling branch, not an immutable revision
