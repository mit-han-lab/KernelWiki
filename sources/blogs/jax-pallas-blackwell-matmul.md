---
id: blog-jax-pallas-blackwell-matmul
title: "Writing High-Performance Matrix Multiplication Kernels for Blackwell with JAX Pallas"
author: The JAX authors
url: https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html
source_category: community-note
architectures: [sm100]
tags: [gemm, warp-specialization, 2sm-cooperative, persistent-kernel, tma, tmem, pipeline-stages, swizzling, jax-pallas, tcgen05, double-buffering, epilogue-fusion]
retrieved_at: 2026-08-18
---

# JAX Pallas Blackwell matmul tutorial

The tutorial, credited by the page to The JAX authors, develops an FP16
Blackwell matmul for
`m=4096`, `k=4096`, and `n=8192`. Its reported measurements use iid normal
inputs; the page warns that data distribution can change matmul timings.

| Implementation | Tensor-core utilization | Percent of measured cuBLAS utilization |
|---|---:|---:|
| Basic kernel | 37.62% | 59.4% |
| Warp specialization | 45.47% | 71.7% |
| Tiled epilogue | 55.82% | 88.1% |
| Collective two-CTA MMA | 59.41% | 93.7% |
| Persistent kernel | 61.46% | 97.0% |
| Dedicated epilogue warpgroup | 63.38% | 100.0% |
| Grid tiling | 69.44% | 109.6% |

These percentages are the tutorial's results for that setup, not general
performance guarantees.

The Blackwell warp-specialized step divides one Pallas warpgroup into CUDA
warps: one issues asynchronous copies and another issues `tcgen05` MMA. It does
not use two Pallas warpgroups for that step. A later persistent version launches
one two-CTA cluster per pair of SMs and loops over output tiles. Only the later
dedicated-epilogue step uses two Pallas threads/warpgroups and double-buffers the
TMEM accumulator.

The tutorial also demonstrates tiled TMEM-to-SMEM-to-GMEM epilogues, collective
two-CTA MMA, and grid reordering for L2 locality. Each optimization is measured
as part of the stated progression; the table does not isolate a universal
benefit for arbitrary shapes.
