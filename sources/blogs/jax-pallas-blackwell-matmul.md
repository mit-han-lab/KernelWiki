---
id: blog-jax-pallas-blackwell-matmul
title: Writing high-performance matmul kernels for Blackwell with JAX Pallas
author: JAX Team
url: https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html
source_category: community-note
architectures: [sm100]
tags: [gemm, warp-specialization, 2sm-cooperative, persistent-kernel, tma, tmem, pipeline-stages, swizzling, jax-pallas, tcgen05, double-buffering, epilogue-fusion]
retrieved_at: 2026-08-16
---

# JAX Pallas Blackwell matmul tutorial

For iid-normal FP16 inputs with `M=4096`, `K=4096`, and `N=8192`, the tutorial reports:

| Stage | Tensor-core utilization | Relative to its cuBLAS run |
|---|---:|---:|
| basic | 37.62% | 59.4% |
| warp specialization | 45.47% | 71.7% |
| tiled epilogue | 55.82% | 88.1% |
| collective 2-CTA MMA | 59.41% | 93.7% |
| persistent | 61.46% | 97.0% |
| dedicated epilogue warpgroup | 63.38% | 100.0% |
| grid tiling | 69.44% | 109.6% |

The article explicitly warns that input distribution affects comparison. “109.6%” means faster than that particular cuBLAS measurement, not more than hardware peak.

The implementation uses Mosaic GPU/Pallas APIs such as `plgpu`, not CUDA Tile's `ct.load`/`ct.store`. Collective copies partition work across the cluster, and only the leader issues the collective MMA; the two CTAs do not each “load half of every operand” as a universal rule. The former local source mixed these programming models and added an unsupported “about 5×” stage attribution, so those claims are removed.
