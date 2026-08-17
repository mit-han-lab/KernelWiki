---
id: blog-simon-nvfp4-gemv
title: NVFP4 GEMV and Improved NVFP4 GEMV
author: Simon Veitner
url: https://veitner.bearblog.dev/nvfp4-gemv/
related_urls:
  - https://veitner.bearblog.dev/nvfp4-gemv-improved/
source_category: community-note
architectures: [sm100]
tags: [nvfp4, gemv, fp4, block-scale, cute-dsl, vectorized-loads, register-reuse, batched-gemv]
retrieved_at: 2026-08-16
---

# NVFP4 GEMV and Improved NVFP4 GEMV

Both configured author pages were fetched successfully on 2026-08-16: the
13-November introduction and the 16-November “improved” follow-up. They are
public explanatory articles and do not state a leaderboard identity or rank.

The first article walks through a CuTe DSL reference kernel configured with an
`(M,N,K) = (128,1,64)` MMA tile, 128 threads per CTA, FP4 E2M1 operands,
FP8 E4M3FN scale factors, an FP16 output, and one scale per 16 values. Those
settings describe the article's reference example; they are not universal GEMV
requirements.

The follow-up parallelizes the K reduction and prints GPU Mode runner means for
the three contest shapes. The article does not label the unit, so this record
preserves the raw values rather than silently converting them:

| Article variant | Shape 1 | Shape 2 | Shape 3 |
|---|---:|---:|---:|
| Original one-thread-per-row reference | 234495.997 | 119713.035 | 38911.998 |
| Extra K-dimension blocks plus FP32 atomic reduction | 36864.001 | 55399.918 | 24576.001 |
| More threads plus atomic reduction | 38911.998 | 67258.720 | 26602.666 |
| More threads without atomics | 38911.998 | 65600.000 | 30719.999 |

The first-shape ratio between the reference and extra-block raw measurements is
about 6.36×; the ratio is unit-independent. This is a comparison between two
article variants on that one shape, not a submitted suite score or an isolated
causal measurement of atomics.

No complete participant submission is archived locally. Earlier local code was
an educational reconstruction and remains excluded from attribution.
