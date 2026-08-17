---
id: blog-blackwell-microbenchmarking
title: Microbenchmarking NVIDIA's Blackwell Architecture
author: Aaron Jarmusch et al.
url: https://arxiv.org/abs/2512.02189
source_category: benchmark-blog
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, fp4, fp8, fp6, gemm, wgmma, cluster, 2sm-cooperative]
retrieved_at: 2026-08-16
---

# Microbenchmarking NVIDIA's Blackwell Architecture

This paper reports measurements on a B200 system, including 148 SMs and unified 192-GB HBM3e. Selected source-reported microbenchmark results include:

- TMEM is organized as 512 columns by 128 lanes of 32-bit cells; the paper quotes a 16-TB/s read-bandwidth specification but measures approximately 8 TB/s for its chained TMEM benchmark and reports its best transfer efficiency around 64×64-element tiles;
- `tcgen05.mma` latency of about 11.0–11.4 cycles for the tested tile forms;
- 7,700.2 TFLOP/s for the tested FP4 tensor-core microbenchmark (96.2% of the paper's theoretical peak); and
- about 4.14 TB/s in its 4–16-GB STREAM-triad runs (51.8% of the cited nominal bandwidth).

These values are experimental results under the paper's clock, instruction stream, working set, and system configuration. They do not establish a universal TMEM tile optimum, cache hit rate, or application bottleneck. The paper's causal decomposition and application comparisons are likewise its analysis, not architecture guarantees.

The earlier local summary converted those microbenchmarks into categorical advice (“all precisions near peak means data movement is the bottleneck”) and conflated the companion SM120 paper with the B200 source. This record retains only clearly scoped results from arXiv:2512.02189.
