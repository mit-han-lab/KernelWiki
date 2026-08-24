---
id: blog-amandeep-nvfp4
title: "Twelve Attempts at an FP4 Kernel"
author: Amandeep Singh
url: https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/
source_category: community-note
architectures: [sm100]
tags: [nvfp4, gemv, fp4, block-scale, batched-gemv]
techniques: [vectorized-loads, cache-policy, register-budgeting, per-k-specialization, data-reuse]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx]
retrieved_at: 2026-08-18
---

# Twelve Attempts at an FP4 Kernel

Amandeep Singh's worklog covers NVFP4 GEMV attempts from the GPU Mode
hackathon on B200. Inputs are packed E2M1 with one E4M3 scale for each group of
16 elements, and the output is FP16.

## Reported baseline

The article reports these results for the author's raw CUDA implementation:

| M | K | L | Kernel (microseconds) | Speed of light (microseconds) | Ratio |
|---:|---:|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 26.7 | 8.6 | 3.1x |
| 4096 | 7168 | 8 | 45.1 | 17.3 | 2.6x |
| 7168 | 2048 | 4 | 16.4 | 4.3 | 3.8x |

That kernel assigns one warp to an output row, decodes packed FP4 pairs, uses
paired half arithmetic, accumulates in FP32, and reduces the row with warp
shuffles. These are source-reported implementation and benchmark facts, not
independently reproduced results.

## Reported negative experiments

- A C++ split-K version with atomics added contention and memory traffic.
- Replacing two `uchar4` reads with one `uint2` read was 16--25% slower because
  the attempted unpacking added instructions.
- Four accumulator chains regressed performance by 32--55% in the attempted
  mapping; the author attributes this to register pressure/spilling and poorer
  coalescing.
- Increasing the unroll factor from four to eight regressed the measured cases
  by 5--87%.
- The attempted manual prefetch pipeline increased register pressure without a
  benefit.

These observations apply to the implementations and shapes in the worklog;
they are not universal rules against split-K, vector loads, ILP, or software
pipelining.

## Comparison with leading entries

The author reports that the top solutions used explicit PTX load/decode paths,
different cache policies for streamed A and reused B, wider vector loads with
efficient unpacking, compile-time specialization by K, and tighter register
budgets. That comparison is a post-contest analysis by the blog author. The
local source map does not promote it to an independently verified performance
claim.
