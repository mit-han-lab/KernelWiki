---
id: blog-simon-nvfp4-gemv
title: NVFP4 GEMV and Improved NVFP4 GEMV
author: Simon Veitner
url: https://veitner.bearblog.dev/nvfp4-gemv/
source_category: community-note
architectures:
- sm100
tags:
- nvfp4
- gemv
- fp4
- block-scale
- cute-dsl
- batched-gemv
techniques:
- k-dimension-parallelism
- reduction
hardware_features:
- nvfp4
- fp4
- block-scale
kernel_types:
- batched-gemv
- gemv
languages:
- cute-dsl
- python
retrieved_at: 2026-08-08
---

# NVFP4 GEMV and Improved NVFP4 GEMV

## Evidence Scope

Simon Veitner's two posts present CuTe DSL Python kernels. They do not publish the CUDA C++ listings formerly attributed to them by this card, and the benchmark outputs are times in nanoseconds rather than operation counts.

## Reference Post

The first post develops a CuTe DSL reference for the official NVFP4 batched GEMV task. It describes a `(128, 1, 64)` M-N-K tile, 128 threads per block, packed E2M1 operands, E4M3 block scales, and FP32 register accumulation before FP16 output.

The post's benchmark output includes:

| Row | Mean time (ns) |
|---:|---:|
| 0 | 234495.997 |
| 1 | 119713.035 |
| 2 | 38911.998 |

These are author-reported benchmark results from the displayed environment and are not independent reproduction.

## Improved Post

The second post parallelizes the K reduction in three ways:

1. Extra K-grid blocks compute partial sums and use an FP32 global atomic before conversion to FP16.
2. Additional threads collaborate on each row and use an atomic reduction path.
3. Additional threads collaborate through shared memory and an atomic-free reduction.

For the extra-block strategy, the displayed means are 36864.001, 55399.918, and 24576.001 ns for rows 0–2. The post reports additional timing blocks for the two thread-collaboration variants and discusses their shape-dependent tradeoffs. The approximately 6.4× statement compares the first reference row, 234495.997 ns, with 36864.001 ns; it is not a leaderboard score or operation-rate unit.

## Provenance Limits

The posts provide CuTe DSL snippets and benchmark text. Locally reconstructed `.cpp` files extracted from an earlier summary are not verbatim author source and must not be labeled contestant submissions. The public leaderboard snapshot fetched on 2026-08-08 places the `Simon` row at current rank 25 with a 25.112153955 microsecond aggregate score; it does not expose that submission's code.

## Primary Sources

- [NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv/)
- [Improved NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv-improved/)
