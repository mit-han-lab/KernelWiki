---
id: blog-tcgen05-tutorial
title: tcgen05 for dummies
author: Thien Tran (gau-nernst)
url: https://gau-nernst.github.io/tcgen05/
source_category: community-note
architectures: [sm100]
tags: [tcgen05, tmem, swizzling, pipeline-stages, persistent-kernel, warp-specialization, mbarrier, cuda-cpp, ptx]
retrieved_at: 2026-04-16
---

# tcgen05 for dummies

This community tutorial develops a B200 BF16 GEMM in CUDA C++ and inline PTX. Its implementation stages are useful case-study evidence, not an architectural requirement or a generic performance prediction.

## Reported benchmark progression

| Tutorial stage | Reported throughput | Fraction of its cuBLAS row |
| --- | ---: | ---: |
| `v1a`, basic kernel | 254.62 TFLOPS | 16.9% |
| `v2b`, changed TMA/layout path | 695.43 TFLOPS | 46.2% |
| `v3`, pipelining | 939.61 TFLOPS | 62.4% |
| `v4`, warp specialization | 1208.83 TFLOPS | 80.2% |
| `v5`, paired-CTA MMA | 1302.29 TFLOPS | 86.4% |
| `v6`, persistent static scheduling | 1475.93 TFLOPS | 98.0% |
| cuBLAS comparison | 1506.74 TFLOPS | 100% |

These rows share the article’s benchmark context. The change from `v1a` to `v2b` is about 2.7×, but it changes the transfer/layout implementation; it does not prove that 128-byte swizzling alone gives that ratio on other kernels.

## Source-bounded lessons

- The tutorial uses TMEM for D and explicitly manages its lifetime.
- MMA issue, TMEM allocation, and TMEM load/store do not have the same issue granularity. Follow the PTX `.sync.aligned` rules rather than guarding allocation with one lane.
- The required MMA operand after the matrix descriptors is an instruction descriptor; it is not an unused scale descriptor.
- Its chosen shared-memory layout and descriptor agree. Other supported swizzle modes remain valid when their own constraints are met.
- The performance percentages are comparisons to the tutorial’s own cuBLAS row, not percentages of a universal hardware peak.

The former extracted-code bundle was removed because it preserved obsolete local reconstructions rather than a commit-pinned upstream tree. For normative instruction syntax, use the [PTX ISA source map](../docs/nvidia-ptx-isa-sm100.md).
