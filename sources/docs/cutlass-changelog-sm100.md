---
id: doc-cutlass-changelog-sm100
title: CUTLASS changelog — SM100 entries through 4.5.0
url: https://github.com/NVIDIA/cutlass/blob/v4.5.0/CHANGELOG.md
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, nvfp4, fp4, fp6, fp8, block-scale, warp-specialization, persistent-kernel, gemm, grouped-gemm, attention, moe, mla, 2sm-cooperative, tile-scheduling, cute-dsl, epilogue-fusion, sparse-attention]
retrieved_at: 2026-08-16
---

# CUTLASS changelog through 4.5.0

This is a curated index of Blackwell-related release milestones, pinned to the 4.5.0 changelog because 4.5.0 is the last stable release before the repository's 2026-05-20 refresh cutoff.

| Release | Changelog date | Selected SM100-related entries |
|---|---|---|
| 3.8.0 | 2025-01-25 | Initial SM100 CuTe/CUTLASS building blocks, TMEM, `tcgen05`, TMA, CLC schedulers, and narrow/block-scaled types |
| 3.9.0 | 2025-04-24 | SM100 sparse GEMM, MLA/FMHA work, and distributed GEMM example |
| 4.0.0 | 2025-06-03 | CuTe DSL examples for SM100 persistent GEMM, grouped GEMM, and FMHA |
| 4.1.0 | 2025-07-16 | SM100 block-scaled persistent GEMM and attention updates |
| 4.2.0 | 2025-09-15 | SM100 low-latency MoE example, FP4 GEMV, and additional attention/GEMM support |
| 4.3.0 | 2025-11-21 | SM100 CuTe DSL tutorial and MLA/backward/grouped/blockwise examples; the 8K tutorial baseline reports 84% SOL |
| 4.4.0 | 2026-02-14 | CuTe DSL CUDA 13.1 support, experimental higher-level APIs, AOT, custom epilogue configuration, GQA/SSD work, and fixes |
| 4.4.2 | 2026-03-13 | Python 3.14 support and Blackwell profiler exposure/fixes |
| 4.5.0 | 2026-05-01 | CuTe DSL block-copy/MXF improvements, SM100 mixed TMA+cp.async 2SM support, static TMEM loads, green-context example 95, and fixes |

The table deliberately reports dates printed inside the changelog. The central
`data/tool-versions.yaml` registry instead uses GitHub release-object publication
dates consistently: 3.8.0 on 2025-02-21, 4.3.0 on 2025-11-24, 4.4.0 on
2026-02-26, 4.4.2 on 2026-03-17, and 4.5.0 on 2026-05-13. The prior local
summary incorrectly dated 4.5.0 March 27 and described entries not present under
that release.

Later CUTLASS releases exist as of this audit but fall after the repository cutoff and are not silently folded into this record.
