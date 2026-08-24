---
id: doc-cutlass-changelog-sm100
title: "CUTLASS Changelog: SM100/Blackwell Entries"
url: https://docs.nvidia.com/cutlass/latest/CHANGELOG.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, nvfp4, fp4, fp6, fp8, block-scale, warp-specialization, persistent-kernel, gemm, grouped-gemm, attention, moe, mla, 2sm-cooperative, tile-scheduling, cute-dsl, epilogue-fusion, sparse-attention]
retrieved_at: 2026-08-18
---

# CUTLASS Changelog: SM100/Blackwell Entries

This page records only release facts needed elsewhere in the wiki. It was
rechecked against NVIDIA's rendered and checked-in changelogs plus GitHub
release records at the audit cutoff.

## Date convention

`data/tool-versions.yaml` uses GitHub's `published_at` calendar date. The
checked-in changelog often carries an earlier project date, so the two must
not be silently interchanged:

| Release | GitHub publication | Changelog label |
|---|---:|---:|
| 4.7.0 | 2026-08-13 | 2026-08-04 |
| 4.6.2 | 2026-08-08 | 2026-08-03 (rendered changelog) |
| 4.6.1 | 2026-07-15 | 2026-07-13 |
| 4.6.0 | 2026-07-13 | 2026-07-01 |
| 4.5.0 | 2026-05-13 | 2026-05-01 |
| 4.4.2 | 2026-03-17 | 2026-03-13 |
| 4.4.0 | 2026-02-26 | 2026-02-14 |
| 4.3.0 | 2025-11-24 | 2025-11-21 |
| 3.8.0 | 2025-02-21 | 2025-01-25 |

The GitHub publication values were read from the official release API; the
labels in the final column are from NVIDIA's official rendered changelog.
The checked-in changelog at tag 4.7.0 includes the dated 4.6.1 section but no
4.6.2 section, so the 4.6.2 cell explicitly identifies its rendered source.
This distinction corrects earlier local pages that presented changelog labels
or unrelated dates as release publication dates.

## Blackwell milestones relevant to this wiki

- 3.8.0 introduced the changelog's initial SM100 support: tcgen05 CuTe MMA
  atoms, Blackwell TMA extensions, TMEM as a CuTe data locale, TMEM movement
  atoms, Blackwell pipelines, and CLC-based scheduling support.
- 4.0.0 introduced the CuTe DSL Python layer and Blackwell SM100 GEMM and
  attention examples.
- 4.2.0 added changelog entries for SM103, SM121, SM100 FP4 GEMV, and further
  block-scaled and attention work.
- 4.5.0 added SM100/SM120 fixes and examples including green-context SM
  partitioning; it is historical, not the current stable release.
- 4.6.0 added the experimental in-kernel event-tracing profiler and the
  `cute.compile_to` compilation API.
- 4.7.0, current at the cutoff, added experimental Primitives and Task
  Scheduling APIs; the latter performs static schedule analysis for
  warp-specialized kernels.

These are release-level statements. They do not imply that every operation,
shape, or CUDA-toolkit combination is supported; use the matching release
notes and code for that narrower question.

## Official records

- [CUTLASS changelog](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- [CUTLASS releases](https://github.com/NVIDIA/cutlass/releases)
