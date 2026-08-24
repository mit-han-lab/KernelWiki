---
id: doc-cutlass-blackwell
title: NVIDIA CUTLASS Blackwell support map
url: https://docs.nvidia.com/cutlass/latest/CHANGELOG.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, tma, clc, 2sm-cooperative, nvfp4, fp8, fp4, fp6, block-scale, cute-dsl]
retrieved_at: 2026-08-18
---

# NVIDIA CUTLASS Blackwell support map

CUTLASS documents SM100 support across its changelog and focused guides rather
than through one universal kernel API. Relevant primary pages are:

- [CUTLASS changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html) for
  release-specific feature additions;
- [Blackwell GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
  for supported schedule and datatype combinations;
- [Cluster Launch Control](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)
  for cancel-and-reuse persistent scheduling;
- [CuTe DSL tcgen05](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html)
  for typed MMA and TMEM operations.

The common architecture boundary is that an SM100 MMA is configured through a
typed atom/operator, operand layouts, an instruction descriptor, and—in
block-scaled modes—scale-factor layouts/descriptors. TMEM allocation and
load/store have warp-collective contracts even though the MMA itself is issued
by one elected thread. One-SM versus two-SM, persistent versus nonpersistent,
and NVFP4 versus MX schedule tags are separate choices.

The former local source page included invented schedule names, a universal
16-warp attention decomposition, and pseudo-C++ epilogue visitors. Those were
removed because the changelog did not establish them as general CUTLASS APIs.
