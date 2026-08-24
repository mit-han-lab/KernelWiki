---
id: blog-colfax-cutlass
title: 'Colfax CUTLASS Tutorial: GEMM Kernels Using Tensor Memory for Blackwell'
author: Colfax Research
url: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
source_category: community-note
architectures: [sm100]
tags: [tcgen05, tmem, cute-dsl, warp-specialization, 2sm-cooperative]
retrieved_at: 2026-08-18
---

# Colfax CUTLASS Blackwell tutorial

This tutorial explains the CUTLASS/CuTe programming model for Blackwell GEMM using UMMA (`tcgen05`) and Tensor Memory.

## Source-backed topics

- the architectural transition from register-resident WGMMA accumulators to TMEM-resident UMMA accumulators;
- the 128-lane by 512-column TMEM organization and column/lane address model;
- descriptor construction and TMEM allocation in CuTe;
- MMA traits/atoms and tiled-copy layouts used to move results from TMEM to registers for the epilogue;
- warp-specialized producer, MMA, and epilogue roles in a complete CUTLASS kernel.

The previous local page contained illustrative inline PTX/C++ snippets labeled as extracted code. Exact searches of the cited tutorial did not find those snippets, so they and the derived artifact bundle were removed. Readers should use the tutorial's own code and the matching CUTLASS version.
