---
id: doc-ga102-whitepaper
title: "NVIDIA Ampere GA102 GPU Architecture Whitepaper (v2.1)"
url: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
source_category: official-doc
architectures: [sm86]
tags: [mma-sync, cuda-cpp, fp8]
retrieved_at: 2026-08-03
---

# NVIDIA Ampere GA102 GPU Architecture Whitepaper (v2.1)

## Overview

The GA102 whitepaper is the official spec source for consumer/workstation Ampere (sm_86): GeForce RTX 3090/3090 Ti/3080, RTX A6000, A40. The single most important fact for kernel work it documents: **on GA10x, FP16 tensor core math with FP32 accumulation runs at HALF the rate of FP16 accumulation** — a product segmentation choice that GA100 (A100) does not have. Every GEMM/attention kernel that accumulates in FP32 (i.e., all numerically-safe LLM kernels) pays this on RTX 3090.

## RTX 3090 key specs (from the whitepaper tables)

| Spec | RTX 3090 |
|---|---|
| SMs | 82 |
| CUDA cores | 10496 (128 FP32/SM) |
| Tensor cores | 328 (3rd gen, 4/SM) |
| Boost clock | 1.70 GHz |
| Memory | 24 GB GDDR6X, 384-bit, 936 GB/s |
| L2 cache | 6 MB |
| RT cores | 82 (irrelevant for compute) |

## Tensor core peak rates, RTX 3090 (dense / sparse)

| Path | Dense TFLOPS/TOPS | With 2:4 sparsity |
|---|---|---|
| FP16 in, **FP16 accumulate** | 142 | 284 |
| FP16 in, **FP32 accumulate** | **71** | 142 |
| BF16 in, FP32 accumulate | 71 | 142 |
| TF32 | 35.6 | 71 |
| INT8 | 284 | 568 |
| INT4 | 568 | 1136 |

Notes:

- The FP32-accumulate half-rate applies to the whole GA10x line including RTX A6000/A40 — it is a GA10x trait, not a GeForce-driver limitation.
- A100 (GA100, sm_80) runs FP16→FP32 accumulate at full rate (312 TFLOPS dense).
- BF16 inputs REQUIRE FP32 accumulation in `mma.sync` PTX, so BF16 on GA10x is always on the half-rate path. FP16 with FP16 accumulate is the only full-rate floating path — usable for attention P·V under careful scaling, risky for general GEMM.
- INT8 (284 TOPS) is 4x the FP32-accumulate FP16 rate — why weight-only INT8/INT4 (Marlin-style) and SmoothQuant-style W8A8 shine on 3090-class cards.

## Architectural notes relevant to kernels

- GA10x SM: 4 processing blocks (warp schedulers), 128 KB unified L1/SMEM, 64K 32-bit registers per SM (255 regs/thread max), 4 tensor cores.
- Datapath: both 64-lane FP32 pipes usable → 128 FP32 FMA/clk/SM; one pipe shares with INT32.
- PCIe 4.0 x16 host link. RTX 3090/3090 Ti, RTX A6000, and A40 support a 2-way NVLink bridge; RTX 3080 and below do not. Multi-GPU rigs without bridges fall back to P2P over PCIe.
- 2:4 structured sparsity supported by the sparse `mma.sp` path.
