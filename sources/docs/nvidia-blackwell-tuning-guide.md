---
id: doc-nvidia-tuning-guide
title: NVIDIA Blackwell Tuning Guide
url: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [cuda-cpp]
retrieved_at: 2026-08-16
---

# NVIDIA Blackwell Tuning Guide

## Scope

The official guide describes application tuning and resource limits for compute capabilities 10.0 and 12.0. It is not the PTX instruction reference for `tcgen05`, TMEM, TMA, or Cluster Launch Control; those claims belong to the PTX ISA, CUDA Programming Guide, Driver API, or CUTLASS documentation.

## Compute capability 10.0 limits stated by the guide

- at most 64 concurrent warps per SM;
- 64K 32-bit registers per SM;
- at most 32 thread blocks per SM;
- 228 KB shared-memory capacity per SM; and
- at most 227 KB shared memory per thread block.

The guide also says B200's combined L1/texture/shared-memory capacity is 256 KB. These are architectural ceilings, not an occupancy promise: register allocation, shared memory, block size, clusters, and other launch constraints determine actual residency.

## Evidence boundary

The former local summary mixed in unsupported values and claims, including a 420-cycle TMEM latency, universal MMA shapes, “PDL enabled by default,” “near-zero” launch gaps, and a step-by-step 98%-of-cuBLAS community result. None came from this guide and they have been removed from this source record.

Primary source: [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
