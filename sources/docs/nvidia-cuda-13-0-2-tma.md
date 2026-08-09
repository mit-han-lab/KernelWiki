---
id: doc-cuda-13-0-2-tma
title: "CUDA 13.0.2 TMA Documentation"
url: https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma
source_category: official-doc
architectures: [sm90, sm90a, sm100, sm100a]
tags: [tma, mbarrier, swizzling]
retrieved_at: 2026-08-08
version: "CUDA 13.0.2"
---

# CUDA 13.0.2 TMA documentation

## Evidence-scoped summary

- TMA uses tensor maps for non-blocking rank-1 through rank-5 tensor copies on compute capability 9.0 and later.
- Global-to-shared loads complete through mbarrier transaction-byte accounting; shared-to-global stores use bulk async groups.
- Tensor-map swizzling changes shared-memory layout and must be paired with matching consumer indexing and alignment.
- The version-pinned Driver API defines encoder alignment, dimension, stride, datatype, interleave, swizzle, and OOB-fill constraints.
- Blackwell supports documented device-side tensor-map construction and modification with tensor-map proxy ordering.

The documentation defines mechanisms and constraints, not a universal stage depth, bandwidth multiplier, or best swizzle for every kernel.

## Primary references

- [CUDA 13.0.2 Programming Guide: TMA](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
- [CUDA Driver API 13.0.97: tensor maps](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
