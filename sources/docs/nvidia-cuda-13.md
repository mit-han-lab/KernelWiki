---
id: doc-cuda-13
title: NVIDIA CUDA Toolkit 13.0 announcement
url: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [cuda-cpp]
retrieved_at: 2026-08-16
---

# CUDA Toolkit 13.0 announcement

NVIDIA published the CUDA 13.0 announcement on 2025-08-06. It says Blackwell was first supported in CUDA 12.8 and that CUDA 13.0 continues support across the then-current Blackwell product families.

The announcement's main platform items include:

- the CUDA Tile IR foundation for a tile-based programming model;
- a unified toolkit for server-class and embedded Arm platforms;
- CCCL 3.0 and its C++17 requirement;
- Zstandard as the new default fatbin compression scheme;
- compiler, library, and Nsight tool updates; and
- 32-byte alignment changes for selected 256-bit vector types on Blackwell.

## Evidence boundary

The former local page presented invented inline-PTX operands, non-existent `clc.arrive`/`clc.wait` instructions, an invalid TMEM allocation example, unsupported FP4 C++ type names, and a claim that PDL is enabled by default. The CUDA 13.0 announcement does not establish those details. Instruction syntax is recorded instead in `doc-ptx-isa-sm100`, with exact PTX versioning.

Primary source: [NVIDIA CUDA 13.0 announcement](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/).
