---
id: doc-cuda-13
title: "NVIDIA CUDA Toolkit 13.3 overview"
url: https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates
source_category: official-doc
architectures: [blackwell]
tags: [cuda-cpp]
retrieved_at: 2026-08-18
---

# NVIDIA CUDA Toolkit 13.3 overview

NVIDIA announced CUDA Toolkit 13.3 on 2026-05-26. Its release highlights include
CUDA Tile C++ support in NVCC and NVRTC, the CompileIQ compiler auto-tuning
framework, and stable CUDA Python 1.0 APIs. The CUDA 13.3 release notes are the
authority for component versions, compatibility, resolved issues, and known
issues.

CUDA Toolkit 13.3 Update 1 followed on 2026-06-29 and is the current stable
toolkit update at this page's 2026-08-18 evidence cutoff. The date is pinned by
NVIDIA's Nsight Compute 2026.2.1 release-history entry, which identifies that
release as accompanying CUDA Toolkit 13.3 Update 1.

This release-level source is not an instruction reference. Exact
`tcgen05.*`, Tensor Memory, cluster launch control, mbarrier, conversion, and
cache-operation syntax belongs to the PTX ISA; runtime and tensor-map contracts
belong to the CUDA Programming Guide and APIs.

Earlier local versions of this page contained invented CLC instructions and
incorrect TMEM operand descriptions. Those snippets were removed rather than
attributed to a toolkit announcement.
