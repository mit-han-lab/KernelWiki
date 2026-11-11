---
id: doc-nvidia-hopper-compatibility-guide
title: "NVIDIA Hopper Compatibility Guide (CUDA 12.8.1)"
author: NVIDIA
url: https://docs.nvidia.com/cuda/archive/12.8.1/hopper-compatibility-guide/index.html
source_category: official-doc
architectures: [sm90, sm90a]
tags: [ptx, cuda-cpp]
retrieved_at: 2026-11-11
---

# NVIDIA Hopper Compatibility Guide (CUDA 12.8.1)

The archived guide distinguishes native cubins, ordinary virtual PTX, and
architecture-conditional code:

- A cubin is compatible within a compute-capability major revision when the
  target GPU's minor revision is the same or higher, not across major
  revisions.
- Ordinary PTX can be JIT-compiled for its compute capability or a higher
  capability, subject to the driver's support for that PTX version.
- `sm_90a` / `compute_90a` targets enable architecture-conditional features;
  those cubins and PTX are not forward- or backward-compatible. A generic PTX
  fallback must not be confused with an architecture-conditional fallback.

H20's compute capability 9.0 is independently established by the NVIDIA MIG
guide. This compatibility guide is not an H20 product catalog or an
instruction-by-instruction target list; consult the PTX ISA for WGMMA target
requirements.

The guide describes `CUDA_FORCE_PTX_JIT=1` as a way to test an application's
embedded PTX path. Such a successful test demonstrates compatibility for the
executed kernels, not optimal native performance or coverage of unexecuted
dispatch paths.
