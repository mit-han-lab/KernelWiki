---
id: doc-nvidia-tuning-guide
title: "NVIDIA Blackwell Tuning Guide"
url: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
source_category: official-doc
architectures: [sm100, sm120]
tags: [cluster]
retrieved_at: 2026-08-18
---

# NVIDIA Blackwell Tuning Guide

This source page records the scope of NVIDIA’s Blackwell Tuning Guide as accessed on 2026-08-18. The guide covers application compatibility and CUDA execution/resource guidance for Blackwell compute capabilities, including device memory, shared memory, occupancy, clusters, and compiler targeting.

It is not the instruction-level authority for `tcgen05`, TMEM allocation, CLC response syntax, or matrix-descriptor encoding. Those claims must cite the [PTX ISA source page](nvidia-ptx-isa-sm100.md) or an official CUTLASS programming document.

## Stable guidance used by this wiki

- Recompile or provide compatible device code for the intended compute capability rather than assuming a Hopper binary exposes Blackwell-specific features.
- Re-evaluate occupancy and launch bounds with the target device’s actual register and shared-memory limits.
- Treat thread-block clusters as an explicit launch/resource choice; cluster size and residency affect portability and performance.
- Query device properties at runtime when a decision depends on the installed GPU rather than copying a product-wide constant into scheduling code.

Historical or product-specific numbers should carry their own locator and access date. The rolling tuning guide alone does not establish benchmark throughput or the behavior of an undocumented instruction form.
