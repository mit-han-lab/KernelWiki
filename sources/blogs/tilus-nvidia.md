---
id: blog-tilus-nvidia
title: "Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation"
author: NVIDIA
url: https://github.com/NVIDIA/tilus
source_category: community-note
architectures: [sm100]
tags: [nvfp4, fp4, fp6, fp8, gemm, swizzling, pipeline-stages, cute-dsl, ptx]
retrieved_at: 2026-08-18
---

# Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## Verified source boundary

Tilus is NVIDIA's research DSL for GPU kernels. Its current README describes
thread-block-level programming with tensors, explicit control over shared
memory and register tensors, low-precision types with arbitrary bit widths,
automatic tuning and caching, and a Python interface.

The release history in that README is more specific about architecture scope:
v0.1.0 initially supported Ampere, while v0.2.0 added Hopper and Blackwell and
linked a step-by-step Blackwell matmul tutorial. The repository also states
that Hidet IR and its runtime are used as the low-level target.

The earlier local summary expanded “arbitrary bit widths” into a made-up list
of semantic formats and claimed particular TMA, TMEM, cluster, vectorization,
and instruction-selection behavior without exact source locators. Those
details have been removed. Consult the versioned programming guide and example
code for the actual operations exposed by a given Tilus release.

## Paper Reference

The repository cites “Tilus: A Tile-Level GPGPU Programming Language for
Low-Precision Computation” in the ASPLOS 2026 proceedings and links
arXiv:2504.12984.

## Resources

- **GitHub**: https://github.com/NVIDIA/tilus
- **Documentation**: https://nvidia.github.io/tilus/
- **Paper**: https://arxiv.org/abs/2504.12984
- **ACM**: https://dl.acm.org/doi/10.1145/3760250.3762219
