---
id: blog-yue-nvfp4
title: Blackwell NVFP4 Kernel Hackathon Journey
author: Yue Zhang
url: https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html
source_category: community-note
architectures: [sm100]
tags: [nvfp4, gemv, fp4, block-scale, batched-gemv]
techniques: [vectorized-loads, register-reuse, loop-unrolling]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx, cute-dsl]
retrieved_at: 2026-08-18
---

# Blackwell NVFP4 Kernel Hackathon Journey

Yue Zhang's worklog covers GPU Mode's B200 NVFP4 batched-GEMV task. The input
uses packed E2M1 values and E4M3 scale factors shared by groups of 16 elements.

## Source-reported progression

| Implementation | Time |
|---|---:|
| Organizer CuTe DSL template | about 100 microseconds |
| Author's optimized CuTe DSL | about 33 microseconds |
| Naive CUDA | 2000 microseconds |
| CUDA with coalesced access and thread collaboration | 443 microseconds |
| CUDA with hardware conversion intrinsics | 39 microseconds |
| CUDA with inline PTX | about 27 microseconds |
| CUDA with two-tile ILP | about 22.9 microseconds |
| CUDA with a larger fused PTX block | about 22.3 microseconds |

The article separately reports a final leaderboard submission of
22.392 microseconds. All values are author-reported for the contest's geometric-mean
metric.

The CuTe path removes redundant scale loads, reduces register precision for
loaded operands, reuses scale products, and introduces thread collaboration.
The CUDA path moves from manual decode to vectorized loads and hardware
conversion, then uses inline PTX for byte unpacking and packed conversion. Later
steps tune launch parameters, process two K tiles per iteration, and fuse more
decode/scale/arithmetic instructions into one inline-PTX block.

The worklog says larger unroll factors and attempted asynchronous
double-buffering did not help this implementation. It does not establish that
PTX is required, that manual CUDA generally beats CuTe DSL, or that these timing
deltas transfer to other shapes.
