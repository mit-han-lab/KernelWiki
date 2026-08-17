---
id: blog-amandeep-nvfp4
title: Twelve Attempts at an FP4 Kernel
author: Amandeep Singh
url: https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/
source_category: community-note
architectures: [sm100, sm100a]
tags: [nvfp4, gemv, fp4, block-scale, batched-gemv]
techniques: [vectorized-loads, cache-policy, register-budgeting, per-k-specialization, data-reuse]
hardware_features: [nvfp4, fp4, block-scale]
kernel_types: [batched-gemv, gemv]
languages: [cuda-cpp, ptx]
retrieved_at: 2026-08-16
---

# Twelve Attempts at an FP4 Kernel

This is Amandeep Singh's post-hackathon worklog for GPU Mode's NVFP4 GEMV task. It reports per-shape latency for the author's raw-CUDA baseline, not a 26.7-microsecond suite geometric mean:

| M | K | L | Author kernel | Reported bandwidth bound | Ratio |
|---:|---:|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 26.7 µs | 8.6 µs | 3.1× |
| 4096 | 7168 | 8 | 45.1 µs | 17.3 µs | 2.6× |
| 7168 | 2048 | 4 | 16.4 µs | 4.3 µs | 3.8× |

The worklog says attempts 8–12—split-K, wider C++ loads, extra accumulator chains, moderate register/block tuning, and manual software pipelining—regressed or did not improve this kernel. It then analyzes public top solutions at roughly 18.5-microsecond geometric mean, including raw PTX load/decode paths, 128/256-bit loads coupled to byte unpacking, tighter register caps, and compile-time specialization. It also reports a separate multi-stream `torch._scaled_mm` entry at 22.4 microseconds.

These are the author's measurements and retrospective attributions. They do not show that a cache hint, vector width, or register cap will transfer unchanged to another shape. The former local record invented a twelve-stage latency progression and incorrectly treated wider loads and `-maxrregcount=40` as the author's successful steps; those claims contradicted the source and are removed.

Primary source: [Amandeep Singh, “Twelve Attempts at an FP4 Kernel”](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/).
