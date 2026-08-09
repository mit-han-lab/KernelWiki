---
id: blog-amandeep-nvfp4
title: Twelve Attempts at an FP4 Kernel
author: Amandeep Singh
url: https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/
source_category: community-note
architectures:
- sm100
- sm100a
tags:
- nvfp4
- gemv
- fp4
- block-scale
- batched-gemv
techniques:
- vectorized-loads
- cache-policy
- register-budgeting
- per-k-specialization
- data-reuse
hardware_features:
- nvfp4
- fp4
- block-scale
kernel_types:
- batched-gemv
- gemv
languages:
- cuda-cpp
- ptx
- cute-dsl
retrieved_at: 2026-08-08
---

# Twelve Attempts at an FP4 Kernel

## Evidence Scope

This card summarizes Amandeep Singh's own retrospective and public code repository. The post distinguishes the author's measured attempts from techniques the author says were observed later in three other solutions. Those observations are not independently reproduced contestant-source evidence.

## Attempt 7 Baseline

After several CuTe DSL experiments, attempt 7 rewrote the kernel in CUDA C++. It assigned one warp to each output row, used four rows per 128-thread block, converted packed FP4 pairs with Blackwell intrinsics, and reduced each row with warp shuffles.

The post reports these per-configuration times, not a 26.7 microsecond geometric mean:

| M | K | L | Author kernel (µs) | Task model (µs) |
|---:|---:|---:|---:|---:|
| 7168 | 16384 | 1 | 26.7 | 8.6 |
| 4096 | 7168 | 8 | 45.1 | 17.3 |
| 7168 | 2048 | 4 | 16.4 | 4.3 |

## Attempts 8–12

The next five attempts did not form a monotonic optimization progression:

- Attempt 8 used split-K plus FP32 atomics and was worse because of contention, extra traffic, and scheduling overhead.
- Attempt 9 replaced two `uchar4` loads with one `uint2` load and was 16–25% slower because byte extraction added instructions.
- Attempt 10 used four accumulator chains and regressed by 32–55% with more register pressure and worse coalescing.
- Attempt 11 found no effect from reducing `-maxrregcount` from 80 to 64 and no effect from the tested block-size change; unroll 8 was worse than unroll 4.
- Attempt 12's explicit software pipeline increased register pressure and was slower.

The author's main retrospective lesson is to run Nsight Compute early and verify the bottleneck before choosing transformations.

## Post-event Observations

The author says three inspected solutions around an 18.5 microsecond aggregate used raw PTX load/decode paths, A `L1::no_allocate`, B `L1::evict_last`, `v2.u64` or `v4.u64` loads with `mov.b32` byte decomposition, exact-K specializations, and tighter register budgets. The post reports 32 registers for one inspected solution and 45 for another, and says a further solution shared B reads across multiple M rows.

These are author-reported observations. They do not supply controlled ablations, public contestant code, or proof that lower register caps and wider loads monotonically improve performance. The author's own attempts provide counterexamples to both generalizations.

## Primary Sources

- [Amandeep Singh, “Twelve Attempts at an FP4 Kernel”](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [Public attempt repository](https://github.com/amandeepsp/cuda/tree/44513ac7d5bbd1cf8109cab952844adac5b6c551/nvfp4/gemv)
