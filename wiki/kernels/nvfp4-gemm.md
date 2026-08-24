---
id: kernel-nvfp4-gemm
title: NVFP4 GEMM
type: kernel
architectures: [sm100, sm100a]
tags: [gemm, nvfp4, fp4, block-scale, tcgen05, tmem, warp-specialization]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp]
related: [hw-nvfp4, hw-tcgen05-mma, hw-tmem, kernel-nvfp4-gemv]
sources: [contest-gpumode-p2, doc-cutlass-blackwell, pr-cutlass-2139]
performance_claims: []
blackwell_relevance: CUTLASS has distinct SM100 NVFP4 block-scaled schedules; using one still requires its exact operand, scale-layout, alignment, and descriptor contracts.
---

# NVFP4 GEMM

The retained GPU Mode workload multiplies block-scaled NVFP4 matrices on B200
and produces FP16 output. Its task prose labels each per-16 scale
`fp8(e4m3fnuz)`, but the organizer's executable reference constructs the scale
tensors as `torch.float8_e4m3fn`. The two organizer artifacts therefore agree
on granularity but not on the signed FP8 encoding name. The organizer API is
the authority for shapes and its live leaderboard snapshot; the reference code
is the executable semantics.

## Retained CUTLASS anchor

CUTLASS PR 2139 added distinct one-SM and two-SM schedule tags for NVFP4. This
contiguous excerpt from its captured `dispatch_policy.hpp` is a configuration
anchor, not a complete GEMM:

```cpp
struct KernelTmaWarpSpecialized1SmNvf4Sm100 final
    : KernelSchedule1Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized2SmNvf4Sm100 final
    : KernelSchedule2Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100 final
    : KernelSchedule1Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100 final
    : KernelSchedule2Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
```

Use the corresponding CUTLASS builder/example to derive legal alignments,
layouts, scale descriptors, tile shapes, and cluster shapes. The former local
kernel skeleton was removed because it invented TMA, TMEM-allocation, and MMA
calls and incorrectly converted E4M3 block scales to UE8M0.

## Reproduction checklist

1. Match the organizer's E2M1 packing, per-16 FP8 scale tensors, output type,
   and reference tolerances; resolve the disclosed task-prose E4M3FNUZ versus
   reference-code `torch.float8_e4m3fn` discrepancy against the pinned contest
   revision.
2. Select a documented CUTLASS NVFP4 schedule and satisfy the builder's reported
   alignment/layout constraints instead of assuming every operand needs one
   universal byte alignment.
3. Test tails and every benchmark shape, and report the live leaderboard time
   with its capture date rather than as a permanent ranking.
