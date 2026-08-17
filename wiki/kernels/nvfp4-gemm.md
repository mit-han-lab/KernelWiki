---
id: kernel-nvfp4-gemm
title: NVFP4 GEMM — 4-bit Floating Point Matrix Multiply
type: kernel
architectures: [sm100, sm100a]
tags: [gemm, nvfp4, fp4, block-scale, tcgen05, tmem, warp-specialization]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, cute-dsl, ptx]
related: [hw-nvfp4, hw-tcgen05-mma, hw-tmem, kernel-nvfp4-gemv, technique-warp-specialization]
sources: [contest-gpumode-p2, doc-cutlass-blackwell, doc-ptx-isa-sm100, pr-cutlass-2139]
performance_claims: []
artifact_dir: artifacts/kernels/nvfp4-gemm
---

# NVFP4 GEMM

## Numerical format

NVFP4 pairs signed E2M1 data with unsigned E4M3 (`ue4m3` in PTX/CUTLASS terminology) scale factors. One scale applies to 16 consecutive dense K elements. This differs from OCP MXFP4, which uses UE8M0 scales and a 32-element scale-vector size.

SM100 executes NVFP4 through the block-scaled `tcgen05.mma` family associated with `kind::mxf4nvf4` (also named `nvf4mxf4` in some CUTLASS documentation/release-era APIs), with scale-factor IDs and TMEM layouts encoded by the instruction descriptor. NVFP4 E4M3 scales are not rounded to UE8M0 before this MMA; doing so would change the represented operand.

## Logical operation

For scale-vector size 16, the block-scaled product is:

```python
def nvfp4_reference(a_q, a_scale, b_q, b_scale, m, n, k):
    total = 0.0
    for kk in range(k):
        a = decode_e2m1(a_q[m, kk]) * decode_ue4m3(a_scale[m, kk // 16])
        b = decode_e2m1(b_q[n, kk]) * decode_ue4m3(b_scale[n, kk // 16])
        total += a * b
    return total
```

Physical scale tensors are not stored as a simple row-major `[M, K/16]` array in optimized CUTLASS kernels. CUTLASS documents a 512-byte basic-block layout and provides `Sm1xxBlockScaledConfig`/CuTe helpers to construct it.

## Kernel pipeline

A typical CUTLASS SM100 NVFP4 kernel uses:

- TMA to stage packed E2M1 operands and scale-factor tiles;
- configuration-selected shared-memory layouts and stage count;
- a single MMA-issue warp issuing the block-scaled operation into TMEM;
- `tcgen05.commit`/`mbarrier` completion before epilogue TMEM loads;
- a configuration-dependent epilogue thread count.

The CUTLASS schedule `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` exists in the captured PR-2139 code, but it is a pointer-array schedule and is not automatically the right device API for every dense single-problem GEMM.

## Alignment and shape constraints

TMA tensor maps and the selected MMA/layout impose alignment, bounding-box, leading-dimension, K-divisibility, and scale-layout constraints. They do not imply that every operand base must be 128-byte aligned or that every NVFP4 K must be a multiple of 256. Query the chosen CUTLASS kernel's `can_implement` result or satisfy the exact PTX/Driver-API constraints, and keep a boundary fallback where required.

TMEM consumption must be obtained from the accumulator/scale layouts (for example, CUTLASS's `get_num_tmem_alloc_cols` helper), not inferred by treating TMEM as a row-major 128-by-N FP32 matrix.

## Contest result boundary

The task file defines three benchmark shapes and a geometric-mean ranking rule;
the separate unauthenticated leaderboard API reports an ended leaderboard but not
implementation details. Its 2026-08-16 snapshot showed `gau.nernst`, `s.am._`,
and `billcarson` at ranks 1–3 with 9.981889, 10.060110, and 10.137411 us.
The old 10.807/10.914/10.931-us values were genuine but were ranks 8–10 for
`Simon`, `yue`, and `currybab`, not the podium. All displayed submissions
precede the 2025-12-21 deadline; the snapshot is not an independent award claim.

## Full Reference Implementation

The reference bundle in [`artifacts/kernels/nvfp4-gemm/full/`](../../artifacts/kernels/nvfp4-gemm/full/) contains a pinned CUTLASS patch. It is upstream implementation evidence, not an unavailable contest submission.
