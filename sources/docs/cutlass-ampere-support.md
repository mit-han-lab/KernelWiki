---
id: doc-cutlass-ampere
title: "CUTLASS Ampere (SM80/SM86) Support: MmaMultistage, CuTe SM80 Atoms, 3xTF32"
url: https://docs.nvidia.com/cutlass/
source_category: official-doc
architectures: [sm80, sm86]
tags: [cp-async, mma-sync, ldmatrix, swizzling, double-buffering, pipeline-stages]
retrieved_at: 2026-08-03
---

# CUTLASS Ampere (SM80/SM86) Support

## Overview

CUTLASS is the reference implementation for high-performance Ampere GEMM. Its SM80 code path is the canonical public example of the full Ampere idiom: multi-stage `cp.async` software pipeline feeding warp-scope `mma.sync` tensor core ops via `ldmatrix`, with swizzled shared-memory layouts. Anyone backporting Hopper/Blackwell kernel structure to sm_86 should read the CUTLASS SM80 mainloop first — it IS the target shape of the port.

## The SM80 mainloop: MmaMultistage

CUTLASS 2.x introduced `cutlass::gemm::threadblock::MmaMultistage` — the Ampere replacement for the Volta/Turing 2-stage `MmaPipelined`:

- **N-stage circular SMEM buffer** (typically `Stages = 3..5`, bounded by SMEM: 99 KB/block on sm_86, 163 KB on sm_80).
- Global→shared operand movement issued with `cp.async` (`CacheOperation::Global` → `.cg` 16B for main operands, `.ca` for small fragments).
- Completion via `cp.async.wait_group`/commit-group counting; no mbarriers in the classic mainloop.
- Warp-level tile compute in `cutlass::gemm::warp::MmaTensorOp`, which lowers to `mma.sync.aligned.m16n8k16` (arch `cutlass::arch::Mma<GemmShape<16,8,16>, 32, ...>`) with `ldmatrix`-based operand fetch (`cutlass::arch::LdMatrix`).
- Shared-memory layouts use XOR-swizzles (`TensorOpMultiplicandCrosswise` etc.) sized to keep `ldmatrix` bank-conflict-free.

## CuTe (C++) SM80 atoms

CUTLASS 3.x re-expresses the same hardware in CuTe:

- MMA atoms: `SM80_16x8x16_F32F16F16F32_TN`, `SM80_16x8x16_F32BF16BF16F32_TN`, `SM80_16x8x8_F32TF32TF32F32_TN`, INT8 variants — all wrap `mma.sync`.
- Copy atoms: `SM80_CP_ASYNC_CACHEALWAYS<T>` / `SM80_CP_ASYNC_CACHEGLOBAL<T>` (cp.async .ca/.cg), `SM75_U32x4_LDSM_N` / `SM75_U16x8_LDSM_T` (ldmatrix, non-transposed/transposed).
- Tutorial `examples/cute/tutorial/sgemm_sm80.cu` walks the full pattern; `cute::cp_async_fence()` / `cute::cp_async_wait<N>()` expose commit-group semantics.
- CollectiveBuilder path: SM80 collectives use the multistage schedule (no warp specialization — contrast with SM90 `KernelTmaWarpSpecialized*` schedules).

## Ampere-specific examples in the CUTLASS tree

| Example | What it demonstrates |
|---|---|
| `14_ampere_tf32_tensorop_gemm` | TF32 tensor-core GEMM on FP32 data |
| `15_ampere_sparse_tensorop_gemm` | 2:4 structured sparsity (`mma.sp`) |
| `27_ampere_3xtf32_fast_accurate_tensorop_gemm` | **3xTF32**: emulate FP32 GEMM with 3 TF32 MMAs (error-compensated), ~2x+ over FP32 CUDA cores at near-FP32 accuracy |
| `sgemm_sm80.cu` (CuTe tutorial) | Minimal readable multistage cp.async + mma.sync mainloop |

Note: the newer Python CuTe DSL examples tree is Hopper/Blackwell-first; check the current `examples/python/CuTeDSL/` tree before assuming SM80 DSL coverage. The mature, performance-proven Ampere path is the C++ one above.

## Practical notes for sm_86 specifically

- CUTLASS kernels compiled for sm_80 run on sm_86, but tile shapes tuned for A100's 164 KB SMEM will fail to launch or spill on GA10x's 99 KB — pick smaller `Stages` or tile sizes (the profiler's sm_86 presets do this).
- Threadblock shapes that assume 2048 threads/SM occupancy targets should be re-tuned for 1536.
- FP16 accumulation (`ElementAccumulator = half_t`) doubles tensor throughput on GA10x (see doc-ga102-whitepaper) at the usual numerical risk; CUTLASS supports it natively.
