---
id: doc-ptx-isa-ampere
title: "PTX ISA Ampere (sm_80/sm_86) Instructions Reference"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
source_category: official-doc
architectures: [sm80, sm86]
tags: [ptx, cp-async, mma-sync, ldmatrix, mbarrier, cache-policy]
retrieved_at: 2026-08-03
---

# PTX ISA Ampere (sm_80/sm_86) Instructions Reference

## Overview

PTX ISA 7.0+ introduces the sm_80-generation instructions that define Ampere kernel programming: asynchronous global→shared copies (`cp.async`), warp-scope synchronous tensor core MMA (`mma.sync` with the m16n8kX shapes), shared-memory matrix loads (`ldmatrix`), and shared-memory barriers (`mbarrier`). This page summarizes syntax and constraints as documented in the PTX ISA, for use when writing sm_86 kernels or backporting sm_90/sm_100 code.

## cp.async — asynchronous copy (sm_80+)

```asm
cp.async.ca.shared.global [smem_dst], [gmem_src], cp-size{, src-size};   // cp-size ∈ {4, 8, 16}
cp.async.cg.shared.global [smem_dst], [gmem_src], 16{, src-size};       // 16 bytes only
cp.async.commit_group;
cp.async.wait_group N;      // wait until ≤ N groups in flight
cp.async.wait_all;
```

- `.ca` allocates in L1 and L2; `.cg` allocates in L2 only (use for tensor-core operand tiles).
- Optional `src-size` (< cp-size) reads fewer bytes and **zero-fills** the remainder of the shared destination — the OOB-guard idiom without branches.
- An optional cache-policy operand (`createpolicy.fractional.L2::evict_first` etc.) can be attached (`cp.async.ca.shared.global.L2::cache_hint`).
- Completion alternatives: commit-group counting (above) or `cp.async.mbarrier.arrive.shared.b64 [mbar];` which makes an mbarrier track the async group.
- Destination must be `.shared` state space, addresses must be naturally aligned to cp-size.

## mma.sync — warp-scope tensor core MMA (shapes per dtype)

```asm
// FP16 inputs, FP32 accumulator (the LLM workhorse):
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f0,%f1,%f2,%f3},        // D: 4x f32 per thread
    {%r0,%r1,%r2,%r3},        // A: 4x b32 (8x f16) per thread
    {%r4,%r5},                // B: 2x b32 (4x f16) per thread
    {%f0,%f1,%f2,%f3};        // C (accumulate in place)
```

Key shapes available on sm_80/sm_86:

| Inputs | Accumulator | Shapes |
|---|---|---|
| f16 | f16 or f32 | m16n8k8, m16n8k16 |
| bf16 | **f32 only** | m16n8k8, m16n8k16 |
| tf32 | f32 | m16n8k4, m16n8k8 |
| s8/u8 | s32 | m16n8k16, m16n8k32 |
| s4/u4 | s32 | m16n8k32, m16n8k64 |
| f64 | f64 | m8n8k4 |

- All operands live in **registers**, fragmented across the 32 lanes of one warp in a fixed documented layout; accumulators stay in registers between iterations (this is what TMEM replaces on Blackwell).
- `mma.sync` is **synchronous at warp scope**: the issuing warp stalls until the result lands — there is no `wgmma.mma_async`/`tcgen05.mma` style asynchronous tensor op before sm_90.
- Sparse variant `mma.sp` implements 2:4 structured sparsity with a metadata operand.
- Legacy `wmma.*` (m16n16k16) remains available but the m16n8kX `mma.sync` family is the canonical high-performance path (what CUTLASS emits).

## ldmatrix — shared→register matrix fragment load (sm_75+)

```asm
ldmatrix.sync.aligned.m8n8.x4.shared.b16      {%r0,%r1,%r2,%r3}, [smem_addr];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r0,%r1,%r2,%r3}, [smem_addr];
```

- Loads 1/2/4 8x8 b16 tiles per instruction into the exact register fragment layout `mma.sync` expects; `.trans` transposes at load (needed for column-major B operands).
- Each of the 32 lanes supplies a row-start address (lane i → row i of the tiles); addresses typically computed with a swizzle to avoid bank conflicts.
- **`stmatrix` (register→shared store) is sm_90+.** On Ampere the epilogue writes accumulators to shared/global with ordinary `st` instructions — one of the notable asymmetries versus Hopper.

## mbarrier — shared-memory barrier (sm_80+)

```asm
mbarrier.init.shared.b64          [mbar], count;
mbarrier.arrive.shared.b64   %r0, [mbar];
mbarrier.test_wait.shared.b64 %p, [mbar], %r0;   // poll
mbarrier.try_wait.shared.b64  %p, [mbar], %r0;   // blocking-ish (sm_90 adds .parity forms used with TMA)
```

- Ampere mbarriers support init/arrive/test_wait/invalidate plus `cp.async.mbarrier.arrive` integration; the C++ view is `cuda::barrier<cuda::thread_scope_block>`.
- **`mbarrier.arrive.expect_tx` and transaction-count tracking are sm_90+** (designed for TMA byte counting) — an Ampere backport must count arrivals, not bytes.

## Cache policy / eviction hints (sm_80+)

```asm
createpolicy.fractional.L2::evict_first.b64 %pol, 1.0;
ld.global.L2::cache_hint.f32 %f0, [gaddr], %pol;
ld.global.L1::no_allocate.v4.f32 {...}, [gaddr];   // streaming loads that skip L1
```

- `evict_first` / `evict_last` / `no_allocate` policies let streaming tensors bypass or minimally pollute caches — pairs with the L2 persistence window set on the host side.
- `ld.global.nc` (`__ldg`) non-coherent constant-cache path remains available.

## Other sm_80-relevant instructions

- `redux.sync.op.u32` — warp-level reduction in one instruction (int only on Ampere).
- `cvt` with `.rn` packed conversions f32↔bf16x2 (`cvt.rn.bf16x2.f32`) for epilogue downcasts.
- `bar.warp.sync`, `shfl.sync`, standard since Volta, unchanged.
- NO `cp.async.bulk*`, NO `tensormap.*`, NO `wgmma.*`, NO `tcgen05.*`, NO `stmatrix`, NO `setmaxnreg`, NO `clusterlaunchcontrol.*` on sm_8x — attempting to compile any of these for sm_86 is an immediate ptxas error (the fastest way to inventory what a Hopper kernel needs replaced).
