---
id: doc-ampere-tuning-guide
title: "NVIDIA Ampere GPU Architecture Tuning Guide (CC 8.0 / 8.6)"
url: https://docs.nvidia.com/cuda/ampere-tuning-guide/
source_category: official-doc
architectures: [sm80, sm86]
tags: [cuda-cpp, cp-async, l2-persistence, mbarrier]
retrieved_at: 2026-08-03
---

# NVIDIA Ampere GPU Architecture Tuning Guide (CC 8.0 / 8.6)

## Overview

The official NVIDIA Ampere tuning guide covers both Ampere compute capabilities: **8.0 (GA100: A100/A800)** and **8.6 (GA10x: RTX 3090/3080, RTX A6000, A40, A10)**. The two are NOT the same architecture from a tuning standpoint — occupancy ceilings, shared memory capacity, and FP32 issue rate all differ. This page records the facts that matter when writing or backporting kernels for sm_86.

## CC 8.0 vs CC 8.6 — the table that matters

| Resource | CC 8.0 (A100) | CC 8.6 (GA10x) |
|---|---|---|
| Max concurrent warps / SM | 64 (2048 threads) | **48 (1536 threads)** |
| Max thread blocks / SM | 32 | **16** |
| Unified L1/SMEM capacity / SM | 192 KB | **128 KB** |
| Max SMEM carveout / SM | 164 KB | **100 KB** |
| Max SMEM per thread block | 163 KB | **99 KB** |
| FP32 ops per cycle per SM | 64 | **128 (2x)** |

Implications:

- **Occupancy math changes on sm_86.** A block size that yields 100% occupancy on A100 (e.g., 2048 threads via 2x1024) caps at 75% of the sm_80 figure on GA10x. Blocks of 256/512 threads with ≤3 blocks resident hit the 1536-thread ceiling exactly.
- **SMEM-heavy Hopper-style kernels do not fit.** 99 KB per block is the hard ceiling on sm_86 (vs 163 KB on A100, 227 KB on H100). Multi-stage pipelines must shrink stage count or tile size accordingly.
- **The 2x FP32 pipe** (both datapaths can issue FP32; on GA100 one is FP32-only, the other INT32) means non-tensor-core elementwise/epilogue code is comparatively cheap on GA10x.

## Asynchronous Data Copies (cp.async)

Ampere introduces asynchronous copy from global to shared memory, bypassing the register file and (optionally) L1:

- `cp.async.ca.shared.global` — 4, 8, or 16 bytes, caches in L1.
- `cp.async.cg.shared.global` — 16 bytes only, caches only in L2 (bypasses L1). Recommended for tensor-core operand staging.
- Completion managed either via **commit-group semantics** (`cp.async.commit_group` / `cp.async.wait_group N`) or via **asynchronous barriers** (`cuda::barrier`, PTX `mbarrier`, available since sm_80).
- The C++ surface is `cuda::memcpy_async` + `cuda::pipeline` (libcudax / `<cuda/pipeline>`).

This is the Ampere ancestor of Hopper TMA: per-thread addressing, no bulk tensor descriptors, no multicast, no swizzle-on-the-fly — but the same "load directly to SMEM without burning registers" principle.

## L2 Persistence (Access Policy Window)

CC 8.0+ can set aside a portion of L2 for **persisting** accesses via `cudaAccessPolicyWindow` (per-stream or per-graph-node attribute):

```cpp
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = kv_cache_ptr;
attr.accessPolicyWindow.num_bytes = window_bytes;          // <= cudaLimitMaxL2FetchGranularity window
attr.accessPolicyWindow.hitRatio  = 0.6f;                   // fraction treated as persisting
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

`cudaDeviceProp::persistingL2CacheMaxSize` reports the maximum set-aside. Useful for small hot operands re-read across many CTAs (e.g., decode-time KV heads, router weights). Streaming accesses can be marked `evict_first` via PTX cache policies to avoid polluting the persisting region.

## Other sm_80+ features recorded by the guide

- **Asynchronous barriers** (`cuda::barrier<cuda::thread_scope_block>`): arrive/wait split, enables producer/consumer overlap without `__syncthreads()` full-block convergence.
- **Warp reduce instructions** (`redux.sync`) for int operands.
- **Improved L2 residency management + larger L2** vs Volta/Turing.
- **Third-generation tensor cores**: BF16 and TF32 inputs; `mma.sync` shapes up to m16n8k16 (FP16/BF16) and m16n8k8 (TF32).

## What Ampere does NOT have (forward-looking)

Recorded here because this KB is used for Hopper/Blackwell backports; all of the following first appear in sm_90 or sm_100:

- No TMA (`cp.async.bulk.tensor`), no multicast loads.
- No thread block clusters, no distributed shared memory (DSMEM).
- No `wgmma` / `tcgen05` asynchronous tensor core instructions — only synchronous warp-scope `mma.sync`.
- No TMEM, no `setmaxnreg` register reallocation, no `stmatrix` (sm_90+), no PDL/GDC kernel-launch overlap.
- No FP8/FP6/FP4 tensor core datatypes (INT8/INT4 via `mma.sync` are available).
