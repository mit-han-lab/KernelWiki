---
id: lang-ptx
title: "PTX Instructions for SM100"
type: language
tags: [ptx, tcgen05, tmem, tma, clc, mbarrier, nvfp4]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, lang-cuda-cpp]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, pr-cutlass-2139, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
---

## Overview

SM100 introduces `tcgen05`, Tensor Memory, and Cluster Launch Control instruction families. The examples below use syntax checked against PTX ISA 9.3. They are instruction-level fragments, not complete kernels: allocation, descriptors, barrier initialization, parity, uniformity, and target directives remain the caller's responsibility.

## `tcgen05` instructions

```ptx
// Allocate TMEM columns; the result address is written to shared memory.
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32
    [smem_tmem_addr], num_cols;
ld.shared.b32 tmem_addr, [smem_tmem_addr];

// Dense MMA. A and B are SMEM descriptors; D is in TMEM.
// Four zero mask registers leave all cta_group::1 output lanes enabled.
tcgen05.mma.cta_group::1.kind::f16
    [tmem_addr], desc_a, desc_b, idesc,
    {mask0, mask1, mask2, mask3}, enable_input_d;

// Copy a shaped SMEM matrix into TMEM.
tcgen05.cp.cta_group::1.128x256b [tmem_addr], sdesc;

// Track completion of preceding MMA/copy/shift operations from this thread.
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbarrier];

// After waiting for that mbarrier, order the dependent TMEM load.
tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.32x32b.x2.b32 {r0, r1}, [tmem_addr];
tcgen05.wait::ld.sync.aligned;

// Register-to-TMEM store and its completion wait.
tcgen05.st.sync.aligned.32x32b.x2.b32 [tmem_addr], {r0, r1};
tcgen05.wait::st.sync.aligned;

// All allocated TMEM must be released before exit.
tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, num_cols;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

All `tcgen05` instructions in a kernel must use the same CTA-group value. A `tcgen05.fence` orders operations but does not replace `tcgen05.commit` plus the mbarrier completion wait.

## Sub-byte conversions

```ptx
// Two f32 inputs to packed FP8 E4M3.
cvt.rn.satfinite.e4m3x2.f32 packed_fp8, a, b;

// Packed FP8 or FP4 to packed f16x2.
cvt.rn.f16x2.e4m3x2 f16_pair, packed_fp8;
cvt.rn.relu.f16x2.e2m1x2 f16_pair, packed_fp4;

// Packed f16x2 to packed FP4 E2M1.
cvt.rn.satfinite.e2m1x2.f16x2 packed_fp4, f16_pair;
```

## Cache-control examples

```ptx
// Avoid allocating the line in L1. This is an eviction-priority hint,
// not a promise to bypass every cache level.
ld.global.L1::no_allocate.v2.u64 {r0, r1}, [addr];

// Prefer retaining a reused line in L1.
ld.global.L1::evict_last.v2.u64 {r0, r1}, [addr];

// A 256-bit vector load; natural alignment requirements still apply.
ld.global.v4.u64 {r0, r1, r2, r3}, [addr];
```

## Cluster Launch Control

CLC asynchronously attempts to cancel a cluster that has not launched; a successful response supplies the first CTA id of the canceled cluster for work stealing.

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    [response], [mbarrier];

// After waiting for the mbarrier and loading the 16-byte response:
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, _}, handle;
```

The CTA id must be queried only when `p` is true.

## TMA tensor copies

```ptx
// Global to this CTA's shared memory.
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.tile
    [smem_ptr], [tensor_map, {x, y}], [mbarrier];

// Global to cluster shared memory with cluster multicast.
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_ptr], [tensor_map, {x, y}], [mbarrier], cta_mask;
```

## Related
- [cuda-cpp](cuda-cpp.md) — Inline PTX in CUDA C++
- [tcgen05-mma](../hardware/tcgen05-mma.md) — MMA instruction details
- [nvfp4](../hardware/nvfp4.md) — FP4 format details
