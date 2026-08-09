---
id: lang-ptx
title: "PTX Instructions for SM100"
type: language
tags: [ptx, tcgen05, tmem, tma, clc, mbarrier, nvfp4]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, lang-cuda-cpp]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
---

# PTX Instructions for SM100

## Scope

This page uses CUDA 13.0.2's archived PTX ISA 9.0 grammar. The fragments are representative instruction forms, not complete inline-assembly functions: operands need correctly typed declarations, descriptors, collective issue, lifetime management, and the instruction-specific completion protocol.

## tcgen05 and Tensor Memory

Representative unscaled F16 forms include:

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [alloc_result_smem], ncols;
ld.shared.b32 taddr, [alloc_result_smem];

tcgen05.mma.cta_group::1.kind::f16 [taddr], a_desc, b_desc, idesc, enable_input_d;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];

tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, ncols;
```

`tcgen05.alloc` writes a 32-bit TMEM address to shared memory and counts columns. Allocation and deallocation are warp-collective for `cta_group::1`, every allocation must be deallocated before kernel exit, and all tcgen05 instructions in one kernel must use the same CTA-group value. The `cta_group::2` MMA form has the same unscaled operand grammar but can access paired-CTA resources.

TMEM/register transfers and shared-memory/TMEM copies have their own asynchronous completion rules:

```ptx
tcgen05.ld.sync.aligned.32x32b.x1.b32 {r0}, [taddr];
tcgen05.wait::ld.sync.aligned;

tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], {r0};
tcgen05.wait::st.sync.aligned;

tcgen05.cp.cta_group::1.128x256b [taddr], sdesc;
```

`tcgen05.ld` and `tcgen05.st` are warp-collective. `tcgen05.cp` copies a shaped shared-memory descriptor into TMEM. MMA and cp completion can be attached to an mbarrier with `tcgen05.commit`; the source and destination must remain live until their documented completion points.

## Completion and Ordering Are Distinct

- A TMA global-to-shared load completes bytes on its mbarrier; a consumer waits for that phase before reading the destination.
- `tcgen05.commit` makes an mbarrier track completion of prior asynchronous tcgen05 MMA/cp/shift operations issued by the thread.
- `tcgen05.wait::ld` and `tcgen05.wait::st` wait for the corresponding prior TMEM/register transfers.
- `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` constrain tcgen05 operations around a documented cross-thread execution-ordering handoff. They are not substitutes for TMA or MMA completion waits.

## Packed FP4 Conversion

PTX ISA 9.0 defines these typed operations:

```ptx
cvt.rn.f16x2.e2m1x2 result_f16x2, packed_fp4_pair;
mov.b32 {byte0, byte1, byte2, byte3}, packed_word;
```

The first converts one byte containing two E2M1 values into a 32-bit F16x2 result. The second can decompose a 32-bit scalar into four byte-sized destinations when operand declarations satisfy the scalar-to-vector size rules. PTX specifies these semantics but does not guarantee that the move is faster than every compiler-generated shift/mask sequence.

## Cache Eviction Hints and Vector Width

```ptx
ld.global.L1::no_allocate.v2.u64 {r0, r1}, [addr];
ld.global.L1::evict_last.v2.u64 {r0, r1}, [addr];
ld.global.v4.u64 {r0, r1, r2, r3}, [addr];
```

`L1::no_allocate` and `L1::evict_last` are eviction-priority hints and may not always be respected. They do not guarantee L1 bypass or residency. The vector forms above move 16 and 32 bytes; whether a width or hint helps depends on alignment, surrounding instructions, reuse, cache state, and the target GPU.

## Cluster Launch Control

CLC cancellation is an asynchronous request followed by response decoding:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    [response_smem], [response_mbarrier];

clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response_b128;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, unused}, response_b128;
```

The request writes an opaque 16-byte response to shared memory and completes on the mbarrier. Code must wait for that phase before loading and querying the response. A successful query returns the first CTA coordinate of a canceled not-yet-launched block or cluster; the request does not take a desired tile ID.

## Tensor Memory Accelerator

A representative 2D global-to-cluster-shared load is:

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [dst_smem], [tensor_map, {x, y}], [full_mbarrier];
```

Cluster multicast adds the full `.multicast::cluster` qualifier and a 16-bit CTA mask:

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
    [dst_smem], [tensor_map, {x, y}], [full_mbarrier], cta_mask;
```

The multicast data and completion signal target the selected CTAs at corresponding shared-memory offsets. Each destination must have a live, initialized matching barrier and wait for its own phase before consuming the tile.

## Related

- [CUDA C++](cuda-cpp.md) — inline PTX integration
- [tcgen05 MMA](../hardware/tcgen05-mma.md) — operand, descriptor, completion, and shape details
- [Tensor Memory](../hardware/tmem.md) — allocation, layout, and transfer details
- [Cluster Launch Control](../hardware/clc.md) — complete request/decode protocol
- [TMA](../hardware/tma.md) — tensor-map and pipeline semantics
- [NVFP4](../hardware/nvfp4.md) — packed format and block scaling

## Primary Reference

- [CUDA 13.0.2, PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html)
