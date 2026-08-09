---
id: hw-tma
title: "Tensor Memory Accelerator (TMA)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [tma, mbarrier]
confidence: verified
evidence_basis:
  - source_id: doc-cuda-13-0-2-tma
    evidence_type: official-doc
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-mbarrier, hw-tcgen05-mma, technique-pipeline-stages, technique-swizzling]
sources: [doc-cuda-13-0-2-tma, doc-ptx-isa-sm100, pr-cutlass-2139]
aliases: [TMA, "tensor memory accelerator", "cp.async.bulk.tensor"]
blackwell_relevance: "Blackwell retains Hopper's descriptor-driven TMA and adds tensor-map and copy modes; a TMA destination layout must match the exact tcgen05 shared-memory descriptor rather than one universal swizzle."
---

# Tensor Memory Accelerator

## Scope

TMA is the descriptor-driven `cp.async.bulk.tensor` facility introduced for Hopper (`sm_90`) and retained on Blackwell. One thread can issue a non-blocking rank-1 through rank-5 tensor copy while the hardware performs multidimensional address traversal and moves the tile.

The principal copy directions and completion mechanisms are:

| Direction | Destination | Optional cluster behavior | Completion |
|---|---|---|---|
| Global to shared | Issuing CTA or a CTA in its cluster | One masked instruction can multicast to selected CTAs | mbarrier `complete_tx` in bytes |
| Shared to global | Issuing CTA's shared memory to a tensor map | Scatter modes are available on supported Blackwell targets | Bulk async group: issue, commit, wait |

TMA copies the datatype represented by the tensor map. It does not provide a general FP32-to/from-FP16 or BF16 conversion step. For a tiled global-to-shared load, out-of-bounds elements are filled according to the supported tensor-map policy rather than clamped to an edge coordinate.

## Tensor maps

A tensor map is opaque and is accessed through the tensor-map proxy. `cuTensorMapEncodeTiled` describes:

- datatype and global base address;
- rank, global dimensions, and byte strides;
- traversal box dimensions and element strides;
- interleave and shared-memory swizzle;
- L2-promotion hint; and
- out-of-bounds fill policy.

For the ordinary non-interleaved tiled path in CUDA Driver API 13.0.97, important constraints include:

- the `CUtensorMap` output object is 64-byte aligned;
- the global base and byte strides satisfy the documented alignment rules, commonly at least 16 bytes for the basic types/path;
- every `boxDim` entry is from 1 through 256 elements;
- `boxDim[0] * element_size` is a multiple of 16 bytes;
- every element stride is from 1 through 8; and
- when swizzling is enabled, the inner box in bytes does not exceed the selected swizzle span.

Datatype, interleave, sub-byte, and architecture-specific modes add further restrictions. Always check the encoder's `CUresult`; never use the output after an encoding failure.

Host encoding is common, but it is not the only Blackwell path. CUDA also documents device-side tensor-map construction and modification on Blackwell. A map modified through the generic proxy must be published to the tensor-map proxy with the required `fence.proxy.tensormap::generic` sequence before a TMA operation consumes it.

## Global-to-shared load

PTX ISA 9.0 defines this representative 2D CTA-local form:

```ptx
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
    [dst_smem], [tensor_map, {x, y}], [full_barrier];
```

The instruction is non-blocking. A common one-producer phase with one or more loads uses this accounting:

1. Initialize and publish the stage's mbarrier with the intended pending-arrival count.
2. The producer performs one `mbarrier.arrive.expect_tx` for its software arrival and the sum of bytes that all loads in this phase will complete.
3. Issue the TMA loads against that barrier.
4. Each completed load performs `complete_tx` for its copied byte count; it does **not** perform another arrival.
5. Consumers wait with the correct state token or per-stage parity and acquire semantics before reading the destination.

The phase completes only after pending arrivals and tx-count are both zero. If a design performs multiple software arrivals instead, its initialized count must match them exactly. See [mbarrier](mbarrier.md) for lifecycle, phase, and memory-ordering rules.

## Shared-to-global store

A tensor store uses bulk-group completion rather than the load's mbarrier protocol:

```ptx
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
    [tensor_map, {x, y}], [src_smem];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

The wait may be delayed to overlap independent work. It must occur before the issuing thread reuses the source shared memory or before code that requires store completion. `commit_group` creates the group; it is not itself a completion wait.

## Cluster multicast

The cluster form copies a global tile to the same shared-memory offset in every CTA selected by a 16-bit mask:

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [dst_smem], [tensor_map, {x, y}], [full_barrier], cta_mask;
```

For `cta_group::1` (the default), the completion signal is also multicast to the same barrier offset in every selected destination CTA. A correct design therefore:

- launches an explicit cluster and keeps every destination CTA's shared memory alive;
- initializes corresponding destination barriers before the elected issuer can start;
- uses one cluster-wide elected issuer, not one `threadIdx.x == 0` issuer in every CTA;
- includes only valid destination CTA ranks in the mask; and
- has every destination wait on its own corresponding barrier phase before consuming the tile.

For GEMM tiles with the same N range and different M ranges, the CTAs use different A tiles but the same B tile, so multicast can avoid duplicate logical B-load requests. It does not promise an exact `cluster_size` reduction in measured DRAM traffic or elapsed time; caches, mask population, transaction behavior, and resource costs affect the result.

## Swizzle and tcgen05

TMA supports no swizzle and multiple swizzled shared-memory layouts, including 32B, 64B, and 128B spans plus newer variants for selected types. Swizzling rearranges chunks across shared-memory banks. The consumer must address the matching logical layout; the exact mapping also depends on the documented shared-memory base-offset rule.

There is no universal rule that every Blackwell or `tcgen05.mma` input uses 128B swizzling. The TMA tensor-map swizzle, destination base alignment, leading dimension, and the tcgen05 shared-memory descriptor must describe the **same** legal layout. PTX defines no-, 32B-, 64B-, and 128B-swizzled tcgen05 descriptors with kind-, type-, shape-, and layout-specific constraints.

A matched swizzle can reduce or remove bank conflicts for a particular access pattern. It does not make every possible consumer access conflict-free.

## Pipeline invariants

A common Blackwell GEMM data path is:

`GMEM -> TMA -> SMEM -> tcgen05.mma -> TMEM -> tcgen05.ld -> registers -> output store`

Each reusable pipeline stage needs two independent ownership transitions:

- **full:** TMA has finished producing the shared-memory operands, so the MMA consumer may read them; and
- **empty:** asynchronous MMA has stopped reading those operands, so the TMA producer may overwrite the stage.

Track full and empty state per stage. Account transaction bytes once, track each barrier's own phase, and do not equate CTA synchronization or a tcgen05 fence with async completion. A complete CUTLASS pipeline is safer evidence than a shortened helper that omits one ownership edge.

Choose tile rank, swizzle, multicast, issue cadence, and stage count from the actual access pattern and resource budget. More distinct stages consume more shared memory; there is no universal optimum of three to five stages. Profile the target GPU and record the copy shape, descriptor, cluster mask, stage count, shared-memory use, occupancy, warmup, repetitions, and baseline.

## References

- [CUDA 13.0.2 Programming Guide: TMA](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
- [CUDA Driver API 13.0.97: tensor-map management](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [PTX ISA 9.0: `cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [CUTLASS 4.5.0: complete CuTe DSL TMA tutorial](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py)
- [mbarrier](mbarrier.md)
- [tcgen05 MMA](tcgen05-mma.md)
