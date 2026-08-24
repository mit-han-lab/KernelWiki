---
id: doc-ptx-isa-sm100
title: "PTX ISA Fifth-Generation Tensor Core and CLC Reference"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
source_category: official-doc
architectures: [sm100, sm100a, sm103, sm110, sm120, sm121]
tags: [ptx, tcgen05, tmem, clc, tma, nvfp4, fp4, fp8, fp6, block-scale, mbarrier]
retrieved_at: 2026-08-18
---

# PTX ISA reference scope

This page is a source map to NVIDIA’s rolling PTX ISA, accessed on 2026-08-18. The online document reported PTX ISA 9.3 on that date. Architecture and PTX-version requirements must be read from each instruction’s own target notes; the summaries below are not substitutes for those tables.

## Tensor Memory

For `sm_100a`/`sm_100f`, PTX describes a CTA-visible two-dimensional TMEM structure with 512 columns, 128 lanes, and 32-bit cells. Allocation is column-based: the unit is 32 columns, the count is a power of two from 32 through 512, and all 128 lanes of an allocated column are included.

`tcgen05.alloc`, `dealloc`, and `relinquish_alloc_permit` have warp issue granularity for `cta_group::1`; their `.sync.aligned` rules require every thread in the issuing warp to participate with matching operands. MMA, copy, shift, and commit operations have single-thread issue granularity.

Exact locators:

- §9.7.17.1, “Tensor Memory”
- §9.7.17.5, “Issue Granularity”
- §9.7.17.7, “Tensor Memory Allocation and Management Instructions”

## tcgen05 MMA

The A matrix may reside in TMEM or shared memory, B in shared memory, and D in TMEM. The required 32-bit instruction descriptor records shapes, exact types, sparsity, and operation details.

Unscaled kinds are `f16`, `tf32`, `f8f6f4`, and `i8`. Block-scaled syntax uses `mxf8f6f4`, `mxf4`, and `mxf4nvf4` with explicit scale-factor operands. The document does not define `kind::mxf8`.

Exact locators:

- §9.7.17.4.2, “Instruction descriptor”
- §9.7.17.10, “TensorCore 5th Generation Matrix Multiply and accumulate Operations”
- §9.7.17.10.7, “Block Scaling for tcgen05.mma”

## Cluster Launch Control

`clusterlaunchcontrol.try_cancel` requests cancellation of a cluster in the same grid that has not launched. It writes an opaque 16-byte response asynchronously to shared memory and signals completion through an mbarrier. `clusterlaunchcontrol.query_cancel` tests success and, on success, extracts the first CTA ID of the canceled cluster.

PTX ISA 9.3 lists the instruction as requiring `sm_100` or higher. The optional
`.multicast::cluster::all` qualifier has a narrower architecture list in its
target notes and must not be inferred from that baseline requirement.

Exact locators:

- §9.7.14.18, `clusterlaunchcontrol.try_cancel`
- §9.7.14.19, `clusterlaunchcontrol.query_cancel`

## TMA and mbarrier

The `cp.async.bulk.tensor` family performs descriptor-driven asynchronous tensor copies. Global-to-shared forms use mbarrier transaction completion. PTX 7.0 introduced mbarrier for `sm_80`; later versions added transaction-count operations and additional scopes used by TMA and cluster operations.

Exact locators:

- §9.7.9.26.5.2, `cp.async.bulk.tensor`
- §9.7.14, parallel synchronization and communication instructions
- PTX ISA release notes for versions 7.0 and 8.x
