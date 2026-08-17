---
id: hw-tma
title: "Tensor Memory Accelerator (TMA)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [tma, mbarrier]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-flashinfer-2387, evidence_type: upstream-code}
related: [hw-tcgen05-mma, technique-pipeline-stages, technique-swizzling]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, pr-flashinfer-2387]
aliases: [TMA, "tensor memory accelerator", "cp.async.bulk"]
blackwell_relevance: "TMA provides asynchronous tensor copies and cluster multicast used by many SM100 pipelines; its swizzle is a descriptor choice constrained by the consumer layout, not universally 128 bytes."
---

# Tensor Memory Accelerator (TMA)

## What TMA does

TMA is the tensor form of CUDA's asynchronous bulk-copy machinery. A thread issues a copy described by a tensor map; hardware performs the multidimensional address generation and data movement. Global-to-shared tensor loads complete through an `mbarrier`. Shared-to-global stores use bulk async-group commit and wait operations. Cluster-scoped loads can multicast a tile to selected CTAs in a cluster.

The tensor map describes the global tensor, box dimensions, element strides, interleave, shared-memory swizzle, L2-promotion hint, and out-of-bounds behavior. The CUDA Driver API documents ranks 1 through 5 and validates descriptor constraints when the map is encoded. Tile-size limits are therefore descriptor- and datatype-specific; there is no general "128x256 tile" limit or in-transfer FP32-to-FP16 conversion guarantee.

## Swizzle is a layout choice

The driver exposes no swizzle and 32-, 64-, and several 128-byte swizzle modes. These rearrange chunks within a span to reduce shared-memory bank conflicts. Each mode imposes alignment and bounding-box constraints. A TMA producer and its consumer must agree on the actual shared-memory layout.

`tcgen05.mma` does **not** impose a universal 128-byte-swizzle rule. Its matrix descriptor encodes a documented leading-dimension/stride layout. Depending on matrix orientation and datatype, the PTX ISA permits no swizzle or one of multiple swizzle modes. A wrong descriptor/layout pairing can produce wrong values, but an unswizzled operand is not intrinsically invalid.

## Synchronization pattern

This PTX-shaped sketch shows the ordering relationships; exact opcodes, scopes, addresses, and phase handling must match the selected PTX ISA version:

```ptx
// Initialize an mbarrier before use.
mbarrier.init.shared::cta.b64 [bar], 1;

// The issuing thread declares the expected bytes and starts the tensor load.
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], bytes;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [dst_smem], [tensor_map, {x, y}], [bar];

// Consumers wait for the appropriate phase before reading dst_smem.
mbarrier.try_wait.parity.shared::cta.b64 ready, [bar], phase;
```

The barrier must be initialized and made visible before the copy. Reusing a pipeline stage requires correct phase tracking and a consumer-release protocol; a single wait without stage ownership is not a complete reusable pipeline.

For a shared-to-global tensor store, the producer must make prior shared-memory writes visible to the async proxy, issue the store, commit the bulk group, and wait before reusing the source buffer when required by the algorithm.

## Multicast

A cluster multicast performs one global-memory transaction stream and deposits the result in the shared memory of CTAs selected by a multicast mask. This can reduce redundant traffic when cluster peers need the same operand, but the reduction is workload- and cache-dependent. Every receiving CTA needs a compatible destination and completion barrier.

## Practical rules

- Build tensor maps with the current CUDA Driver API and check every returned error.
- Choose swizzle from the consumer's documented layout, then satisfy its alignment and box constraints.
- Treat the tensor map as immutable while an operation using it may be in flight unless the documented proxy-fence protocol is followed.
- Verify `mbarrier` phase ownership, expected-byte counts, and stage reuse with boundary shapes.
- Measure multicast and pipeline-depth choices; they trade bandwidth against shared memory, barrier traffic, and occupancy.

## Evidence basis

The descriptor fields and swizzle constraints come from the CUDA Driver API tensor-memory documentation. Instruction forms and completion rules come from the PTX ISA. CUTLASS/CuTe code is useful as a concrete implementation, but its selected swizzle and stage count are kernel configurations rather than architecture-wide requirements.
