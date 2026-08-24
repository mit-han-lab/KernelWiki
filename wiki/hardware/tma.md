---
id: hw-tma
title: "Tensor Memory Accelerator (TMA)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [tma, mbarrier]
confidence: source-reported
related: [hw-tcgen05-mma, technique-pipeline-stages, technique-swizzling]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, doc-cutlass-cute-dsl]
aliases: [TMA, "tensor memory accelerator", "cp.async.bulk"]
blackwell_relevance: "Blackwell retains Hopper's descriptor-driven asynchronous tensor copies; descriptors and shared-memory layouts must agree when feeding tcgen05."
---

# Tensor Memory Accelerator (TMA)

TMA is the descriptor-driven asynchronous bulk-copy path introduced with Hopper and retained on Blackwell. A thread issues a tensor copy; hardware performs multidimensional address generation and the transfer without a per-element load/store loop in the issuing thread.

## Programming contract

- A tensor-map descriptor supplies global shape, strides, element format, bounds behavior, and any shared-memory swizzle.
- Global-to-shared tensor copies complete through an mbarrier transaction-count mechanism. Consumers must wait for that completion before reading the destination.
- Shared-to-global copies have their own completion and ordering rules; do not reuse a global-to-shared barrier recipe by analogy.
- Cluster multicast targets shared memory in multiple CTAs and requires a valid cluster configuration and mask.
- Descriptor base-address, stride, box-size, and alignment constraints depend on rank, element type, swizzle, and instruction form.

A representative global-to-shared form is:

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes
    [smem_dst], [tensor_map, {x, y}], [mbarrier_addr];
```

## Swizzling

TMA supports multiple swizzle modes. The shared-memory layout constructed by the program must match the mode encoded in the descriptor. A 128-byte swizzle can be effective for a particular tile, but `tcgen05.mma` does not universally require it: non-swizzled, 32-byte, and 64-byte layouts are also represented by supported descriptor/layout forms when their shape constraints are satisfied.

Use the CUDA Driver API tensor-map documentation and the PTX ISA for exact encoding constraints. CUTLASS and CuTe helpers are preferable when they cover the intended layout because they construct matching tensor maps and shared-memory views together.
