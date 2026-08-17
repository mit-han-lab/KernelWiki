---
id: doc-ptx-isa-sm100
title: "PTX ISA SM100 Instructions Reference"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [ptx, tcgen05, tmem, clc, tma, nvfp4, fp4, fp8, fp6, block-scale, mbarrier]
retrieved_at: 2026-08-16
---

# PTX ISA SM100 Instructions Reference

## Scope and version

This page records the SM100 instruction families in the current PTX ISA 9.3 documentation. The core `tcgen05` allocation, load/store, MMA, fence, commit, and Cluster Launch Control instructions were introduced in PTX ISA 8.6; later PTX revisions add qualifiers, shapes, and family-specific targets. Exact target notes belong to each instruction in the official ISA and should be checked when emitting PTX for a particular target.

## `tcgen05.mma`

`tcgen05.mma` is asynchronous and is issued by one thread. Its dense, non-block-scaled floating-point form is:

```ptx
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc,
    {disable0, disable1, disable2, disable3}, enable_input_d;
```

The destination is in TMEM. Operand B is a shared-memory descriptor; operand A may be either a shared-memory descriptor or a TMEM address, depending on the instruction form. The `disable-output-lane` vector has four `.b32` members for `cta_group::1` and eight for `cta_group::2`; an all-zero vector disables no output lanes. `enable_input_d` selects `D = A*B+D` when true and `D = A*B` when false.

The documented kind groups are:

| Form | Kinds |
|---|---|
| Floating point, no block scaling | `kind::f16`, `kind::tf32`, `kind::f8f6f4` |
| Floating point, block scaling | `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4` |
| Integer | `kind::i8` |

Block-scaled forms add `.block_scale` (and an optional scale-vector-size qualifier) plus TMEM addresses for the A and B scale-factor matrices. Both `cta_group::1` and `cta_group::2` forms exist; the legal M/N/K shapes depend on the kind, CTA group, layout, and instruction descriptor, so this summary intentionally does not collapse them into one fixed shape.

PTX ISA 9.3's `tcgen05.mma` Target ISA Notes distinguish the qualifier targets: `.scale_vec::1X`, `.scale_vec::2X`, and `.scale_vec::4X` require `sm_100a`, while `.block16` and `.block32` require `sm_100f` or `sm_110f`. Section 11.1.2 separately defines an `f` target as a family-specific feature target and an `a` target as architecture-specific; a family feature may be used by a later `a` target in that same family. These are target-feature rules, not a claim that a same-generation toolchain cannot accept a block alias under its corresponding architecture-specific path.

The block aliases are kind- and K-dependent rather than interchangeable shorthand:

| Alias | Equivalent scale-vector qualifier | Applicable kind and K |
|---|---|---|
| `.block16` | `.scale_vec::4X` | `kind::mxf4nvf4` with K=64 or K=128 |
| `.block32` | `.scale_vec::1X` | `kind::mxf8f6f4` for every supported K |
| `.block32` | `.scale_vec::2X` | `kind::mxf4` or `kind::mxf4nvf4` with K=64 or K=128 |

## TMEM allocation and register transfers

Allocation counts TMEM **columns**, not rows, and writes the allocated address to shared memory:

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_tmem_addr], n_cols;
ld.shared.b32 taddr, [smem_tmem_addr];

// ... use taddr ...

tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, n_cols;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

All dynamically allocated TMEM must be deallocated before kernel exit. Loads and stores between TMEM and registers are asynchronous collective warp operations:

```ptx
tcgen05.ld.sync.aligned.32x32b.x2.b32 {r0, r1}, [taddr];
tcgen05.wait::ld.sync.aligned;

tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], {r0, r1};
tcgen05.wait::st.sync.aligned;
```

All threads in the executing warp participate and must use the same base address.

## MMA completion and cross-thread ordering

`tcgen05.fence` is an ordering primitive, not by itself a completion wait. The mbarrier completion path for an MMA is:

```ptx
// Issuing thread
tcgen05.mma.cta_group::1.kind::f16
    [taddr], a_desc, b_desc, idesc, {mask0, mask1, mask2, mask3}, p;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar];

// Wait until mbarrier.try_wait.parity reports completion, then order
// this thread's following tcgen05 operation after the synchronization.
tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.32x32b.x2.b32 {r0, r1}, [taddr];
tcgen05.wait::ld.sync.aligned;
```

`tcgen05.commit` tracks prior asynchronous MMA/copy/shift operations issued by the same thread. `before_thread_sync` prevents preceding `tcgen05` operations from moving after a following execution-ordering operation; `after_thread_sync` prevents subsequent `tcgen05` operations from moving before a preceding synchronization. A bare `before_thread_sync` plus `bar.sync` is therefore not a substitute for tracking and waiting for MMA completion.

## Shared-memory layouts

There is no blanket “128-byte swizzle required” rule for `tcgen05.mma`. For the common row-major A / column-major B cases, PTX ISA table 57 permits all swizzling modes. Transposed cases have narrower legal combinations. The matrix descriptor's swizzle field represents no swizzle, 32-byte, 64-byte, and two 128-byte modes; 96-byte swizzle belongs to TMA tensor-map encoding, not this descriptor. Descriptor construction, alignment, leading/stride offsets, and swizzle must be derived from the chosen kind, major mode, and shape; hand-written descriptor bit fields should be checked against the current ISA tables.

## Cluster Launch Control

Cluster Launch Control does not use `clc.arrive` / `clc.wait` instructions. A cancellation request asynchronously tries to cancel a cluster that has not launched, writes a 16-byte opaque response to shared memory, and uses an mbarrier completion mechanism:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    [response], [mbar];

// After waiting for the mbarrier and loading the 16-byte response:
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, _}, handle;
```

The returned CTA id is valid only after a successful cancellation. CLC enables work-stealing designs, but it does not guarantee that all tail effects or load imbalance disappear.

## Sub-byte conversion examples

These examples use instruction forms shown in the PTX ISA conversion section:

```ptx
// Two f32 values -> packed FP8 E4M3 values.
cvt.rn.satfinite.e4m3x2.f32 d, a, b;

// Packed FP8 E4M3 -> packed f16x2.
cvt.rn.f16x2.e4m3x2 d, a;

// Packed FP4 E2M1 -> packed f16x2.
cvt.rn.relu.f16x2.e2m1x2 d, a;

// Packed f16x2 -> packed FP4 E2M1; saturation is required.
cvt.rn.satfinite.e2m1x2.f16x2 d, a;
```

## Sources

- [PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/), especially §§9.7.14.18–19, 9.7.17.6–12, and 11.1.2
- [CUDA Cluster Launch Control guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
