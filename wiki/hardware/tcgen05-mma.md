---
id: hw-tcgen05-mma
title: "tcgen05.mma — Blackwell MMA Instruction"
type: hardware
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, mbarrier]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tmem, hw-2sm-cooperative, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-colfax-cutlass]
aliases: [UMMA, tcgen05, "tensor core gen 05"]
---

# tcgen05.mma -- Blackwell MMA Instruction

## Overview

`tcgen05.mma` is the fifth-generation Tensor Core MMA instruction family used by Blackwell SM100. NVIDIA's CUTLASS documentation also uses **UMMA** for the corresponding higher-level abstraction. Unlike Hopper `wgmma`, an MMA is issued by one thread and writes its accumulator to Tensor Memory (TMEM).

| Property | Hopper `wgmma` | Blackwell `tcgen05.mma` |
|---|---|---|
| Issuing scope | Warpgroup | One issuing thread |
| Accumulator | Registers | TMEM |
| Operand A | Registers or shared-memory descriptor | Shared-memory descriptor or TMEM address, depending on form |
| Operand B | Shared-memory descriptor | Shared-memory descriptor |
| Completion | `commit_group` / `wait_group` | `tcgen05.commit` plus mbarrier wait |

`tcgen05.mma` is asynchronous. Setting `enable-input-d` false computes `D = A*B`; setting it true computes `D = A*B+D`.

## Kinds and CTA groups

The current PTX ISA groups the dense forms as follows:

| Form | Kinds |
|---|---|
| No block scaling | `kind::f16`, `kind::tf32`, `kind::f8f6f4` |
| Block scaling | `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4` |
| Integer | `kind::i8` |

Both `cta_group::1` and cooperative `cta_group::2` forms are documented. The legal M/N/K ranges depend on kind, CTA group, instruction descriptor, operand major modes, and swizzle. Values such as m128n256k16 are common CUTLASS atoms, not a complete statement of every legal shape.

## Issuance and completion

The following PTX shows the essential sequencing. It is schematic—the descriptors, barrier initialization, parity loop, and TMEM allocation must already be correct for the selected MMA shape.

```ptx
// A and B are shared-memory descriptors; D is a TMEM address.
// p = 0 overwrites D with A*B; p = 1 accumulates into D.
// A cta_group::1 disable-output-lane tuple has four .b32 members; all-zero
// masks leave every output lane enabled.
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc, {mask0, mask1, mask2, mask3}, p;

// Track completion of prior cta_group::1 tcgen05 operations from this thread.
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];

// Loop on mbarrier.try_wait.parity until complete, then order the load
// after that synchronization.
tcgen05.fence::after_thread_sync;
tcgen05.ld.sync.aligned.32x32b.x2.b32 {r0, r1}, [d_tmem];
tcgen05.wait::ld.sync.aligned;
```

The fence is an ordering primitive, not a completion wait. In particular, `tcgen05.fence::before_thread_sync` followed by `bar.sync` does **not** establish MMA completion by itself. Use the PTX-documented `tcgen05.commit`/mbarrier mechanism, then `after_thread_sync` before a dependent `tcgen05` operation. `tcgen05.ld` is itself asynchronous and must be followed by `tcgen05.wait::ld` before its destination registers are consumed.

## Block-scaled form

The microscaling kind is `kind::mxf8f6f4`, not `kind::mxf8`. Block-scaled syntax carries separate TMEM addresses for the A and B scale-factor matrices:

```ptx
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
    [d_tmem], a_desc, b_desc, idesc,
    [scale_a_tmem], [scale_b_tmem],
    enable_input_d;
```

This `scale_vec::1X` example requires an architecture-specific target such as
`sm_100a`. The PTX 8.8 `block16`/`block32` aliases require the family-specific
`sm_100f` or `sm_110f` feature set in the PTX 9.3 Target ISA Notes. That is a
forward-portability distinction, not a claim that the aliases cannot assemble
for the corresponding same-generation `a` target: CUTLASS v4.5 emits
`.block16`/`.block32` when its SM100A path is enabled under CUDA 12.9 or later.
An `f` feature may also be used by a later target in the same architecture
family (for example, an `sm_103a` target can use an `sm_100f` family feature).
The aliases are kind-specific: `.block16` aliases `.scale_vec::4X` for
`kind::mxf4nvf4` at K=64 or K=128, while `.block32` aliases
`.scale_vec::1X` for `kind::mxf8f6f4` at every supported K and
`.scale_vec::2X` for `kind::mxf4` or `kind::mxf4nvf4` at K=64 or K=128.
The block-scaled syntax does not carry the dense form's `disable-output-lane`
tuple. The precise scale-vector mode and scale-factor layouts are kind/shape
dependent; see PTX ISA §9.7.17.10.7 rather than treating a single “scale
descriptor” argument as universal.

## Shared-memory layouts

`tcgen05.mma` does **not** universally require 128-byte swizzling. PTX ISA table 57 permits all swizzling modes for the common row-major A and column-major B cases. Transposed operand cases have additional restrictions. A layout that is legal for one type/major-mode pair may be illegal for another, so descriptors should be built using the ISA tables or a checked CUTLASS/CuTe layout rather than hard-coded guessed bit fields.

Swizzling can still be a large performance optimization. That is distinct from correctness: the matrix descriptor encodes no-swizzle, 32-byte, 64-byte, and two 128-byte modes (one with 32-byte atomicity) in documented combinations. The 96-byte TMA tensor-map mode is not a `tcgen05` matrix-descriptor mode.

## Programming implications

- TMEM removes the large register-resident accumulator fragments used by `wgmma`, leaving a different register budget for the rest of the kernel. It does not automatically guarantee higher occupancy; shared memory, threads, barriers, and TMEM allocation also constrain residency.
- One thread issues an MMA, but a practical kernel commonly dedicates a warp to the MMA-control role and other warps to TMA, scheduling, and epilogue work.
- Warp counts and roles are kernel choices. CUTLASS SM100 kernels do not define a universal 16-warp “1 load + 1 MMA + 14 epilogue” mapping.

## Tutorial-reported performance progression

The community `tcgen05 for dummies` tutorial reports this single BF16 GEMM optimization progression on B200 for `M=N=K=4096`, using PyTorch 2.9.1 with CUDA 13 and a 1,506.74-TFLOP/s cuBLAS reference. These figures are source-reported, workload-specific measurements—not architectural throughput guarantees or a general comparison for all shapes. The percentages below are locally derived ratios to that reference, not all percentages printed by the article:

| Tutorial stage | Reported TFLOP/s | Derived fraction of its cuBLAS reference |
|---|---:|---:|
| Basic, 2D 16-byte TMA (v1a) | 254.62 | 16.90% |
| Basic, 3D 16-byte TMA (v1b) | 252.81 | 16.78% |
| 2D 128-byte TMA (v2a) | 681.20 | 45.21% |
| 3D 128-byte TMA (v2b) | 695.43 | 46.15% |
| Pipelining (v3) | 939.61 | 62.36% |
| Warp specialization (v4) | 1,208.83 | 80.23% |
| 2-SM MMA (v5) | 1,302.29 | 86.43% |
| Persistent static scheduling (v6) | 1,475.93 | 97.96% |
| cuBLAS reference | 1,506.74 | 100% |

## Authoritative references

- [`doc-ptx-isa-sm100`](../../sources/docs/nvidia-ptx-isa-sm100.md) — current PTX syntax, operand forms, legal layouts, and synchronization
- [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) — supported SM100 MMA families
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html) — CuTe DSL construction and pipeline concepts
