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
  - source_id: blog-tcgen05-tutorial
    evidence_type: source-reported
related: [hw-tmem, hw-2sm-cooperative, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, blog-tcgen05-tutorial]
aliases: [UMMA, tcgen05, "tensor core gen 05"]
---

# tcgen05.mma -- Blackwell MMA Instruction

## Overview

`tcgen05.mma` is NVIDIA PTX's fifth-generation TensorCore matrix-multiply-accumulate family. PTX ISA 9.0 supports its architecture-specific forms on `sm_100a`; CUTLASS uses the `UMMA` namespace for its SM100 wrappers. NVIDIA's pinned PTX and CUTLASS sources do not define an expanded form of the `UMMA` acronym.

The programming-model shift from Hopper WGMMA is precise:

| Property | WGMMA | tcgen05.mma |
|---|---|---|
| Issue granularity | Warpgroup | One thread for `cta_group::1` or `cta_group::2` |
| D accumulator | Per-thread registers | Tensor Memory (TMEM) |
| A location | Register or SMEM forms, depending on instruction | SMEM descriptor or TMEM address |
| B location | SMEM descriptor | SMEM descriptor |
| Completion | WGMMA commit/wait groups | `tcgen05.commit` plus an mbarrier wait |

Single-thread issue reduces the number of issuing threads, but it does not remove asynchronous completion, operand-lifetime, or inter-thread ordering requirements.

## Kinds and shapes

PTX ISA 9.0 divides dense `tcgen05.mma` into these grammar groups:

| Grammar group | Kind qualifiers |
|---|---|
| Floating point, without block scaling | `f16`, `tf32`, `f8f6f4` |
| Floating point, with block scaling | `mxf8f6f4`, `mxf4`, `mxf4nvf4` |
| Integer | `i8` |

Block-scaled forms add `.block_scale` and accept separate TMEM addresses for A and B scale factors. `f8f6f4` is not the block-scaled MX kind, and `mxf8` is not a dense kind token in this grammar.

M and N are encoded in the instruction descriptor (`idesc`) and are constrained by kind, CTA group, layouts, and target ISA. Consequently, names such as m128n256k16 and m256n256k16 are useful maximum-shape examples for F16/BF16 configurations, not the only legal M and N values. `cta_group::1` uses current-CTA resources; `cta_group::2` can also access the paired peer CTA's SMEM and TMEM resources. All `tcgen05` instructions in a kernel must use the same CTA-group value.

## Operand grammar

The unscaled floating-point grammar includes a disable-output-lane vector before the accumulation predicate:

```ptx
tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc,
                           { disable-output-lane }, enable-input-d
                           {, scale-input-d};
```

For `cta_group::1`, the lane-disable vector contains four 32-bit values; for `cta_group::2`, it contains eight. Setting `enable-input-d` false computes `D = A*B`; setting it true computes `D = A*B+D`. The optional `scale-input-d` is limited to the forms documented by the ISA.

The block-scaled grammar is structurally different:

```ptx
tcgen05.mma.cta_group.kind.block_scale.scale_vectorsize
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;
```

These are normative grammar excerpts, not complete inline-assembly functions: real code must also build valid descriptors, allocate TMEM, preserve operand lifetimes, and implement completion.

## Completion and inter-thread ordering

`tcgen05.mma` is asynchronous. Two mechanisms serve different purposes:

1. `tcgen05.commit.cta_group::N.mbarrier::arrive::one.b64` makes an mbarrier track completion of prior asynchronous tcgen05 operations issued by the executing thread. Waiting on that mbarrier observes completion.
2. `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` order tcgen05 operations across an execution-ordering handoff and constrain code motion. A fence is not a completion wait.

A cross-thread result handoff therefore needs both the applicable completion protocol and the applicable fence/execution-ordering protocol. Likewise, a pipelined mainloop must not release or overwrite an SMEM stage until the asynchronous MMA has finished consuming it.

## Shared-memory descriptors

The tcgen05 shared-memory descriptor is a 64-bit runtime value. PTX ISA 9.0 assigns fields for the encoded base address, leading dimension, stride dimension, fixed bits, base offset, leading-dimension mode, and a three-bit swizzle mode.

Valid swizzle encodings include no swizzle, 128-byte, 64-byte, and 32-byte layouts (plus a 128-byte/32-byte-atomic mode). Values 3, 5, and 7 are invalid; ordinary 128-byte swizzling uses value 2. A layout must satisfy the addressing and alignment constraints for its chosen mode. Although 128-byte swizzling can materially improve a particular GEMM, it is not a universal correctness requirement.

## Register pressure and specialization

For comparison, CUTLASS's pinned Hopper wrapper for m64n256k16 with FP32 accumulation declares 128 accumulator registers per participating thread. tcgen05 keeps D in TMEM during the MMA sequence, which can free GPR capacity for data movement and epilogue work. CUTLASS examples use this model in warp-specialized pipelines with distinct TMA, MMA-control, and accumulator-consumer roles; the best partition remains kernel-dependent.

## Source-reported performance progression

Gau Nernst reports the following results for M=N=K=4096 on a Modal B200, using PyTorch 2.9.1 with CUDA 13. Values are measurements for that setup, not architecture-wide guarantees.

| Tutorial version | Reported TFLOP/s | Approx. cuBLAS share |
|---|---:|---:|
| cuBLAS | 1506.74 | 100% |
| v1a: basic tcgen05 + 2D 16B TMA | 254.62 | 17% |
| v1b: 3D 16B TMA | 252.81 | 17% |
| v2a: 2D 128B TMA | 681.20 | 45% |
| v2b: 3D 128B TMA | 695.43 | 46% |
| v3: pipelining | 939.61 | 62% |
| v4: warp specialization | 1208.83 | 80% |
| v5: 2-SM MMA | 1302.29 | 86% |
| v6: persistent, static scheduling | 1475.93 | 98% |

The v6 result did not use Cluster Launch Control; the author lists CLC and threadblock swizzling as unimplemented follow-up ideas.
