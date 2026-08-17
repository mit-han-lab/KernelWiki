---
id: lang-cuda-cpp
title: "CUDA C++ for Blackwell Kernels"
type: language
tags: [cuda-cpp, ptx, tcgen05, tmem]
related: [lang-ptx, hw-tcgen05-mma, hw-tmem, blog-tcgen05-tutorial]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, pr-cutlass-2139, blog-tcgen05-tutorial, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
---

# CUDA C++ for Blackwell Kernels

## Overview

CUDA C++ supplies the kernel/launch model, while low-level SM100 kernels often reach `tcgen05`, TMEM, CLC, and specialized cache controls through CUTLASS/CuTe wrappers or inline PTX. Inline PTX is version-sensitive: operand types, register tuple sizes, participation, and memory constraints must match the exact ISA form.

Tutorial results near a library baseline demonstrate one tuned kernel/configuration. They are not a performance property of “plain CUDA C++.”

## Prefer typed wrappers when available

```cuda
template <class TiledMma, class Accumulator, class TensorA, class TensorB>
__device__ void issue_mma(TiledMma const& mma,
                          Accumulator& accumulator,
                          TensorA const& a,
                          TensorB const& b) {
  // CUTLASS/CuTe traits carry instruction descriptors, operand source,
  // layouts, and TMEM mapping for the selected target.
  cute::gemm(mma, a, b, accumulator);
}
```

This is a structural example; use the API signature from the pinned CUTLASS version. A hand-written `tcgen05.mma` wrapper that invents separate C/D descriptors or assumes one result register is unsafe.

## Inline-PTX checklist

- Compile for the feature-specific SM100 target required by the instruction.
- Declare every operand with the PTX-prescribed register width and constraint.
- Include `volatile` and a `memory` clobber when compiler-visible memory ordering requires them.
- Follow single-thread/CTA-pair issue and uniformity rules.
- Publish a TMEM allocation address through shared memory exactly as specified.
- Derive TMEM load/store register tuples from the selected fragment shape.
- Commit and wait for asynchronous MMA completion; do not use a fence as a wait.
- Match allocation/deallocation base and column count and finish outstanding users first.

## Barrier state

Wrap pipeline state as a stage token rather than an unscoped spin loop:

```cuda
struct StageState {
  uint64_t* full_barrier;
  uint64_t* empty_barrier;
  unsigned producer_phase;
  unsigned consumer_phase;
  uint32_t expected_bytes;
};
```

The producer's `arrive.expect_tx` and TMA completion must satisfy the initialized arrival/transaction counts. Each circular stage tracks its own phase. Cross-thread tcgen ordering and async-proxy ordering are separate from completion observation.

## Role dispatch

Warp specialization is ordinary control flow plus a rigorously defined dependency graph. Role indices are configuration-specific; do not assume warp 0 is TMA, warp 1 MMA, and every later warp epilogue. Current CUTLASS SM100 schedules include distinct scheduler, mainloop-load, MMA, epilogue-load, and variable epilogue roles.

Use [`ptx-sm100`](ptx-sm100.md) for checked instruction fragments and the shipped upstream CUTLASS artifacts for complete context.
