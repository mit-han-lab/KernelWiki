---
id: technique-pipeline-stages
title: "Software Pipelining and Multi-Stage Buffering"
type: technique
architectures: [sm100, sm90]
tags: [pipeline-stages, double-buffering, tma, mbarrier]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
reproducibility: pseudocode
prerequisites: [hw-tma, hw-tmem]
related: [technique-warp-specialization, technique-double-buffering, hw-tma]
sources: [blog-tcgen05-tutorial, blog-modular-blackwell, doc-nvidia-tuning-guide]
blackwell_relevance: "SM100 pipelines TMA-produced SMEM stages into tcgen05 consumers; correctness depends on transaction, MMA-completion, phase, and reuse edges."
---

# Software Pipelining and Multi-Stage Buffering

## What a stage means

A multi-stage pipeline reserves disjoint shared-memory operand buffers so production of one tile can overlap consumption of another. Stage count controls how far producer and consumer progress may separate; it does not guarantee that latency is fully hidden or that the schedule is faster.

Each stage cycles through this ownership state machine:

| State | Owner | Transition condition |
|---|---|---|
| empty | producer | the prior consumer has completed every read of the stage |
| loading | TMA async proxy | expected transaction bytes are registered before the copy and the TMA transaction is outstanding |
| ready | consumer | the full barrier's matching phase has completed |
| consuming | MMA path | the consumer has acquired the stage and no producer may overwrite it |
| empty again | producer | every asynchronous MMA use of the SMEM operands is complete and the consumer releases the matching empty phase |

Modulo indexing selects storage, but it does not track ownership by itself. A correct implementation also supplies barrier initialization/publication, expected arrival counts, phase or token state, async-proxy ordering, a bounded prologue, steady-state advancement, pipeline tail, and error-free behavior when K tiles are fewer than stages.

On SM100, do not release a stage merely after issuing `tcgen05.mma`. The MMA is asynchronous; use its defined completion path before allowing a producer to overwrite A/B storage. Likewise, `__syncthreads()` is not a substitute for the required tcgen05/TMA async-proxy completion and ordering operations.

## TMA/full-barrier edge

For a TMA load into shared memory, the producer initializes/publishes the mbarrier, accounts for the copy's expected transaction bytes, and issues the tensor copy with that barrier. Hardware completes the transaction on the barrier; the consumer waits for the matching phase before reading the stage.

This autonomous completion avoids a producer instruction that manually announces data-ready after the transfer. It does not prove that producer issue work or the overall load path is absent from the measured critical path.

Use the exact PTX ISA tensor-copy grammar or a version-pinned library pipeline. An illustrative `arrive`/spin loop that omits expected transaction bytes, copy descriptors, proxy fences, phase initialization, and the consumer-to-producer reuse edge is not a safe pipeline.

## Stage capacity arithmetic

For binary16 `A[128,64]` and `B[64,256]`, unpadded operand payload is:

```text
A = 128 × 64 × 2 bytes = 16 KiB
B =  64 × 256 × 2 bytes = 32 KiB
one stage = 48 KiB
three stages = 144 KiB
five stages = 240 KiB
```

Compute capability 10.0 supports up to 228 KiB of shared memory per SM. Thus five stages of that exact payload already exceed the per-SM capacity, while three stages consume about 63.2% before barriers, padding, descriptors, epilogue buffers, and other shared storage. Per-block opt-in limits and occupancy constraints also apply.

Payload grows linearly with stage count. Total allocation can additionally contain fixed and stage-dependent metadata, so derive the concrete shared-storage type rather than multiplying payload alone.

## Select a stage count

There is no architecture-wide table in which two stages are always partial, three always fully hide latency, or more than five are always excessive. Legal and useful depth depends on:

- bytes per stage, alignment/swizzle padding, other SMEM, and occupancy;
- TMA issue/transfer rate and descriptor/transaction shape;
- MMA work and completion time per K tile;
- producer/consumer role schedule and register pressure;
- K-loop length, prologue/tail fraction, and output schedule.

Compile every candidate, reject resource-invalid variants, then compare controlled timings and pipeline/barrier stalls. Include `num_k_tiles` values below, equal to, and above the stage count to exercise every boundary.

## Evidence-scoped examples

Gau Nernst's disclosed B200 4096-cubed experiment reports 695.43 TFLOP/s for v2b and 939.61 TFLOP/s for v3 after adding pipelining, about a 35.1% increase for that source progression. Warp specialization is the later v4 result at 1208.83 TFLOP/s; the 1475.93-TFLOP/s v6 endpoint also includes 2-SM MMA and static persistence.

Modular Part 3 uses five A/B stages in its particular 2-SM matmul. The eventual author-reported 85%-of-SOTA endpoint also includes 2-SM work, warp specialization, and double-buffered output writeback; it is not a five-stage-only result. The article does not supply the fixed “~400 cycle HBM latency” premise previously attributed to it.

## Validation checklist

1. Verify every stage has independent storage and full/empty synchronization state.
2. Initialize correct expected participants and publish barrier state before async use.
3. Register expected TMA transaction bytes before issuing the copy.
4. Wait for TMA completion before the first dependent read.
5. Wait for every asynchronous MMA read to complete before releasing the stage.
6. Track the correct phase/token on first use and every wraparound.
7. Drain producer and consumer tails, including output-store tails where present.
8. Fault-test wrong phase, early release, omitted tail, and short-K paths under a bounded watchdog.

## Primary references

- [PTX ISA 9.0 mbarrier](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- [PTX ISA 9.0 bulk tensor copy](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [PTX ISA 9.0 tcgen05 completion](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen-async-sync-operations-commit)
- [CUTLASS 4.5.0 CuTe DSL pipeline example](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_2.py)
- [Pinned tcgen05 tutorial source](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v3.cu)
- [Modular Part 3](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)

## Related

- [TMA](../hardware/tma.md) — tensor-copy descriptors and completion
- [double buffering](double-buffering.md) — shared/TMEM ownership patterns
- [warp specialization](warp-specialization.md) — optional dedicated producer/consumer roles
