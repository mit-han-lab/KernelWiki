---
id: technique-pipeline-stages
title: "Software Pipelining and Multi-Stage Buffering"
type: technique
architectures: [sm100, sm90]
tags: [pipeline-stages, double-buffering, tma, mbarrier]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tma, hw-tmem]
related: [technique-warp-specialization, technique-double-buffering, hw-tma]
sources: [blog-tcgen05-tutorial, blog-modular-blackwell, doc-nvidia-tuning-guide, pr-flashinfer-1039]
blackwell_relevance: TMA and barriers support producer/consumer overlap; the correct stage count depends on the kernel's tile resources and timings.
---

# Software Pipelining and Multi-Stage Buffering

Software pipelining keeps multiple shared-memory tile buffers in flight so a producer can load a later tile while a consumer performs MMA on an earlier tile. Each buffer stage needs an ownership protocol: the consumer waits for the transfer-complete state, and the producer waits until the consumer releases the stage before reuse.

## Source-reported tutorial step

In the single-B200 GEMM tutorial by Thien Tran (`gau-nernst`), the no-pipeline v2b row reports 695.43 TFLOPS and the pipelined v3 row reports 939.61 TFLOPS, about a 35% increase for that exact benchmark. Later rows add warp specialization, two-SM MMA, and persistent static scheduling, so their gains are not attributed to pipeline staging alone.

## Choosing a stage count

There is no universally optimal count. More stages can expose additional overlap but consume more shared memory and barrier state, can reduce residency, and add prologue/epilogue work. Choose the count from:

- bytes of A/B/scale data per stage;
- available dynamic shared memory after other storage;
- measured load and MMA duration per tile;
- K-loop length and reuse frequency;
- occupancy and register effects of the producer/consumer roles.

The Modular article uses a multi-stage circular buffer as one component of a longer optimization sequence ending at 85% of its stated reference. That final percentage is not a pipeline-only result.

## Correctness checks

Validate phase transitions, expected transaction bytes, and buffer reuse with short and non-multiple K loops as well as the steady state. An off-by-one barrier phase can be a correctness bug or deadlock, and a stage-count change must not be treated as a performance-only edit.

FlashInfer PR 1039 provides a concrete producer/consumer sequence. This
contiguous excerpt from its captured SM100 attention mainloop is illustrative,
not a standalone kernel:

```cpp
pipeline_q.consumer_wait(pipeline_q_consumer_state);
++pipeline_q_consumer_state;
pipeline_s0.producer_acquire(pipeline_s0_producer_state);
gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);
pipeline_s0.producer_commit(pipeline_s0_producer_state);
++pipeline_s0_producer_state;
```
