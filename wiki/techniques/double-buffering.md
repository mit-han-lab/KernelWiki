---
id: technique-double-buffering
title: "Double/Multi-Buffering Patterns"
type: technique
architectures: [sm100, sm90]
tags: [double-buffering, tmem, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem]
related: [hw-tmem, technique-pipeline-stages, technique-epilogue-fusion]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, pr-flashinfer-1039]
blackwell_relevance: "TMEM can hold multiple logical regions while shared-memory stages overlap copy and compute; both allocations must fit the selected kernel resources."
---

# Double and multi-buffering

Buffering uses distinct storage regions for pipeline iterations so one agent can produce a stage while another consumes a different stage. On SM100, a kernel may pipeline shared-memory operand stages and may also partition its allocated TMEM columns among multiple accumulators or intermediates.

The PTX ISA exposes 512 TMEM columns to a CTA, with allocation in power-of-two column counts. It does not guarantee that every accumulator occupies one column per logical output column; the mapping depends on the selected MMA kind and layout. Derive the budget from the actual tiled MMA rather than assuming two equal 256-column halves.

## Source-backed ownership fragment

This excerpt from FlashInfer PR 1039’s captured Blackwell attention mainloop shows a consumer waiting on one pipeline before acquiring and committing the next:

```cpp
pipeline_q.consumer_wait(pipeline_q_consumer_state);
++pipeline_q_consumer_state;
pipeline_s0.producer_acquire(pipeline_s0_producer_state);
gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);
pipeline_s0.producer_commit(pipeline_s0_producer_state);
```

## Correctness and tuning checks

- A stage must not be overwritten until every consumer has released it.
- Producer and consumer phase/parity state must advance exactly once per reuse.
- TMEM allocations must obey column-count and warp-collective lifetime rules.
- Extra stages consume shared memory or TMEM and may reduce residency.
- Select stage count from measured wait time and resource use; there is no universally correct default.
