---
id: pattern-pipeline-stalls
title: "Pipeline Stalls"
type: pattern
tags: [pipeline-stages, warp-specialization, tma, tcgen05, mbarrier]
symptoms: [pipeline-stalls, compute-bound, low-tensor-core-utilization]
candidate_techniques: [technique-pipeline-stages, technique-warp-specialization, technique-double-buffering, technique-ping-pong-scheduling]
related: [pattern-compute-bound, pattern-tail-effect]
sources: [blog-tcgen05-tutorial, blog-flash-attention-4, doc-nvidia-tuning-guide, doc-ptx-isa-sm100]
---

# Pipeline Stalls

## Symptom

Timeline or profiler data shows producer or consumer agents idle while another pipeline stage is late. A large count of barrier-wait samples is a clue, but a wait can be intentional overlap rather than the root bottleneck.

## Likely causes

1. The producer service rate is lower than the consumer rate, or vice versa.
2. Too few stages expose latency; too many stages reduce occupancy or add backpressure.
3. Barrier phase, expected-byte, arrival-count, or stage-index bookkeeping is wrong.
4. A stage is released before the final asynchronous/read consumer, causing a race.
5. MMA completion is confused with memory ordering: `tcgen05.fence` is not a completion wait.
6. Warp roles share an execution dependency or resource, preventing intended overlap.
7. Short loops spend most time filling and draining the pipeline.

## Diagnosis

```text
1. Reproduce one representative shape and capture per-warp timing.
2. Label every wait with the stage, phase, producer, and releasing consumer.
3. Check TMA expected bytes and confirm that hardware completion—not an extra
   manual arrival—satisfies the transaction component.
4. Check tcgen05.commit/mbarrier waits before dependent TMEM access.
5. Sweep stage count while recording resources and occupancy.
6. Compare steady-state intervals separately from prologue, tail, and epilogue.
```

Use [pipeline stages](../techniques/pipeline-stages.md) when load latency is exposed, [warp specialization](../techniques/warp-specialization.md) when independent agents can make concurrent progress, and [double-buffering](../techniques/double-buffering.md) when storage reuse is the dependency.

Tutorial throughput progressions are whole-kernel observations. They should not be relabeled as isolated gains from one technique unless only that technique changed.

## Caveats

Correctness comes first: adding stages to a phase bug can make a rare race harder to reproduce. Run boundary-shape and repeated-launch tests before interpreting speedups.
