---
id: technique-pipeline-stages
title: "Software Pipelining and Multi-Stage Buffering"
type: technique
architectures: [sm100, sm90]
tags: [pipeline-stages, double-buffering, tma, mbarrier]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
reproducibility: snippet
prerequisites: [hw-tma, hw-tmem]
related: [technique-warp-specialization, technique-double-buffering, hw-tma]
sources: [blog-tcgen05-tutorial, blog-modular-blackwell, doc-nvidia-tuning-guide, doc-ptx-isa-sm100, pr-cutlass-2139]
blackwell_relevance: "SM100 software pipelines coordinate TMA operand stages and asynchronous tcgen05 completion with phase-correct barriers."
---

# Software Pipelining and Multi-Stage Buffering

## Overview

A software pipeline overlaps operations from different loop iterations—for example, a TMA producer loading K tile `k+1` while an MMA consumer processes tile `k`. A circular set of shared-memory stages prevents the producer from overwriting data still in use.

Pipelining can reduce exposed latency, but it cannot guarantee that latency is “entirely hidden.” The result depends on transfer and compute rates, instruction dependencies, pipeline fill/drain, resource occupancy, and problem length.

## State machine

```python
def staged_mainloop(k_tiles, stages):
    for k, tile in enumerate(k_tiles):
        stage = stages[k % len(stages)]

        producer_waits_for_empty_phase(stage)
        producer_issues_tma(stage, tile)
        # TMA completion supplies the stage-full transaction bytes.

        consumer_waits_for_full_phase(stage)
        consumer_issues_mma(stage)
        consumer_releases_stage_after_last_read(stage)

    drain_outstanding_operations(stages)
```

This pseudocode expresses ownership only. A correct implementation uses the precise `mbarrier` phase/parity protocol. For SM100, it separately commits and waits for asynchronous `tcgen05` work before dependent TMEM access; `tcgen05.fence` provides memory ordering, not completion.

## Selecting stage count

Stage count is a tuning parameter constrained by:

- bytes per A/B stage and required alignment;
- barrier and epilogue shared storage;
- register state needed to track in-flight work;
- CTA occupancy and cluster residency;
- K-loop length and load/compute balance.

Two stages may suffice for a compute-heavy loop; a longer-latency path may benefit from more. Extra stages can reduce performance by lowering occupancy or extending the prologue. CUTLASS stage-count traits and CuTe layouts are useful starting points, but the chosen tile and datatype still need compilation/resource checks and measurement.

## Reading tutorial progressions

The `tcgen05` and Modular tutorial results document complete stepwise kernels. If a later step also changes swizzle, tile shape, launch bounds, role mapping, or synchronization, its throughput delta is not a controlled estimate of “pipelining alone.” Preserve the reported environment and describe the result as tutorial-specific.

## Verification

- Test fewer K tiles than stages, exact wraparound, and partial boundary tiles.
- Detect deadlocks with repeated launches and randomized shapes.
- Confirm expected-byte counts and phase changes for every stage.
- Check the generated resource usage and achieved occupancy.
- Profile whether producer starvation or consumer backpressure remains before increasing stages.
