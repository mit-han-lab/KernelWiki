---
id: technique-double-buffering
title: "Double/Multi-Buffering Patterns"
type: technique
architectures: [sm100, sm90]
tags: [double-buffering, tmem, pipeline-stages]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
reproducibility: snippet
prerequisites: [hw-tmem]
related: [hw-tmem, technique-pipeline-stages, technique-epilogue-fusion]
sources: [blog-tcgen05-tutorial, doc-nvidia-tuning-guide, doc-ptx-isa-sm100, doc-cutlass-blackwell, pr-cutlass-2139, pr-flashinfer-2387]
blackwell_relevance: "SM100 kernels can pipeline distinct TMEM accumulator stages and SMEM operand stages when resource budgets and dependency protocols permit."
---

# Double/Multi-Buffering Patterns

## Overview

Multi-buffering gives a producer and consumer distinct storage stages so operations on different work units may overlap. SM100 kernels commonly apply it to:

- shared-memory operand stages: TMA fills one stage while MMA consumes another;
- TMEM accumulator stages: MMA writes one allocation region while an epilogue drains a completed region.

The two pipelines have different completion mechanisms and must not be conflated.

## Ownership model

```python
def ping_pong(work_units, stages):
    for sequence, work in enumerate(work_units):
        stage = stages[sequence % len(stages)]
        wait_until_all_consumers_release(stage)
        producer_fills(stage, work)
        signal_stage_ready(stage)
        consumer_waits_and_uses(stage)
        signal_stage_released(stage)
```

Real code also tracks barrier phases. A stage is reusable only after every asynchronous operation and every consumer that can access it has completed.

## TMEM sizing

TMEM is organized as 128 lanes by 512 columns per SM, but a logical accumulator does not consume one 32-bit TMEM cell per matrix element in the simple row-major sense. Allocation is expressed in columns, and the mapping depends on the `tcgen05` instruction and TMEM load/store shapes. Use the PTX allocation and layout tables or a library trait to compute the required columns.

Two 256-column regions are a valid possible partition, not a universal GEMM layout. Some kernels use one stage, more than two stages, narrower regions, or overlapping-accumulator policies selected by CUTLASS.

## SMEM stages

Each input stage consumes the sum of its operand storage plus alignment and barrier metadata. More stages can hide latency only when the producer can run ahead and the consumer has enough work. They also reduce the shared-memory headroom available to occupancy and the epilogue.

## Correctness protocol

- Initialize barriers before any agent can access them.
- Match expected transaction bytes to every TMA operation completing on the barrier.
- Track barrier phases across circular reuse.
- Commit `tcgen05` completion to an `mbarrier` and wait before draining the corresponding TMEM stage.
- Count every epilogue participant before declaring a stage free.
- Tail the pipeline: complete pending loads/stores and drain the final accumulator stages.

## Performance boundary

Buffering is not free and is not always beneficial. It adds storage, barrier traffic, prologue/epilogue work, and sometimes register state. Short loops, small problems, or occupancy-sensitive kernels may prefer fewer stages. Choose stage counts with resource accounting and an end-to-end benchmark rather than a fixed “three to five” rule.
