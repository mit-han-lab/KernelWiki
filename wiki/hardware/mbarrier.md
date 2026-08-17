---
id: hw-mbarrier
title: "mbarrier (Memory Barrier Primitives)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [mbarrier, tma]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
related: [hw-tma, hw-tcgen05-mma, technique-warp-specialization, technique-pipeline-stages]
sources: [doc-ptx-isa-sm100, blog-tcgen05-tutorial, doc-nvidia-tuning-guide, pr-cutlass-2139]
aliases: [mbarrier, "memory barrier", "mbar"]
blackwell_relevance: "mbarrier tracks thread arrivals and asynchronous transaction completion for TMA/tcgen05 pipelines; circular reuse requires per-stage phase state."
---

# `mbarrier`

## Overview

An `mbarrier` is a 64-bit object in shared memory that tracks a phase, pending arrivals, and (when used) pending asynchronous transaction bytes. The primitive requires SM80 or later; it was not first introduced on Hopper.

## Core operations

```ptx
mbarrier.init.shared::cta.b64 [bar], arrival_count;
mbarrier.arrive.shared::cta.b64 state, [bar];
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], expected_bytes;
mbarrier.try_wait.parity.shared::cta.b64 ready, [bar], phase;
```

Exact qualifiers and return operands vary by instruction form. Initialization must complete and become visible before other threads use the object.

## Arrivals and transactions

Phase completion requires both the arrival-count and transaction-count conditions to be satisfied. A TMA `mbarrier::complete_tx::bytes` operation reduces the pending transaction bytes when the transfer completes. `tcgen05.commit` can similarly attach MMA completion to an `mbarrier`.

`arrive.expect_tx` is itself an arrival plus an increase in expected transaction bytes. Whether an additional software arrival is required depends on the barrier's initialized count and the exact producer protocol; the safe rule is to derive the counts explicitly, not “always arrive” or “never arrive.”

## Circular reuse

Each stage has its own phase history. If stage `s` is reused every `N` loop iterations, its parity advances when **that stage's** previous phase completes—not on every global loop iteration.

```python
def use_stage(stage):
    wait_for_expected_phase(stage.barrier, stage.consumer_phase)
    consume(stage)
    stage.consumer_phase ^= 1
```

Practical pipelines often use separate full/empty barriers or token objects so producer and consumer ownership is explicit.

## Failure modes

- Reusing or reinitializing a live barrier.
- Incorrect expected-byte sum for multiple TMA operations.
- Flipping a single global parity instead of per-stage state.
- Confusing observation of barrier completion with unrelated proxy/order requirements.
- Using a CTA-scoped address/qualifier when a remote cluster CTA must access the object.
- Destroying barrier storage before pending async operations finish.
