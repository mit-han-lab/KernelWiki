---
id: hw-mbarrier
title: "mbarrier (Memory Barrier Primitives)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [mbarrier, tma, tcgen05]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tma, hw-tcgen05-mma, technique-warp-specialization, technique-pipeline-stages]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139]
aliases: [mbarrier, "memory barrier", "mbar"]
blackwell_relevance: "TMA complete-tx and tcgen05 commit use mbarriers for asynchronous completion; each pipeline stage must track its own phase or state token."
---

# mbarrier

## Scope and history

An mbarrier is an opaque, naturally aligned 64-bit object in shared memory. It synchronizes threads and can track asynchronous operations. The base instructions were introduced in PTX ISA 7.0 for `sm_80`; Hopper (`sm_90`) added transaction-count operations and cluster scope. Blackwell uses the same object for TMA transaction completion and for `tcgen05.commit` arrival-on completion.

mbarriers are useful in warp-specialized pipelines, but they are a mechanism rather than a requirement for every such kernel.

## Object state and phase completion

For the current phase, an mbarrier tracks:

- pending arrivals;
- the expected arrival count for the next phase; and
- a transaction count (`tx-count`) for outstanding asynchronous work.

The current phase completes only when **both** pending arrivals and tx-count reach zero. Completion atomically advances to the next phase and restores pending arrivals from the expected count. Before an arrival in the following phase, at least one `test_wait` or `try_wait` for the completed phase must have returned true.

Initialization sets phase 0, initializes expected and pending arrivals to `count`, and sets tx-count to zero:

```ptx
mbarrier.init.shared::cta.b64 [bar], arrival_count;
```

Invalidate the object with `mbarrier.inval` before reusing its storage for another purpose or reinitializing an already-valid object.

## Arrival, transaction, and wait operations

These operations affect different parts of the state:

| Operation | Effect |
|---|---|
| `mbarrier.arrive` | Decrements pending arrivals; returns a phase state token for a CTA-shared object. |
| `mbarrier.arrive.expect_tx` | Performs an arrival and increments tx-count by `txCount`. |
| `mbarrier.expect_tx` | Increments tx-count without an arrival. |
| `mbarrier.complete_tx` | Decrements tx-count; it is not an arrival. |
| `mbarrier.test_wait` / `try_wait` | Tests completion of the phase identified by a state token or parity. |

Representative CTA-shared forms from PTX ISA 9.0 are:

```ptx
mbarrier.arrive.shared::cta.b64 state, [bar];
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], tx_count;
mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [bar], phase_parity;
```

`try_wait` can suspend temporarily and must still be retried until its predicate is true. Use acquire/release semantics appropriate to the producer-consumer handoff; `.relaxed` does not provide memory-ordering or visibility guarantees.

## Phase and parity in a stage ring

Parity is the low bit of an individual mbarrier object's phase: even phases use 0 and odd phases use 1. A parity wait can refer only to the current or immediately preceding phase, so software must track phase for the entire lifetime of that object.

For an N-stage ring, track state per stage. When stage `s` is reused, pass the parity expected for `bar[s]`; toggle that stage's parity only after its phase completes. One global bit toggled on every loop iteration is generally wrong because different stage barriers advance independently.

Using the opaque state returned by `mbarrier.arrive` is an alternative when the same participant can carry that token to its wait.

## TMA completion accounting

For a common TMA-load phase initialized with one pending arrival:

1. The producer executes `mbarrier.arrive.expect_tx` with the sum of bytes that all TMA operations in this phase will report.
2. The producer issues the TMA operations with `.mbarrier::complete_tx::bytes` and the same barrier.
3. Each TMA completion performs `complete-tx` for the bytes it copied.
4. The consumer waits for the phase to complete before reading the destination.

The `arrive.expect_tx` operation accounts for the software arrival and establishes the expected byte total. TMA hardware does **not** perform a second arrival: it decrements tx-count through complete-tx. Therefore, do not add an unmatched `mbarrier.arrive`, and do not manually simulate TMA's complete-tx. Either mistake can advance or strand the wrong phase.

The expected byte total, barrier address, destination ownership, and exact `cp.async.bulk.tensor` form must match. Follow the complete instruction grammar in the PTX ISA rather than using placeholder helper calls.

## tcgen05 completion accounting

`tcgen05.commit.cta_group::*.mbarrier::arrive::one` makes an mbarrier track prior asynchronous `tcgen05.mma`, `tcgen05.cp`, or `tcgen05.shift` operations issued by that thread. When those operations complete, the system performs one arrive-on operation. This uses pending-arrival accounting, not TMA's byte-valued tx-count protocol.

A consumer in another thread waits for the mbarrier and then uses the applicable `tcgen05.fence::after_thread_sync` ordering sequence. The fence participates in the handoff but does not replace the completion wait.

## CTA and cluster distinctions

State space and synchronization scope are separate concepts:

- `.shared::cta` identifies a barrier in the current CTA's shared memory.
- `.shared::cluster` identifies a cluster-shared address, such as a mapped address for another CTA's shared memory.
- `.cta` or `.cluster` on an operation specifies its synchronization scope.

Only `arrive`, `expect_tx`, and `complete_tx` support an mbarrier address in `.shared::cluster`. Other mbarrier operations, including initialization and waits, target a CTA-shared object. A cluster can initialize an owner CTA's object, map its address for remote arrivals, and have the owner wait with the required cluster scope, as shown by the PTX cluster example. Do not infer that adding `.shared::cluster` alone turns every operation into a cluster barrier.

## Correctness checklist

- Initialize the object before any other mbarrier operation.
- Match all arrivals and byte-valued complete-tx operations to the initialized counts.
- Keep phase or opaque state separately for each pipeline stage.
- Retry `try_wait` until it reports completion.
- Use the required acquire/release semantics for data visibility.
- Distinguish TMA complete-tx from tcgen05 arrive-on completion.
- Respect the supported combinations of object state space and operation scope.
- Invalidate the object before repurposing its storage.

## References

- [PTX ISA 9.0: mbarrier object and lifecycle](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- [PTX ISA 9.0: mbarrier waits](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
- [PTX ISA 9.0: TMA tensor copies](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [TMA](tma.md)
- [Warp specialization](../techniques/warp-specialization.md)
