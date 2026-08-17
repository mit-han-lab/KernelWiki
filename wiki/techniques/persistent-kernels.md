---
id: technique-persistent-kernels
title: Persistent Kernels with CLC
type: technique
architectures: [sm100]
tags: [persistent-kernel, clc, tile-scheduling]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2161, evidence_type: upstream-code}
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, doc-cutlass-blackwell, pr-cutlass-2161]
artifact_dir: artifacts/kernels/persistent-kernels
---

# Persistent Kernels with CLC

## Overview

A persistent kernel keeps resident CTAs or clusters alive across multiple logical work units. The physical grid is commonly sized from an occupancy-aware resident-worker count, but it need not equal the SM count: clusters may span multiple SMs, more than one CTA may reside per SM, and resource limits can change residency.

SM100 CLC gives a running cluster a hardware-assisted way to take responsibility for the launch ID of an unlaunched cluster. Software still maps each launch ID to a logical tile.

## Control-flow sketch

```python
def persistent_cluster(original_cluster_id):
    work_id = original_cluster_id
    while True:
        compute_work_for_launch_id(work_id)
        canceled_id = try_cancel_and_query_unlaunched_cluster()
        if canceled_id is None:
            break
        work_id = canceled_id
```

The real cancel/query uses a shared-memory response and an `mbarrier`. Every CTA in a cluster must follow control flow compatible with cluster collectives. Resources such as TMEM and pipeline barriers can be allocated once and reused only after per-work-unit completion is established.

## Comparison with a software stride

A static persistent worker can process `worker_id + n * resident_workers`. That is deterministic and portable but does not adapt to work-duration variance. A software atomic queue adapts but introduces atomic traffic. CLC can avoid that counter for future cluster IDs, subject to the “not yet launched” constraint.

None of these approaches inherently chooses the best cache order. Rasterization and swizzle remain scheduler policy.

## When persistence may help

- There are substantially more logical work units than resident workers.
- Per-launch overhead or partially filled waves are material.
- Reusing long-lived kernel state is valuable.
- Work can be reassigned without violating ordering or determinism requirements.

For grids smaller than machine capacity, persistence cannot manufacture missing parallelism. For uniform large grids, a conventional launch may already schedule well and have lower in-kernel overhead.

## Validation

- Prove exactly-once coverage for original and canceled cluster IDs.
- Test grids smaller than, equal to, and slightly larger than resident capacity.
- Test failed cancellation and multidimensional cluster IDs.
- Ensure all asynchronous work completes before state reuse or exit.
- Compare full latency and throughput, not only the steady-state loop.

## Full Reference Implementation

Verbatim upstream code lives in [`artifacts/kernels/persistent-kernels/full/`](../../artifacts/kernels/persistent-kernels/full/). The former CLC teaching skeleton was removed because it represented CLC as an arbitrary tile queue rather than the documented launch-ID cancellation protocol.
