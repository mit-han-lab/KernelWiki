---
id: technique-persistent-kernels
title: Persistent Kernels with Cluster Launch Control
type: technique
architectures: [sm100]
tags: [persistent-kernel, clc, tile-scheduling]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-cutlass-clc, doc-ptx-isa-sm100, pr-flash-attention-2441, blog-tcgen05-tutorial]
---

# Persistent Kernels with Cluster Launch Control

A persistent kernel keeps a resident worker CTA or cluster alive for more than
one logical tile. A static scheduler derives later tiles in software. On
Blackwell, a CLC scheduler can instead try to cancel a same-grid cluster that
has not launched and reuse the returned cluster ID as the next work ID.

CLC does not expose an arbitrary tile queue. The grid defines the candidate
cluster IDs. `clusterlaunchcontrol.try_cancel` is asynchronous and atomic; a
successful response identifies the first CTA of the canceled cluster. A failed
request means that this CTA must not issue another request. Correct code must
also observe the shared-memory response, mbarrier transaction, cluster
granularity, and proxy-ordering rules in the PTX ISA.

## Retained implementation excerpt

FlashAttention PR 2441 contains a CuTe DSL wrapper around its hardware
scheduler and CLC response pipeline. The following is a contiguous excerpt from
the captured upstream `flash_attn/cute/tile_scheduler.py`:

```python
self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
mbarrier_addr = self._pipeline.producer_get_barrier(
    self._producer_state, loc=loc, ip=ip
)
self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
self._producer_state.advance(loc=loc, ip=ip)
```

This excerpt shows request production only. Consumers still wait for the
pipeline stage and interpret the scheduler response through the surrounding
implementation.

## Choosing between static and CLC scheduling

- Static persistence has deterministic mapping and no CLC-query path.
- CLC can redistribute cluster IDs that had not yet launched when actual SM
  availability differs from the static assumption.
- Neither mode creates additional independent work at the tail of a grid.
- CLC overhead, cluster shape, locality, tile-cost variation, and kernel
  concurrency must be measured for the actual workload.

The tutorial by Thien Tran (`gau-nernst`) reports 939.61 TFLOPS at its pipelined step and 1475.93
TFLOPS at a later static-persistent step. Intervening changes include warp
specialization and two-SM MMA, so that delta is not an isolated persistence or
CLC measurement.

The former local CUDA skeleton and “CLC full implementation” bundle were
removed: the skeleton used nonexistent PTX forms, and the bundle conflated a
PDL PR with the separate CLC mechanism.
