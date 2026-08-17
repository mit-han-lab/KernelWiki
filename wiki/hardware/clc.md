---
id: hw-clc
title: "Cluster Launch Control (CLC)"
type: hardware
architectures: [sm100, sm100a]
tags: [clc, persistent-kernel, tile-scheduling]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2161
    evidence_type: upstream-code
related: [technique-persistent-kernels, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, doc-cutlass-blackwell, pr-cutlass-2161]
aliases: [CLC, "cluster launch control"]
---

# Cluster Launch Control (CLC)

## What CLC does

On SM100, a running cluster can ask the launch controller to cancel a cluster in the same grid that has not yet started. If cancellation succeeds, the response contains the CTA ID of the first CTA in the canceled cluster. Software can map that coordinate to the work the canceled cluster would have performed and execute it in the running cluster.

This supports persistent tail work without a global atomic work counter. It does **not** replace CUDA grid scheduling with an arbitrary hardware tile queue, cancel running work, or cancel application outputs such as rejected speculative tokens.

## PTX protocol

The operation is asynchronous:

1. Prepare a 16-byte aligned shared-memory response area and an initialized `mbarrier`.
2. Issue `clusterlaunchcontrol.try_cancel` with the response address and completion barrier.
3. Wait for the barrier phase.
4. Use `clusterlaunchcontrol.query_cancel.is_canceled` to test success.
5. If successful, use `clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128` to read the first canceled-cluster CTA ID from the 128-bit response.

```python
def clc_iteration(response, barrier):
    issue_try_cancel(response, barrier)
    wait_for_response_completion(barrier)
    if not query_is_canceled(response):
        return None
    return query_canceled_cluster_id(response)
```

The pseudocode hides exact PTX operands deliberately. `try_cancel` does not directly return a predicate and tile coordinates, and there is no `clc.arrive`, `clc.wait`, or `try_acquire` instruction.

## Scheduling remains software-defined

The returned CTA ID is a grid coordinate for the first CTA in the canceled cluster. CUTLASS converts it to its own logical work coordinate using scheduler metadata and may apply rasterization or swizzling in software. CLC itself has no “Hilbert,” “swizzled raster,” or grouped-GEMM policy setting.

Because only unlaunched clusters can be canceled, a request can fail while other clusters are already resident or no cancelable work remains. Correct code treats failure as termination of the steal loop and still accounts for the running cluster's original work.

## Performance boundary

CLC can reduce a partially occupied later wave when running clusters finish early enough to cancel future launches. It cannot create parallelism when the grid has fewer clusters than the machine, split a tile across idle SMs, or guarantee balance for variable-duration work already running.

Measure cancellation success rate, scheduler overhead, cluster residency, and tail duration. Tutorial improvements that also change persistence, tile order, pipeline, or launch shape are not controlled measurements of CLC alone.
