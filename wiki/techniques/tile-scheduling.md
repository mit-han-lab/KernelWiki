---
id: technique-tile-scheduling
title: "Tile Scheduling Strategies"
type: technique
architectures: [sm100, sm90]
tags: [tile-scheduling, clc, persistent-kernel]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2161, evidence_type: upstream-code}
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-persistent-kernels, pattern-low-sm-utilization]
sources: [doc-nvidia-tuning-guide, doc-cutlass-blackwell, doc-ptx-isa-sm100, pr-cutlass-2161, pr-cutlass-2139]
blackwell_relevance: "SM100 Cluster Launch Control can transfer an unlaunched cluster's work identity to a running persistent cluster; software still defines the tile mapping and ordering."
---

# Tile Scheduling Strategies

## Overview

Tile scheduling maps logical work units to CTAs or clusters and orders their execution. The mapping affects locality, wave quantization, load balance, and synchronization overhead. No ordering is universally best.

## Static mappings

```python
def persistent_static_stride(worker_id, resident_workers, tile_count):
    tile = worker_id
    while tile < tile_count:
        compute_tile(tile)
        tile += resident_workers
```

Linear, column-major, grouped, and swizzled raster orders are software mappings. Swizzling the logical traversal can improve reuse for some dimensions, but it can also enlarge the live working set or harm another operand's locality. Tune it using measured cache traffic.

## What CLC actually provides

Cluster Launch Control does not expose an API to initialize “linear,” “Hilbert,” or “swizzled” hardware policies. A running cluster uses `clusterlaunchcontrol.try_cancel` to request cancellation of a cluster that has not yet launched. After waiting on the associated `mbarrier`, `clusterlaunchcontrol.query_cancel` reports whether cancellation succeeded and, if so, the canceled cluster's multidimensional launch ID. The running cluster can execute the work that software maps to that ID.

```python
def clc_worker(own_launch_id):
    work_id = own_launch_id
    while True:
        compute_work_mapped_by_software(work_id)
        response = try_cancel_an_unlaunched_cluster()
        wait_for_clc_response(response.barrier)
        if not response.succeeded:
            break
        work_id = response.canceled_cluster_id
```

CLC therefore supports persistent tail work and dynamic reassignment of not-yet-launched cluster IDs. It does not steal an already running CTA, split one residual tile across all SMs, or guarantee a balanced final wave.

## Other scheduling families

- Grouped GEMM schedules maintain problem descriptors and prefix/range mappings for variable-sized groups.
- Stream-K partitions the reduction dimension and then combines partial results. It can reduce wave quantization but adds workspace, fix-up, or atomic/reduction costs.
- Cluster-shape selection can trade multicast/reuse against the number of independently schedulable workers.

Use the exact scheduler types available in the pinned CUTLASS release; illustrative class names from older code are not an API guarantee.

## Correctness and performance checks

- Prove that every logical tile is executed exactly once, including failed cancellation and boundary IDs.
- Keep cluster-wide control flow compatible with collective instructions.
- Account for split-K reduction order and numerical differences.
- Compare scheduler overhead against tile duration; CLC latency is not a fixed architecture-wide number published by the cited sources.
- Benchmark cold/warm cache behavior and the final partial wave separately.
