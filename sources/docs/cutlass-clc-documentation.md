---
id: doc-cutlass-clc
title: "CUTLASS 4.5.0 Cluster Launch Control Documentation"
url: https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/blackwell_cluster_launch_control.md
source_category: official-doc
architectures: [sm100, sm100a]
tags: [clc, cluster, tile-scheduling, persistent-kernel, mbarrier, 2sm-cooperative, pipeline-stages, gemm]
retrieved_at: 2026-08-09
---

# CUTLASS 4.5.0 Cluster Launch Control

## Verified scope

This card is pinned to CUTLASS v4.5.0 commit `e406c186f510a15091cce01f782020ceb7ba8eb5`.

CLC launches the full problem grid. Every ClcID is processed either by the block/cluster that launches at that coordinate or by a running worker that successfully cancels that not-yet-started ClcID and receives its coordinate. A worker therefore processes its initial `blockIdx` before requesting later work.

`clusterlaunchcontrol.try_cancel` writes a 16-byte response asynchronously to shared memory and completes a transaction on an mbarrier. The response is decoded only after completion. A failed response is terminal for requests from that thread; issuing another request from the same thread after observing failure is undefined.

For thread-block clusters, cancellation and the returned first coordinate are cluster-granular. Each participating CTA adds its local cluster rank to derive its coordinate.

## CUTLASS integration

CUTLASS 4.5.0 uses `PersistentTileSchedulerSm100` and `PipelineCLCFetchAsync`. Scheduler methods such as `advance_to_next_work()` and `fetch_next_work()` stage and consume CLC requests. Exact pipeline depth and participant counts are kernel configuration, not universal CLC constants.

Swizzle size and raster order are CUTLASS software coordinate-transform policy applied to initial and returned coordinates. They are not operands or policies programmed into the raw CLC instruction. Stream-K decomposition and multi-problem scheduling are likewise higher-level scheduler decisions, not work synthesized by CLC.

## References

- [Pinned CLC documentation](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/blackwell_cluster_launch_control.md)
- [Pinned SM100 scheduler](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp)
- [PTX ISA 9.0 CLC instructions](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
