---
id: hw-clc
title: "Cluster Launch Control (CLC)"
type: hardware
architectures: [sm100, sm100a]
tags: [clc, persistent-kernel, tile-scheduling]
confidence: source-reported
related: [technique-persistent-kernels, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-ptx-isa-sm100, doc-cutlass-clc]
aliases: [CLC, "cluster launch control"]
---

# Cluster Launch Control (CLC)

## Overview

Cluster Launch Control is a Blackwell compute-capability 10.0 mechanism for
work stealing between running and not-yet-started thread blocks or thread block
clusters. A CLC kernel still launches the problem-sized grid. Each grid
coordinate, called a ClcID by CUTLASS, is processed in exactly one of two ways:

1. The block or cluster launches normally and first processes its own
   `blockIdx`.
2. An existing worker successfully cancels that not-yet-started ClcID and
   processes the returned coordinate itself.

This lets a persistent worker request subsequent work without maintaining a
separate software work queue. It is especially useful when the set of SMs
available to a kernel is uneven or changes while the grid is running.

## Request and Decode Protocol

`clusterlaunchcontrol.try_cancel` is asynchronous. It writes an opaque 16-byte
response to shared memory and completes a transaction on a shared-memory
`mbarrier`. The request does not accept a tile coordinate to cancel.

The normative PTX instruction forms are:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response], [mbar];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response_b128;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x, y, z, unused}, response_b128;
```

A complete loop must:

1. Allocate and initialize a shared response and `mbarrier`.
2. Submit one asynchronous request from the selected thread and set the
   expected transaction size to 16 bytes.
3. Wait for the corresponding barrier phase to complete.
4. Decode `is_canceled`; decode `get_first_ctaid` only after success.
5. Apply the async/generic proxy fences needed before reusing the response.

After a thread has observed a failed request, issuing another request from that
thread is undefined. Decoding a CTA ID from a failed response is also undefined.
A request can fail because no ClcIDs remain or for another scheduling reason,
including pending higher-priority work.

## Thread Block Cluster Rules

CLC cancellation is cluster-granular when the kernel uses thread block
clusters. One cluster thread submits the multicast request. Every CTA tracks
completion with its local shared-memory barrier at cluster scope and receives
the same encoded first-CTA coordinate. Each CTA then adds its local block rank
to that first coordinate. A cluster synchronization is needed to guarantee all
blocks exist before cluster cancellation begins.

For example, a successful query by a 2x2 worker cluster consumes the matching
2x2 group of ClcIDs; it is not four unrelated CTA-level cancellations.

## CUTLASS 4.5.0 Integration

At CUTLASS 4.5.0, the `PersistentScheduler` tag for `arch::Sm100` maps to
`PersistentTileSchedulerSm100`; newer code can select the intent explicitly
with `DynamicPersistentScheduler`. The SM100 scheduler uses
`PipelineCLCFetchAsync`: `advance_to_next_work()` submits a request, while
`fetch_next_work()` waits for and decodes the staged response.

A schematic of the relevant CUTLASS 3.x kernel composition is:

```cpp
using TileScheduler = cutlass::gemm::DynamicPersistentScheduler;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;
```

CUTLASS can apply a software coordinate transform to both the initial
`blockIdx` and decoded CLC responses. `max_swizzle_size` and `raster_order` are
scheduler arguments used by `swizzle_and_rasterize()`; they are not operands of
the CLC PTX instruction or a CLC hardware policy configured at launch.

## Scope and Limits

CLC redistributes ClcIDs that already exist in the launched grid. It does not:

- discard an application-selected output tile;
- create more independent tiles than the problem grid contains;
- guarantee that every SM remains occupied; or
- universally eliminate GEMM wave quantization.

Consequently, a 32-tile grid exposes at most 32 independent ClcIDs even on a
148-SM B200. Any CLC performance result must state the GPU, available SMs,
software versions, tile and cluster shapes, data types, timed region, and
measurement method.

## Related

- [Persistent kernels](../techniques/persistent-kernels.md)
- [Tile scheduling](../techniques/tile-scheduling.md)
- [Two-SM cooperative MMA](2sm-cooperative.md)
