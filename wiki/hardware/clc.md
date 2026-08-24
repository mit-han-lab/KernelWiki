---
id: hw-clc
title: "Cluster Launch Control (CLC)"
type: hardware
architectures: [sm100, sm103, sm110, sm120, sm121]
tags: [clc, persistent-kernel, tile-scheduling]
confidence: source-reported
related: [technique-persistent-kernels, technique-tile-scheduling, pattern-tail-effect]
sources: [doc-ptx-isa-sm100, doc-cutlass-blackwell]
aliases: [CLC, "cluster launch control"]
---

# Cluster Launch Control (CLC)

Cluster Launch Control lets a running CTA request cancellation of a cluster from the same grid that has not launched yet. The request is atomic and asynchronous. If it succeeds, the response contains the CTA ID of the first CTA in the canceled cluster; a persistent scheduler can map that ID to the work that the running CTA or cluster will execute next.

The current PTX ISA says `clusterlaunchcontrol.try_cancel` requires an
`sm_100` target or higher. Product support and performance should still be
checked with the toolkit and device actually used.

This is work acquisition, not cancellation of a chosen output tile. The instruction does not take a tile coordinate and does not discard completed or unwanted results.

## PTX protocol

The PTX ISA defines the request shape as:

```ptx
clusterlaunchcontrol.try_cancel.async{.shared::cta}
    .mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128
    [response_addr], [mbarrier_addr];
```

The operation writes an opaque 16-byte response to shared memory and completes through an mbarrier transaction. After waiting for completion, code loads the response and uses:

```ptx
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, ignored}, response;
```

`get_first_ctaid` is valid only after a successful request. After a CTA observes a failed request, issuing another `try_cancel` from that CTA has undefined behavior.

## Scheduling consequences

- The original grid still defines the available cluster IDs and work domain.
- Successful cancellation prevents one not-yet-launched cluster from running and transfers its identity to already-running work.
- This can reduce replacement-launch overhead and help a persistent scheduler redistribute uneven work, but it cannot create parallelism after fewer independent work items remain than resident SMs.
- Cluster dimensions, response storage, mbarrier accounting, and proxy ordering are correctness requirements, not optional tuning details.

See PTX ISA §9.7.14.18 for [`clusterlaunchcontrol.try_cancel`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel) and §9.7.14.19 for [`clusterlaunchcontrol.query_cancel`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel), including the complete sequence and qualifiers.
