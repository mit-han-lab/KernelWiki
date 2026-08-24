---
id: doc-cutlass-clc
title: CUTLASS Blackwell Cluster Launch Control
url: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
source_category: official-doc
architectures: [sm100]
tags: [clc, cluster, tile-scheduling, persistent-kernel, mbarrier, pipeline-stages, gemm]
retrieved_at: 2026-08-18
---

# CUTLASS Blackwell Cluster Launch Control

The CUTLASS documentation contrasts static persistent scheduling with
Blackwell CLC dynamic scheduling. A CLC kernel launches a grid whose coordinates
represent logical work IDs (`ClcID`s). Each worker begins with its own
`blockIdx`; for later work, an elected scheduler thread issues
`clusterlaunchcontrol.try_cancel`. A success response supplies the ID of a
same-grid cluster that was canceled before launch, while a decline supplies no
new work.

Important documented constraints are:

- cancellation and ID consumption happen at cluster granularity;
- the query response is asynchronous and is coordinated through a CLC fetch
  pipeline;
- only one elected scheduler-warp thread issues the request in the described
  CUTLASS design;
- `advance_to_next_work()` produces a query and `get_current_work()` reads the
  response from the shared-memory pipeline stage;
- the example CUTLASS warp-specialized scheduler uses a three-stage CLC
  pipeline, but that is an implementation choice in the documented scheduler,
  not an ISA-wide required depth.

The mechanism can improve load balance when fewer SM resources are actually
available than static scheduling assumed. The documentation does not make CLC
an arbitrary hardware tile queue, guarantee removal of tail imbalance, or
permit retry after a failed `try_cancel`; those details are bounded by the PTX
ISA contract.
