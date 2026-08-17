---
id: doc-cutlass-clc
title: CUTLASS Cluster Launch Control documentation
url: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [clc, cluster, tile-scheduling, persistent-kernel, mbarrier, 2sm-cooperative, pipeline-stages, gemm]
retrieved_at: 2026-08-16
---

# CUTLASS Cluster Launch Control documentation

## Mechanism

Cluster Launch Control (CLC) lets a running cluster attempt to cancel a cluster that has not yet launched. A successful cancellation returns the canceled cluster's launch identity; the running cluster can use that identity to execute the corresponding logical work. A failed cancellation means no work identity was transferred. This is hardware-assisted work stealing from the unlaunched tail of the grid, not an arbitrary device-side queue of tiles already executing.

CUTLASS's persistent SM100 schedulers wrap the query/response operation in an asynchronous pipeline and map returned launch coordinates to GEMM work. Pipeline depth, scheduler-warp number, response staging, cluster shape, and stream-K behavior are properties of the selected CUTLASS scheduler, not fixed CLC architectural constants.

## Consequences

- The original grid still defines the launch identities available for cancellation.
- Only a successful cancellation yields additional work.
- Cluster-granular kernels must preserve the selected cluster shape and coordinate mapping.
- CLC can reduce tail underutilization, but cannot guarantee balance or eliminate every tail effect.
- Preferred/fallback cluster launch configuration is related launch machinery; it should not be described as CLC dynamically choosing any cluster size per tile.

The former local page generalized one CUTLASS configuration into universal values such as a 16-byte response, pipeline depth three, “warp 1” scheduler, and a 2×2 consumption rule. Those details are no longer presented as architecture-wide facts.

Primary sources: [CUTLASS CLC documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html) and the [PTX ISA CLC instruction section](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-launch-control-instructions).
