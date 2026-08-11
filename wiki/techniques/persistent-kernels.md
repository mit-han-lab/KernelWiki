---
id: technique-persistent-kernels
title: Persistent Kernels with CLC
type: technique
architectures:
- sm100
tags:
- persistent-kernel
- clc
- tile-scheduling
confidence: verified
evidence_basis:
- source_id: doc-cutlass-clc
  evidence_type: official-doc
reproducibility: pseudocode
prerequisites:
- hw-clc
related:
- hw-clc
- technique-tile-scheduling
- pattern-tail-effect
sources:
- doc-nvidia-tuning-guide
- blog-tcgen05-tutorial
- doc-cutlass-clc
artifact_dir: artifacts/kernels/persistent-kernels
---

# Persistent Kernels with CLC

## Persistence and CLC are separate choices

A persistent kernel keeps a resident worker alive for multiple logical work items. Its launch size is chosen from the problem decomposition, cluster shape, resource limits, and scheduling policy; “persistent” does not require exactly one CTA per physical SM.

Cluster Launch Control is a compute-capability-10.0 mechanism for redistributing grid coordinates that have not started. A CLC GEMM still launches its problem-sized grid. Each ClcID is processed exactly once through one of two paths:

1. A block or cluster launches normally and processes its initial `blockIdx`.
2. A running worker successfully cancels another not-yet-started block/cluster and processes the returned coordinate.

CLC does not create tiles, cancel the requesting worker, or chain independent problems. It can help when the SMs available to a grid are uneven, but it cannot expose more independent parallel work than exists in the launched grid.

## Request lifecycle

The normative request is asynchronous and returns an opaque 16-byte shared-memory response:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response], [mbar];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response_b128;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x, y, z, unused}, response_b128;
```

A complete implementation must initialize/publish the shared response and mbarrier, set the expected 16 transaction bytes, submit from the designated participant, wait for the matching phase, apply required proxy fences before reading/reusing the response, and decode a coordinate only when `is_canceled` succeeds.

After a thread observes a failed request, another request from that thread is undefined. Failure is therefore not a “retry until another CTA creates work” condition; no CTA creates new ClcIDs.

For a thread-block cluster, one cluster participant submits the multicast request. Every CTA tracks completion at cluster scope, receives the same first coordinate, and adds its local block rank. Cluster synchronization is required before cancellation begins so all peers exist.

## Static stride baseline

A software persistent scheduler can assign a flat tile index by fixed stride:

```cpp
for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
  int tile_m = tile / tiles_n;
  int tile_n = tile % tiles_n;
  compute_tile(tile_m, tile_n);
}
```

For positive `gridDim.x`, this covers each index in `[0,total_tiles)` exactly once, including nondivisible tails. It remains useful as a controlled baseline on Hopper and Blackwell.

## CUTLASS 4.5.0 route

At CUTLASS 4.5.0, the SM100 persistent scheduler integrates CLC through `PersistentTileSchedulerSm100` and `PipelineCLCFetchAsync`. The scheduler's work flow is expressed through `WorkTileInfo` and methods including `get_current_work()`, `advance_to_next_work()`, and `fetch_next_work()`; it is not an API built from `clc_init`, `clc_query_tile`, or a caller-self-cancel helper.

`advance_to_next_work()` submits/stages the next request, while `fetch_next_work()` waits for and decodes a response. CUTLASS applies `swizzle_and_rasterize()` as a software coordinate transform to both the initial `blockIdx` and returned CLC coordinates. `max_swizzle_size` and `raster_order` are scheduler arguments, not CLC hardware policy operands.

The official documentation is the authoritative implementation map; use its pinned scheduler and pipeline links rather than a short pseudo-class.

## Evidence-scoped performance

Gau Nernst's pinned tutorial reports 939.61 TFLOP/s for v3 pipelining and 1475.93 TFLOP/s for v6 on its disclosed 4096-cubed B200 experiment. The v6 endpoint adds warp specialization, 2-SM MMA, and static persistence. The author explicitly did not add CLC or threadblock swizzling, so the endpoint difference is not a CLC ablation and cannot be decomposed into tail, launch, or cache contributions.

To evaluate CLC, compare a static and dynamic scheduler with identical math, tile/cluster shapes, epilogue, launch environment, available-SM policy, warmup, and timing statistics. Record successful/failed requests, work distribution per worker, scheduler stalls, occupancy, and end-to-end time. Test ordinary availability, intentionally uneven availability, nondivisible grids, and cluster modes.

Useful negative controls include decoding a failed response, re-requesting from the same thread after observed failure, omitting initial `blockIdx`, and applying the software swizzle to only one of the initial/returned paths. Each should fail validation or produce missing/duplicate coordinates.

When the grid contains fewer independent tiles than available SMs, CLC cannot manufacture parallelism. Stream-K or another decomposition may create additional work, but that is a separate scheduler transformation with its own reduction and synchronization costs.

## Artifact provenance

- `full/PR-2161-persistent-scheduler-clc.patch` is the recorded upstream PR patch with manifest SHA-256 `aac62aa643631ba452953b61243b657c8f2de11480cbfb6802fdb0157b2ab017`.
- `full/dense_gemm_persistent_prefetch.py` byte-matches CUTLASS PR 2881 merge `3f4c086d09bd1dc55defb955862f333893bbb28b` with SHA-256 `704811ee96850b404b7e1d47e2c0d67f6b19219e3fd8081dc9396b52163f7045`.
- `variants/01-clc-persistent-loop-skeleton.cu` is labeled derived/not-upstream and is not a complete safe CLC kernel.

Retrieve the page and bundle with:

```bash
conda run -n base python scripts/get_page.py technique-persistent-kernels --include-code
```

## Primary references

- [PTX ISA 9.0 CLC instructions](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [CUTLASS 4.5.0 CLC documentation](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/blackwell_cluster_launch_control.md)
- [CUTLASS 4.5.0 SM100 scheduler](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp)
- [Pinned tutorial v6 source](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v6.cu)

## Related

- [Cluster Launch Control](../hardware/clc.md) — exact hardware semantics
- [tile scheduling](tile-scheduling.md) — static, swizzled, dynamic, and Stream-K choices
- [tail effect](../patterns/tail-effect.md) — wave-quantization diagnosis
