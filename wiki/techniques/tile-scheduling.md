---
id: technique-tile-scheduling
title: "Tile Scheduling Strategies"
type: technique
architectures: [sm100, sm90]
tags: [tile-scheduling, clc, persistent-kernel]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-clc]
related: [hw-clc, technique-persistent-kernels, pattern-low-sm-utilization]
sources: [doc-cutlass-clc, doc-cutlass-blackwell]
blackwell_relevance: "SM100 CLC can reassign unlaunched grid IDs; raster order, swizzle, persistence, and Stream-K remain distinct software scheduling choices."
---

## Keep the Scheduling Layers Separate

“Tile scheduling” can refer to four different choices:

1. **Coordinate order:** a software mapping from a logical work index to `(tile_m, tile_n, ...)`, such as row-major, column-major, or a blocked/swizzled raster.
2. **Resident-worker iteration:** whether a CTA handles one logical tile or repeatedly advances through work, for example by a static grid stride.
3. **Work reassignment:** on SM100, Cluster Launch Control (CLC) lets a running worker cancel another grid entity that has not started and process the returned ClcID.
4. **Problem decomposition:** Stream-K or Split-K can partition a tile's K work and then reduce partial results.

CLC does not accept a raster-order or swizzle-policy operand. CUTLASS applies those coordinate transforms in software to initial or returned coordinates. CLC also does not itself create K partitions or synthesize work beyond the launched grid.

## Exact Static Mappings

A flat row-major mapping is:

```cuda
__device__ void row_major_tile(int tile_idx, int tiles_n,
                               int& tile_m, int& tile_n) {
  tile_m = tile_idx / tiles_n;
  tile_n = tile_idx % tiles_n;
}
```

For a positive grid size, a static persistent worker can cover the flat index set without overlap:

```cuda
for (int tile_idx = int(blockIdx.x);
     tile_idx < total_tiles;
     tile_idx += int(gridDim.x)) {
  int tile_m = tile_idx / tiles_n;
  int tile_n = tile_idx % tiles_n;
  compute_tile(tile_m, tile_n);
}
```

Changing coordinate order changes the sequence of operand panels presented to the cache. It does not by itself prove a hit rate or speedup. A valid transform must first be shown to be in-bounds and bijective for edge groups; then its locality must be evaluated for the actual tile shape, batch/group structure, concurrent traffic, and cache capacity.

## SM100 CLC Work Acquisition

CUTLASS 4.5.0 documents this lifecycle for a CLC-backed persistent scheduler:

1. Launch the full problem grid and process the worker's initial `blockIdx` coordinate.
2. Submit `clusterlaunchcontrol.try_cancel` against another not-yet-started grid entity. The asynchronous 16-byte response completes a transaction on an mbarrier.
3. After the transaction completes, query whether cancellation succeeded. On success, decode and process the returned ClcID; for clusters, combine the returned first coordinate with the local cluster rank.
4. Treat an observed failed request as terminal for requests by that thread. Retrying from the same thread after failure is undefined.

This can redistribute existing work when SM availability is uneven. It changes which resident worker handles an unlaunched ID, not the number of independent IDs in the grid. Software raster/swizzle may be applied consistently to both initial and returned coordinates.

## CUTLASS 4.5.0 Scheduler Routes

The public scheduler tags select implementation classes; users should prefer those tags over spelling detail types directly.

| Public tag for `arch::Sm100` | Selected implementation | Scope |
|---|---|---|
| `PersistentScheduler` (also the default `void` tag) | `PersistentTileSchedulerSm100` | CLC-backed persistent data-parallel scheduling |
| `DynamicPersistentScheduler` | `PersistentTileSchedulerSm100` | Explicit dynamic route to the same SM100 implementation |
| `StaticPersistentScheduler` | `StaticPersistentTileScheduler100` | Static persistent scheduling |
| `StreamKScheduler` | `PersistentTileSchedulerSm100StreamK` | Parameterized data-parallel, Stream-K, or Split-K decomposition |
| `GroupScheduler` | `PersistentTileSchedulerSm100Group` | Grouped-problem route; in this tag it wraps the SM90-style static group scheduler |

The SM100 Stream-K route accepts decomposition, split, raster, swizzle, and reduction settings. CUTLASS 4.5.0 has a deterministic lock/turnstile reduction and a nondeterministic atomic-workspace reduction; atomic accumulation is therefore not a universal property of every Stream-K configuration.

## Tail Arithmetic

For a deliberately simplified model with `T` equal-duration independent tiles, `W` available one-tile workers, and no K decomposition, write:

```text
T = qW + r,  0 <= r < W
```

If `r > 0`, the final partial wave has `r` active workers and occupancy `r / W`. If `r == 0` and `T > 0`, the final wave is full; reporting zero occupancy from the remainder alone is an error.

For `T = 150` and `W = 142`, the second wave contains 8 tiles, so 8 workers are active, 134 are idle, and wave occupancy is `8 / 142 = 5.63%` under those assumptions. CLC may change which workers receive those eight IDs and reduce delays caused by uneven worker availability, but it cannot turn eight independent IDs into 142. Stream-K may create additional K partitions, but its selected decomposition and reduction overhead determine whether that is profitable; near-100-percent occupancy is not guaranteed.

## Selection and Verification Workflow

1. Establish a correct unswizzled data-parallel mapping and verify exact tile coverage, including nondivisible edge groups.
2. Sweep legal raster orders and swizzle sizes. Record kernel time plus L2 hit/sector traffic; do not infer eviction from coordinate order alone.
3. Compare static and CLC-backed persistence at identical tile, cluster, grid, and occupancy settings. Record successful/failed CLC requests and per-worker tile counts where instrumentation permits.
4. Test Stream-K separately with explicit decomposition, split, and reduction modes. Include workspace traffic, synchronization, determinism, and numerical tolerance.
5. Repeat across representative shapes and grouped-size distributions. Report regressions as well as wins; there is no universal best scheduler for all GEMMs, attention kernels, or MoE workloads.

Without target-GPU measurements, this page makes no fixed claim for CLC acquisition latency, L2-miss reduction, or scheduler speedup.
