---
id: technique-warp-specialization
title: Warp Specialization on Blackwell
type: technique
architectures:
- sm100
- sm90
tags:
- warp-specialization
- tcgen05
- tmem
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
reproducibility: snippet
prerequisites:
- hw-tmem
- hw-tcgen05-mma
related:
- technique-persistent-kernels
- technique-pipeline-stages
- hw-tcgen05-mma
sources:
- doc-ptx-isa-sm100
- pr-cutlass-2139
- doc-nvidia-tuning-guide
- blog-tcgen05-tutorial
- blog-colfax-cutlass
blackwell_relevance: Blackwell tcgen05 uses single-thread MMA issuance and TMEM accumulators,
  enabling kernel-specific warp roles for scheduling, TMA, MMA control, and epilogues.
artifact_dir: artifacts/kernels/warp-specialization
---

## Overview

Warp specialization assigns different warps in one CTA to different pipeline roles. On SM100, `tcgen05.mma` is issued by one thread and accumulates in TMEM, so a warp can serve as an MMA-control role while other warps schedule tiles, move operands with TMA, or run the epilogue.

Single-thread issuance does not define a fixed CTA size or warp map. The earlier “canonical 16 warps: 1 TMA + 1 MMA + 14 epilogue” description was not a CUTLASS invariant.

## A concrete CUTLASS 4.5.0 mapping

The official `sm100_gemm_tma_warpspecialized.hpp` kernel defines these thread counts:

| Role | Threads in the kernel header |
|---|---:|
| Tile scheduler / CLC | 32 (one warp) |
| MMA control | 32 (one warp) |
| Mainloop TMA load | 32 (one warp) |
| Epilogue TMA load | 32 (one warp) |
| Epilogue compute/store | `CollectiveEpilogue::ThreadCount` (configuration-dependent) |

Its `WarpCategory` assigns warp indices 0–3 to MMA, scheduler, mainloop load, and epilogue load respectively; indices 4 and above are epilogue warps. Consequently the total is `4 + NumEpilogueWarps`, not universally 16. Other CUTLASS schedules and attention kernels can choose other decompositions.

Source: [CUTLASS v4.5.0 SM100 warp-specialized GEMM kernel](https://github.com/NVIDIA/cutlass/blob/v4.5.0/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp).

## Pipeline dependencies

A typical SM100 GEMM has several distinct handoffs:

1. The scheduler supplies a work tile, optionally using Cluster Launch Control for persistent scheduling.
2. The mainloop-load warp waits until a shared-memory stage is empty, sets the transaction expectation, and launches TMA copies.
3. The MMA warp waits for the stage-full condition, orders its `tcgen05` access after that synchronization, and issues MMA operations.
4. The MMA completion path uses `tcgen05.commit` and an mbarrier; an ordering fence alone is not an MMA completion event.
5. Epilogue warps wait for an accumulator subtile, collectively load from TMEM, apply the output operation, and store. The epilogue-load warp exists only when source-C data must be loaded.

The stage-full and stage-empty barriers form a ring. Phase/parity state must be advanced according to the pipeline implementation so a later reuse cannot be mistaken for an earlier arrival.

## Illustrative role dispatch

This pseudocode conveys role ownership only. It is not a drop-in kernel and intentionally delegates the barrier protocol to checked pipeline objects:

```cuda
int warp = cutlass::canonical_warp_idx_sync();

if (warp == 0) {
  mma_consumer.run(mainloop_full, accumulator_full);
} else if (warp == 1) {
  scheduler.run(clc_pipeline);
} else if (warp == 2) {
  mainloop_loader.run(mainloop_empty, mainloop_full);
} else if (warp == 3) {
  epilogue_loader.run(epilogue_load_pipeline);  // when source C is needed
} else {
  epilogue.run(accumulator_full, epilogue_store_pipeline);
}
```

In real CUTLASS code the pipeline objects carry producer/consumer roles, arrival counts, stage state, and cluster semantics. Copying only the branch structure without those invariants is not correct.

## Hopper comparison

| Aspect | Hopper SM90 | Blackwell SM100 |
|---|---|---|
| MMA issue model | Warpgroup executes `wgmma` | One thread issues `tcgen05.mma` |
| Accumulator location | Registers | TMEM |
| Mainloop overlap | Commonly warpgroup-specialized TMA/MMA | Warp roles can separately schedule, load, issue MMA, and run epilogue |
| Completion primitive | WGMMA group commit/wait | `tcgen05.commit` plus mbarrier wait |

TMEM reduces accumulator register pressure, but it does not automatically imply higher occupancy; threads, shared memory, barriers, and TMEM allocation remain residency constraints.

## When it helps

- TMA transfers and Tensor Core work have enough independent work to overlap.
- Persistent scheduling can hide tail imbalance across work tiles.
- The epilogue has enough work to overlap with later mainloop stages or tiles.
- Attention pipelines need separate tensor-core, softmax, and data-movement roles.

It is not mandatory for every Blackwell kernel. Small or memory-bound kernels may not have enough pipeline work to repay extra warps and synchronization.

## Correctness caveats

- Initialize shared pipeline state before any participant waits on it.
- Match transaction-byte expectations to the actual TMA transfers.
- Do not release a shared-memory stage merely because MMA was issued; use the pipeline's documented consumption/completion semantics.
- Before a different thread performs dependent `tcgen05` work, use the required thread synchronization plus `tcgen05.fence::after_thread_sync` ordering.
- Before epilogue reads, track MMA completion with `tcgen05.commit` and wait on its mbarrier. A plain CTA barrier is insufficient.

## Full Reference Implementation

Local verbatim upstream code lives in [`artifacts/kernels/warp-specialization/full/`](../../artifacts/kernels/warp-specialization/full/) (see its `PROVENANCE.yaml` for the pinned upstream SHA and byte-verified SHA-256). The former teaching skeleton was removed because it did not preserve the upstream role and synchronization contract closely enough to serve as implementation evidence.

Query via:

```bash
python3 scripts/get_page.py technique-warp-specialization --include-code
```
