---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, doc-cutlass-clc, doc-ptx-isa-sm100]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
reproducibility: concept
---

## Define the Missing Parallelism

Low SM utilization means that fewer SMs perform useful work than the workload could profitably use during a material part of its measured time. It is not defined by a universal 60% threshold. Theoretical occupancy describes how many blocks or warps can reside from resource limits; it does not prove that the grid supplies those workers, that they are simultaneously active, or that their work is balanced.

Collect a synchronized kernel time and a time-resolved view where supported. Record the physical and application-constrained SM count, grid and cluster dimensions, blocks resident per SM, logical work count, work duration distribution, and active/idle intervals. Aggregate SM activity alone cannot distinguish a small grid, a partial final wave, variable-duration work, resource-limited residency, or phase-local serial work.

## Distinguish Four Cases

| Case | Discriminating evidence | Appropriate next experiment |
|---|---|---|
| Too little independent work | Logical block/cluster count is below available one-wave worker capacity | Change problem decomposition, accounting for reduction and synchronization cost |
| Wave quantization | Equal-duration data-parallel work has a small nonzero final-wave remainder | Compare tile shapes or a decomposition that changes tile count |
| Variable work duration | Per-worker tile counts/times have a long tail despite enough pending work | Compare static and dynamic acquisition with identical tile/decomposition |
| Residency or phase limitation | Grid is large, but resources or serial phases limit active blocks/warps | Change the limiting resource or phase; more grid blocks alone do not help |

Static assignment is nonadaptive, but it is not automatically imbalanced. A grid with fewer independent blocks than SMs cannot occupy every SM, while a grid much larger than the SM count can still show a tail or long-running stragglers. Do not prescribe `grid size >> SM count` without a decomposition that creates useful independent work.

## Scheduler Choices

Persistence, coordinate order, CLC reassignment, and K decomposition are separate controls:

- A static persistent worker can process multiple logical tiles by grid stride. It can amortize setup and change wave behavior but cannot guarantee removal of the final tail.
- A row/column raster or swizzle changes coordinate order and may change operand locality. It does not guarantee a better L2 hit rate or lower work variance.
- Cluster Launch Control lets a running worker cancel an unspecified not-yet-started block or cluster from the launched grid and process the returned ID. It redistributes existing work; it cannot create independent tiles, discard required work, or guarantee equal worker times.
- Stream-K or Split-K can create additional partitions when tile-level parallelism is insufficient, at the cost of partial-result reduction, workspace, synchronization, and possible determinism changes.

For CLC, compare static and dynamic scheduling with identical math, tile/cluster shapes, problem-sized grid, resource limits, and timing method. Record successful and failed requests plus per-worker work counts where instrumentation permits. Include ordinary and intentionally uneven SM availability as separate cases.

PTX ISA 9.0 says `clusterlaunchcontrol.try_cancel` requires `sm_100` or higher; its cluster-wide multicast qualifier explicitly lists `sm_120a` and the SM120 family. Therefore CLC is not categorically excluded from SM120 by the ISA. The pinned CUTLASS 4.5.0 persistent scheduler discussed in this wiki is specifically an SM100 integration; do not generalize a library route into an architecture exclusion.

## Source-Reported Tutorial Result

In Gau Nernst's `M=N=K=4096` B200 experiment, v5 reports 1302.29 TFLOP/s (86.43% of its 1506.74-TFLOP/s cuBLAS value), while v6 reports 1475.93 TFLOP/s (97.96%). The v6 endpoint combines static persistence, a changed output pipeline, and epilogue-specialized warps on top of earlier changes. The author says CLC was not added and observes that overlap remains imperfect. This is not a CLC result, an all-SMs-busy measurement, or an isolated tail-effect ablation.

## Verification Checklist

1. Prove exact logical-work coverage and no duplicates for initial and reassigned coordinates.
2. Compare requested work with available worker capacity and calculate the equal-duration final-wave remainder as a diagnostic baseline.
3. Instrument work count and time per worker to separate shortage from variable-duration imbalance.
4. Hold decomposition constant when testing static versus dynamic acquisition; test decomposition separately.
5. Report end-to-end time along with active-SM distribution, scheduler overhead, cache traffic, occupancy, and regressions.

## Primary References

- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [PTX ISA 9.0 CLC target and request semantics](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [CUTLASS 4.5.0 CLC scheduling](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/blackwell_cluster_launch_control.md)
- [Pinned tutorial v6](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v6.cu)
- [Tutorial result progression](https://gau-nernst.github.io/tcgen05/)
