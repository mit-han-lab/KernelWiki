---
id: pattern-tail-effect
title: "Tail Effect — Last-Wave Underutilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-low-sm-utilization]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, doc-cutlass-clc, doc-ptx-isa-sm100]
confidence: verified
evidence_basis:
  - source_id: doc-cutlass-clc
    evidence_type: official-doc
reproducibility: concept
---

## Exact Simplified Model

For `T` equal-duration independent tiles and `W` available one-tile workers, with no K decomposition, write:

```text
T = qW + r,  0 <= r < W
```

There are `q` full waves. If `r>0`, a final partial wave uses `r` workers and has instantaneous worker utilization `r/W`; if `r=0` and `T>0`, the final wave is full. For example, `T=150` and `W=142` gives one full wave plus eight tiles: the partial wave has `8/142 = 5.63%` worker utilization under these assumptions.

This is an analytical model, not a statement that a B200 has 142 SMs or that physical SM count always equals `W`. Block resources can make multiple blocks resident per SM, cluster shape changes scheduling granularity, application policies may restrict available SMs, and tile durations can differ. Measure the actual worker capacity and timeline.

Under the equal-duration model, the time fraction attributable to at most one partial wave shrinks as the number of full waves grows. There is no universal “below four times the SM count” cutoff: the remainder, wave duration, other phases, and imbalance determine significance.

## Confirm a Last-Wave Cause

1. Record logical tiles/clusters, grid and cluster dimensions, available worker capacity, residency limits, and per-work-item duration.
2. Predict wave count and the final remainder from the simplified model.
3. Use time-resolved activity or instrumented worker records to locate the underfilled interval at the end, rather than inferring it from aggregate utilization.
4. Sweep nearby problem sizes or tile shapes. A wave-quantization hypothesis predicts a sawtooth response aligned with changes in the remainder, after controlling total work.
5. Separate unequal tile durations: a long straggler is load imbalance even if the tile-count remainder is zero.

CUDA schedules ordinary grid blocks onto available SMs; a static grid-stride loop fixes logical worker indices, not `blockIdx`-to-physical-SM placement. A fixed grid therefore does not mean CUDA lacks dynamic block scheduling.

## Scheduler and Decomposition Choices

- A static persistent loop lets a resident CTA process multiple logical work items. Worker count is selected from problem decomposition, cluster/resource limits, and policy—not necessarily one CTA per physical SM. Finite work still has a final subset and may remain imbalanced.
- Cluster Launch Control lets a selected thread request cancellation of an unspecified not-yet-started block or cluster and process the returned grid ID. CLC redistributes existing IDs; it cannot turn eight remaining independent tiles into 142 or 148 concurrent tiles.
- Raster order and swizzle are coordinate transforms. Test them for locality while holding work acquisition fixed; they do not create work or guarantee balanced durations.
- Stream-K or Split-K may create more independent partitions when tile-level work is insufficient. Evaluate reduction traffic, workspace, synchronization, determinism, and numerical effects separately.

Compare static and CLC-backed persistence with identical math, decomposition, tile/cluster shapes, grid, resources, and timing. Record CLC request results and per-worker work/time distribution. PTX ISA 9.0 requires `sm_100` or higher for CLC and explicitly lists SM120-family support for the cluster-wide multicast qualifier; the pinned CUTLASS 4.5.0 route used here is specifically its SM100 scheduler.

## Tutorial Scope

Gau Nernst's Modal B200 has 148 SMs. In the author's `M=N=K=4096` experiment, v5 reports 1302.29 TFLOP/s (86.43% of cuBLAS) and v6 reports 1475.93 TFLOP/s (97.96%). The v6 change is static persistence plus output-pipeline/epilogue-role changes accumulated over earlier versions. The author explicitly did not add CLC and says overlap remains imperfect. Those results cannot be used as a CLC ablation or proof that all SMs stay busy.

## Primary References

- [CUDA Programming Guide: block and cluster scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [PTX ISA 9.0 CLC](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#clusterlaunchcontrol-try-cancel)
- [CUTLASS 4.5.0 CLC scheduling](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/blackwell_cluster_launch_control.md)
- [Pinned tutorial v6 source](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v6.cu)
- [Tutorial result progression](https://gau-nernst.github.io/tcgen05/)
