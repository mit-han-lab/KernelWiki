---
id: pattern-pipeline-stalls
title: "Pipeline Stalls"
type: pattern
tags: [pipeline-stages, warp-specialization, tma, tcgen05, mbarrier]
symptoms: [pipeline-stalls, compute-bound, low-tensor-core-utilization]
candidate_techniques: [technique-pipeline-stages, technique-warp-specialization, technique-double-buffering, technique-ping-pong-scheduling]
related: [pattern-compute-bound, pattern-tail-effect]
sources: [blog-tcgen05-tutorial, doc-flash-attention-4, doc-nvidia-tuning-guide, doc-ptx-isa-sm100]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
reproducibility: concept
---

## Symptom, Not Diagnosis

A pipeline stall is lost issue opportunity caused by a producer-consumer dependency on the critical path. Low TMA or tensor-core activity, a warp sampled at a barrier wait, or a phase-local utilization drop is only a lead. Expected dependency waits, memory latency, execution-pipeline contention, load imbalance, prologue/tail work, and synchronization defects can produce similar observations.

Nsight Compute's Warp State Statistics describes why sampled warps could not issue, while Scheduler Statistics shows whether schedulers had eligible warps and issued instructions. NVIDIA cautions that stalls are not necessarily performance-limiting and should be prioritized when schedulers fail to issue. Metric availability and names vary by tool and chip, so query the installed profiler and use its shipped sections rather than assuming a Blackwell metric name. PM sampling can add a timeline on supported systems, but warp sampling itself has no time resolution.

## Correctness Gate: Audit the Wait Edge

Do this before tuning around an mbarrier wait:

1. Identify the exact barrier object, owner, initialized arrival count, initial phase, and storage stage for the wait.
2. For a TMA global-to-shared load, account for the producer's software arrival and expected transaction bytes. `mbarrier.arrive.expect_tx` performs an arrival and adds transaction bytes; the TMA `.mbarrier::complete_tx::bytes` operation decrements transaction bytes, not a second arrival.
3. Wait for the matching phase of that same object. Phase parity changes when that barrier completes a phase and is reused; “flip after every wait” is not a valid global rule when a thread waits on multiple objects or phases.
4. A successful acquire wait supplies the documented visibility for associated prior `cp.async.bulk` work before the consumer reads SMEM.
5. Before reusing SMEM operands read by asynchronous MMA, attach completion of the relevant tcgen05 work with `tcgen05.commit` and observe its mbarrier. Issue is not completion.
6. When asynchronous tcgen05 operations cross a thread handoff, place `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` around the applicable execution-ordering operation. The after fence is not a generic replacement for the TMA acquire wait.
7. Verify prologue, steady-state wraparound, short loops, and the producer/consumer tails. A correct steady-state loop can still hang or reuse live storage at a boundary.

An extra software arrival is erroneous only relative to the barrier's initialized and phase-specific accounting. Diagnose pending arrivals and transaction bytes separately instead of assuming that TMA hardware performs another arrival.

## Causal Diagnosis Workflow

1. Establish a synchronized end-to-end timing regression on fixed inputs and confirm outputs against a reference. Record GPU, clocks, toolkit, build flags, kernel name, launch shape, and resource use.
2. Collect Speed-of-Light, Scheduler Statistics, Warp State Statistics, and source/SASS correlation with the sections available in the installed Nsight Compute. Account for replay and sampling effects.
3. Map the dominant not-issued locations to concrete edges: TMA-full wait, MMA-completion wait, output-buffer reuse, CTA barrier, queue starvation, or pipeline prologue/tail.
4. State a prediction. For example: if TMA readiness is the critical edge, a legal extra operand stage should reduce that wait and runtime; if only the tail is exposed, the steady-state wait distribution should remain largely unchanged.
5. Change one variable and repeat identical correctness, warmup, timing, and profiler collection. Reject a cause when its predicted counter and time movement do not occur.

| Controlled change | Plausible target | Required controls and costs |
|---|---|---|
| Vary legal SMEM stage count | Producer cannot stay far enough ahead of MMA | Same tile/math/roles; record SMEM, occupancy, short-loop behavior, and tail |
| Separate long-lived warp roles | Role handoff or control work is on the issue path | Same stages and tile; record registers, active warps, and per-role idle/backpressure |
| Add a second TMEM output region | Epilogue holds the only accumulator region | Prove disjoint lifetime; record TMEM columns and epilogue/MMA completion waits |
| Interleave two query/output tiles | Softmax/correction and MMA have independent ready work | Prove separate state and dependencies; compare one versus two query stages |

None is a universal cure. More stages consume storage and can reduce occupancy; specialization adds warps and synchronization; TMEM buffering consumes columns; multi-tile schedules increase live state. A memory-bound roofline classification also does not prove pipelining is useless: test it only when the proposed change has a specific issue-gap, latency, or transaction-efficiency prediction.

## Source-Reported Tutorial Progression

Gau Nernst reports the following for one `M=N=K=4096` BF16 GEMM on a Modal B200 using PyTorch 2.9.1 and CUDA 13. Percentages below are computed from the reported 1506.74-TFLOP/s cuBLAS value.

| Version | Author's cumulative version label | TFLOP/s | cuBLAS ratio |
|---|---|---:|---:|
| v1a | basic tcgen05 + 2D 16-byte TMA | 254.62 | 16.90% |
| v2b | 3D 128-byte TMA | 695.43 | 46.15% |
| v3 | pipelining | 939.61 | 62.36% |
| v4 | warp specialization | 1208.83 | 80.23% |
| v5 | 2-SM MMA | 1302.29 | 86.43% |
| v6 | persistent kernel with static scheduling | 1475.93 | 97.96% |

The pinned v3 source instantiates two stages, not three. Each row is a cumulative source version rather than an isolated microbenchmark of the named mechanism. The final version uses static scheduling; the author explicitly says Cluster Launch Control was not added. The result is evidence that these changes helped that implementation and shape, not a Blackwell progression template.

## Primary References

- [Nsight Compute 2025.3 Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
- [PTX ISA 9.0 mbarrier waits and visibility](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
- [PTX ISA 9.0 tcgen05 execution ordering](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-special-sync-operations-fence)
- [Pinned tutorial v3 source](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v3.cu)
- [Tutorial progression](https://gau-nernst.github.io/tcgen05/)
