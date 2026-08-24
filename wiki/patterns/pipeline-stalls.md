---
id: pattern-pipeline-stalls
title: "Pipeline Stalls"
type: pattern
tags: [pipeline-stages, warp-specialization, tma, tcgen05, mbarrier]
symptoms: [pipeline-stalls, compute-bound, low-tensor-core-utilization]
candidate_techniques: [technique-pipeline-stages, technique-warp-specialization, technique-double-buffering, technique-ping-pong-scheduling]
related: [pattern-compute-bound, pattern-tail-effect]
sources: [blog-tcgen05-tutorial, blog-flash-attention-4, doc-nvidia-tuning-guide]
---

# Pipeline Stalls

## Symptom

Nsight Compute shows TMA or tcgen05 units idle despite nominally compute-bound workload. Tensor core utilization drops during specific phases of the kernel. Warp-level profiling reveals threads blocked on `mbarrier.try_wait` more than expected.

## Likely Causes

1. **Too little in-flight work**: the measured load or compute latency is
   longer than the work available in the other stages.
2. **Incorrect barrier phase or participant accounting**: a consumer waits on
   the wrong phase, or a producer reuses a stage before every consumer releases
   it.
3. **Incomplete operation-specific ordering**: TMA, tcgen05, and ordinary
   shared-memory accesses have distinct completion and proxy rules. A generic
   fence recipe is not valid for every combination.
4. **Role imbalance**: one producer, MMA, softmax, or epilogue role takes longer
   than the roles it is meant to overlap.
5. **Memory-system or tail backpressure**: nominal stage depth cannot compensate
   for bandwidth saturation, cache misses, or too little remaining grid work.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Pipeline stages](../techniques/pipeline-stages.md) | Sweep stage depth while measuring stalls and resource use |
| [Warp specialization](../techniques/warp-specialization.md) | Separate roles when their measured overlap justifies the synchronization cost |
| [Double-buffering](../techniques/double-buffering.md) | Give producer and consumer distinct live stages |
| [Ping-pong scheduling](../techniques/ping-pong-scheduling.md) | Two query tiles alternate softmax/MMA (FA4 pattern) |

## Diagnosis Checklist

```
1. Profile the actual kernel and identify which role is waiting.
2. Reconstruct producer/consumer stage ownership and barrier phases.
3. Verify expected-arrival counts and transaction bytes against the exact API
   used by the implementation.
4. Check the PTX ISA or pinned library abstraction for the operation-specific
   completion, proxy, and fence sequence.
5. Sweep stage depth and role allocation while watching shared memory,
   registers, occupancy, and total runtime.
```

## Example Progression (tcgen05 tutorial)

- v3 pipelining: 939.61 TFLOPS (62.4% of the tutorial's cuBLAS result)
- v4 warp specialization: 1208.83 TFLOPS (80.2%)
- v5 2-SM MMA: 1302.29 TFLOPS (86.4%)
- v6 persistence with static scheduling: 1475.93 TFLOPS (98.0%)

## Caveats

- Additional stages consume shared memory and sometimes registers; derive the
  legal budget from the compiled kernel and selected device.
- Phase bugs can be intermittent; instrument stage ownership and phases during
  development.
- A lower wait counter does not by itself mean lower end-to-end runtime.
