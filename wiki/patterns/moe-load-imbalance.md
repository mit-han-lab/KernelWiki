---
id: pattern-moe-load-imbalance
title: "MoE Expert Load Imbalance"
type: pattern
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm, doc-cutlass-clc, blog-gpu-mode-reward-hack]
---

# MoE Expert Load Imbalance

## Diagnose the level that is imbalanced

Do not infer one expert per physical SM from a grouped-GEMM trace. CUDA assigns
thread blocks to available SMs, while grouped kernels may split an expert into
multiple output tiles or let one resident worker process several logical tiles.
Measure at each relevant level:

- routed tokens per expert for the exact batch and prefill/decode phase;
- valid rows and output-tile counts per expert;
- completed tiles and elapsed work per logical worker and, when instrumented,
  per SM;
- dispatch, grouped-GEMM, combine, and end-to-end times per expert-parallel
  rank.

Small expert segments and partial output tiles can expose too little parallel
work or leave lanes predicated out. The effect depends on the kernel tile,
other GEMM dimensions, resident-worker count, and competing implementation;
there is no universal `M < BLOCK_M` diagnosis or minimum viable size.

Uniform expected routing, a larger batch, or an auxiliary balancing loss may
change token-count skew, but none proves balanced tile counts, worker
durations, communication, or runtime. Treat imbalance as absent only when the
measured distributions are narrow and a matched balancing variant does not
materially improve the timed operation.

## Grouped-layout choices

At pinned DeepGEMM commit
[`891d57b4`](https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f),
the three grouped interfaces have different axes and metadata:

| Layout | Pinned contract | Relevant control |
| --- | --- | --- |
| M-grouped contiguous | Pack `A` and `D` along variable M with N/K fixed; identify groups by per-row expert IDs or per-group prefix-sum ends; segments are M-block aligned | Record valid rows, alignment padding, tile counts, and metadata form |
| M-grouped masked | Allocate `[G, M_max, ...]`, pass one valid-M count per group, and compute valid portions; the fixed allocation is documented for a CUDA-graph decode case | Separate allocated rows from valid rows and inspect edge predication in generated code |
| K-grouped contiguous | Pack variable K with M/N fixed for MoE weight backward | Do not use its K-axis contract to describe forward token imbalance |

A fixed `M_max` allocation therefore does not establish that all padding rows
are computed. Any residual edge-tile or predication cost needs generated-code
and profile evidence for the selected kernel.

## Scheduler comparisons

Keep persistence, work acquisition, and decomposition as separate variables:

1. A static persistent worker can advance through a deterministic logical-work
   sequence. CUDA still decides where its block runs; the rule is not a
   precomputed tile-to-physical-SM map.
2. In a CLC-backed scheduler, a selected thread requests cancellation of an
   unspecified block or cluster that has not launched. A successful response
   returns that existing grid coordinate; a request can fail. CLC does not
   create tiles or make the fastest SM autonomously steal arbitrary work.
3. The initial `blockIdx` in CUTLASS's pinned CLC scheduler is static and later
   requests are asynchronous, pipelined, and cluster-granular. There is no
   universal per-tile latency bound: compare request activity and end-to-end
   time against the matched static scheduler.
4. Splitting a large expert or K range may expose more independent work, but it
   can add partial-result and reduction costs. Hold math, tile shape, cluster
   shape, launch resources, and output semantics constant in the comparison.

PTX ISA 9.0 specifies CLC for `sm_100` or higher and explicitly includes
SM120-family targets for the multicast form. CUTLASS 4.5.0's documented
`PersistentTileSchedulerSm100` integration is a narrower library example, not
the complete ISA compatibility boundary.

## Expert placement is a separate system layer

[DeepSeek EPLB at commit
`d52c72d`](https://github.com/deepseek-ai/EPLB/tree/d52c72d5b2f2fb4c41afbf8eb21366820239913d)
takes per-logical-expert load statistics, replicates heavily loaded logical
experts, and packs physical experts across configured GPUs and nodes. It is a
host-side expert-parallel placement planner, not a device-kernel tile
scheduler; use it as a complement only when multi-GPU expert placement is in
the measured system scope.

An LMSYS/SGLang study on a 96-H100 deployment reported `1.49x` prefill and
`2.54x` decode throughput speedups in its large-scale EPLB ablation. The same
study says it used in-distribution data and that production distribution
shifts require further testing. Those figures are scoped results, not generic
EPLB speedups.

## GPU Mode Problem 4 reward-hack boundary

GPU Mode's official postmortem records a temporarily first-place submission
that combined a real grouped-GEMM kernel with a timing-harness exploit. The
correctness path ran a padded eight-group computation on each of 15 cloned
objects. During timing, the first call merged them into one 120-group launch,
calls 2 through 15 returned cached output pointers, and the harness divided the
combined timing by 15.

This is evidence about evaluator state reuse, not a valid load-balancing
performance result. The postmortem identifies `gpu-mode/reference-kernels`
PR 104 as the harness response; neither that record nor the pinned FlashInfer
MLSys 2026 evaluation contract establishes a causal link between this incident
and the later contest design.

## Primary references

- [CUDA thread-block scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [PTX ISA 9.0 CLC request semantics](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
- [CUTLASS 4.5.0 CLC scheduler](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/blackwell_cluster_launch_control.md)
- [DeepGEMM grouped interfaces at `891d57b4`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/README.md)
- [DeepSeek EPLB at `d52c72d`](https://github.com/deepseek-ai/EPLB/tree/d52c72d5b2f2fb4c41afbf8eb21366820239913d)
- [LMSYS/SGLang 96-H100 deployment study](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- [GPU Mode reward-hack postmortem](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [Pinned FlashInfer MLSys 2026 evaluator](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/75ccd05cafceb0fd1f86be4cd0f2117249463c66/EVALUATION.md)
