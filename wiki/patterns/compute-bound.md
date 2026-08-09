---
id: pattern-compute-bound
title: "Not Reaching the Relevant Compute Ceiling"
type: pattern
tags: [tcgen05, 2sm-cooperative, pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [hw-2sm-cooperative, technique-pipeline-stages, technique-warp-specialization, technique-epilogue-fusion, technique-software-exp]
related: [pattern-pipeline-stalls, pattern-low-sm-utilization, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, doc-flash-attention-4, blog-flash-attention-4]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
reproducibility: concept
---

## Classify the Gap First

“Below peak FLOPS” is not itself a bottleneck diagnosis. Select the peak for the executed datatype, instruction kind, sparsity/scaling mode, clocks, and number of participating SMs. Then use arithmetic intensity and achieved compute/memory ceilings to decide whether the measured kernel lies on the compute side of the relevant roofline. Unsaturated DRAM bandwidth and tensor-core utilization below an arbitrary threshold such as 70% do not prove compute boundedness.

Also distinguish whole-kernel throughput from tensor-core active time. A correct kernel can have efficient MMA intervals yet spend material time in TMA readiness, CUDA-core transforms, reductions, synchronization, epilogue work, or the grid tail. Scheduler under-issue plus source/SASS correlation identifies where issue opportunity is lost; a high stall sample count alone does not establish the cause.

## Separate the Limiting Edge

| Observation and matched control | Supported inference if runtime also improves |
|---|---|
| An extra legal SMEM operand stage reduces TMA-full waits | Operand production was exposed on the tested K loop |
| Separating producer/MMA roles reduces control or dependency gaps | The prior role schedule was on the critical issue path |
| A second TMEM output region reduces mainloop-to-epilogue waits | Accumulator reuse was serialized by the epilogue |
| Moving only selected non-MMA operations changes their pipeline activity and runtime | Those operations contributed to the tested critical path |
| A legal 2-SM variant changes traffic/reuse and aggregate throughput | Cooperative mapping helped that shape and resource configuration |

For every comparison, hold math, datatype, tile coverage, launch environment, warmup, and timing statistic fixed. Record registers, spills, SMEM, TMEM columns, occupancy, cluster size, achieved bandwidth, scheduler issue activity, and per-pipeline activity. A change in utilization without lower end-to-end time is not a win.

## Candidate-Specific Checks

### Pipeline stages

Enumerate stage counts that compile and fit the complete shared-storage allocation. More stages can overlap TMA production with MMA consumption, but they also consume SMEM, change occupancy, and enlarge prologue/tail costs. There is no architecture-wide “three to five stages” range; the cited tutorial's beneficial v3 kernel uses two stages.

### Warp specialization and output overlap

Dedicated producer, MMA, or epilogue warps can remove role switching from a loop and allow independent work to progress, but add live warps, registers, and synchronization. They do not eliminate stalls. If the epilogue owns the only output region, compare one versus multiple disjoint TMEM regions and prove MMA-completion and epilogue-load completion before each handoff.

### 2-SM cooperative MMA

`cta_group::2` requires a valid cluster and exact kind-, shape-, layout-, descriptor-, and peer-resource constraints. It does not universally mean `m256n256`, require identical SMEM layouts, or promise twice the compute per cycle: one cooperative operation consumes resources from two CTAs. Compare total and per-SM throughput against a legal one-CTA mapping and measure whether peer operand reuse or a larger aggregate tile reduces traffic enough to repay coordination and occupancy costs.

### Non-MMA work

Optimize non-MMA work only after showing that it lies on the critical path. Fusion may remove intermediate traffic; tile interleaving may expose independent work; an approximation may exchange numerical error for throughput. Validate error against the application oracle rather than assuming a software transcendental is interchangeable with the hardware operation.

## FlashAttention-4 Case Study

FA4's B200 analysis reports 8192 BF16 MMA operations per clock per SM versus 4096 on Hopper, while exponential throughput is 16 operations per clock per SM on both. Its response is a coordinated design: two-output-tile scheduling, selected software exponentials, conditional rescaling, TMEM partitioning, and pipeline/register choices.

The software path uses base-2 range reduction with `n=floor(x)` and a cubic FMA polynomial for a selected fraction of exponential evaluations; other values still use hardware `ex2`. The paper evaluates approximation error and end-to-end accuracy. This is not a claim that every exponential uses Cody-Waite reduction or that polynomial evaluation universally multiplies exponential throughput.

The paper reports up to 1613 TFLOP/s and 71% for complete FA4 forward kernels; the author blog reports up to 1605 TFLOP/s and 71%. Neither number isolates software exponentiation or any other single technique. Use FA4 as evidence that non-MMA work can become material after tensor-core throughput increases, not as a recipe for an unrelated compute-bound kernel.

## Primary References

- [Nsight Compute 2025.3 roofline and profiling guidance](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
- [PTX ISA 9.0 tcgen05 MMA forms](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma)
- [Pinned two-stage tutorial pipeline](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/matmul_v3.cu)
- [FlashAttention-4 paper v1](https://arxiv.org/html/2603.05451v1)
- [FlashAttention-4 author blog](https://tridao.me/blog/2026/flash4/)
