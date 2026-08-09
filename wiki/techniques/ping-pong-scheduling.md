---
id: technique-ping-pong-scheduling
title: Ping-Pong Scheduling
type: technique
architectures: [sm100]
tags: [ping-pong-scheduling, warp-specialization, tmem, pipeline-stages]
confidence: verified
evidence_basis:
  - source_id: doc-flash-attention-4
    evidence_type: official-doc
reproducibility: concept
prerequisites: [hw-tmem, technique-warp-specialization]
related: [kernel-flash-attention-4, technique-double-buffering]
sources: [doc-flash-attention-4, blog-flash-attention-4]
blackwell_relevance: "The pinned FlashAttention-4 SM100 forward kernel uses two query/output-tile states to overlap MMA for one tile with softmax for the other; the benefit and remaining stalls are configuration-dependent."
---

## Definition

Ping-pong scheduling interleaves two output-tile states in one CTA. In the pinned FlashAttention-4 (FA4) SM100 forward implementation, `q_stage=2` makes the CTA cover two 128-row query tiles. One dedicated warp controls MMA, two four-warp groups perform softmax for the respective query stages, and a separate four-warp group performs correction work. While MMA advances one tile, the schedule tries to overlap softmax work for the other tile.

This is not merely ordinary double buffering of one producer-consumer stream. Each query stage owns a distinct output state, including its softmax and output TMEM regions, and the algorithm must preserve the dependencies among MMA accumulation, row statistics, rescaling/correction, and output consumption. “Ping-pong” describes this interleaving; it does not guarantee that either execution resource is stall-free.

## Version-Pinned Implementation Contract

The concrete reference is [`flash_fwd_sm100.py` at commit `a369df707e1980fb328abcc1733e3457ec10155f`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_fwd_sm100.py). Its default two-stage layout has the following software roles:

| Warp IDs | Role |
|---|---|
| 0–3 | softmax for query stage 0 |
| 4–7 | softmax for query stage 1 |
| 8–11 | correction |
| 12 | MMA control |
| 13 | epilogue |
| 14 | load |
| 15 | empty in this configuration |

These IDs are implementation choices, not SM100 architectural roles. The source changes assignments for other configurations, including `q_stage=1`.

The implementation performs one collective TMEM allocation sized for its required columns and gives the two query stages disjoint offsets within that allocation. It also constructs multiple producer/consumer pipelines for distinct handoffs. A single generic barrier after issuing `tcgen05.mma` is not an equivalent implementation: tcgen05 work is asynchronous, and completion, memory visibility, stage reuse, and TMEM lifetime must follow the corresponding pipeline and ISA contracts.

## Correctness Checklist

Before treating a two-tile schedule as valid, verify all of the following in the complete kernel:

1. TMEM regions for simultaneous tile states are disjoint, allocated and deallocated collectively, and remain live until their final consumers finish.
2. Every SMEM or TMEM stage has one unambiguous producer/consumer phase owner; a stage cannot be overwritten until all operations that read it have completed.
3. Asynchronous MMA completion is connected to the barrier that guards accumulator consumption and operand-stage reuse. A CTA barrier or an unrelated mbarrier arrival is insufficient.
4. Softmax groups obey the implementation's explicit synchronization. FA4 serializes their exponential critical sections rather than allowing both groups to contend there simultaneously.
5. Correction and epilogue work wait for the row statistics and accumulators they consume, including the proper proxy and execution-order fences.
6. Boundary query rows, final key tiles, causal/window masks, and the pipeline tail preserve the same mathematical result as the reference attention computation.

## Hardware Motivation and Evidence Scope

FA4's paper models B200 BF16 tensor-core throughput at 8192 operations per clock per SM versus 4096 on Hopper, while exponential throughput is 16 operations per clock per SM on both. For its simplified `M=N=d=128` model, the paper assigns 1024 cycles each to MMA and exponentials and 768 cycles to shared-memory traffic; doubling the query extent doubles each of those modeled counts. The model explains why overlapping independent tile work is attractive, but the paper says that this roofline omits other resources.

The paper reports up to 1613 TFLOP/s and 71% utilization for complete FA4 forward kernels. The accompanying author blog reports up to 1605 TFLOP/s and 71%. Those endpoints include partial software exponentials, conditional rescaling, TMEM partitioning, register allocation choices, and the rest of the pipeline. Neither source supplies a ping-pong-only ablation, and neither establishes 100% tensor-core or exponential-unit utilization.

## When to Test It

Use the pinned SM100 kernel as the concrete reference when two independent query/output states can coexist within the register, shared-memory, and TMEM budgets. Compare `q_stage=2` with the source-supported `q_stage=1` configuration while holding the remaining build, input, tile, and launch choices fixed. Measure end-to-end time, MMA and exponential-pipeline activity, barrier stalls, per-role idle time, register spills, and occupancy across representative shapes.

Do not select the technique from a generic “compute-bound” or “SFU-heavy” label alone. The FA4 paper says its schedule is similar to FlashAttention-3's Hopper ping-pong schedule, so the scheduling idea is not intrinsically Blackwell-only; this page's pinned implementation and hardware figures are specifically SM100. Keep the two-stage form only when the controlled comparison shows a relevant improvement and correctness tests cover boundaries and pipeline tails.
