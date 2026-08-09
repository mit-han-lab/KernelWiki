---
id: technique-warp-specialization
title: Warp Specialization on Blackwell
type: technique
architectures: [sm100, sm90]
tags: [warp-specialization, tcgen05, tmem]
confidence: source-reported
reproducibility: concept
prerequisites: [hw-tmem, hw-tcgen05-mma]
related: [technique-persistent-kernels, technique-pipeline-stages, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, doc-cutlass-blackwell, blog-tcgen05-tutorial, pr-flashinfer-1039]
blackwell_relevance: "Single-thread tcgen05 issue and TMEM accumulators enable new role partitions, but warp counts and role IDs are software-configuration choices."
artifact_dir: artifacts/kernels/warp-specialization
---

## Definition and ISA Boundary

Warp specialization assigns different long-lived functions to disjoint warps in a CTA—for example scheduler, TMA load, MMA control, softmax/correction, or epilogue work. It is a software organization for overlapping pipeline stages, not a fixed SM100 CTA layout.

The architecture-level comparison is narrower:

| Property | Hopper `wgmma.mma_async` | Blackwell `tcgen05.mma` |
|---|---|---|
| Issue granularity | Warpgroup collective | One thread for `cta_group::1` or `cta_group::2` |
| D accumulator | Per-thread registers | TMEM |
| A/B movement | Explicit register/SMEM operands and producer work | Explicit SMEM descriptors or TMEM addresses and producer work |
| Completion | WGMMA commit/wait groups | `tcgen05.commit` tied to an mbarrier, followed by a wait |

Single-thread issue reduces the number of threads needed to submit MMA instructions. It does not make operands move automatically, turn an entire warp into an ISA-level role, complete MMA synchronously, or choose how many epilogue warps a kernel should launch.

## Version-Pinned Role Maps

There is no universal “warp 0 load, warp 1 MMA, warps 2–15 epilogue” rule. Three primary implementations illustrate the range:

| Implementation | Total warps and roles |
|---|---|
| Gau Nernst tutorial v4, commit `3b90ac9b...` | 4 warps. An elected lane in warp 0 runs the TMA loop; an elected lane in warp 1 runs the MMA loop; after completion, all four warps participate in TMEM load/conversion/output. |
| CUTLASS 4.5.0 generic SM100 GEMM | One warp each for `MMA`, `Sched`, `MainloopLoad`, and `EpilogueLoad`, followed by `CollectiveEpilogue::ThreadCount / 32` epilogue warps. Optional work can make some control roles nonparticipants. |
| FlashInfer PR 1039 context FMHA | 16 warps: 0–3 `Softmax0`, 4–7 `Softmax1`, 8–11 correction, 12 MMA, 13 load, 14 epilogue, and 15 empty. The kernel has explicit pipelines between these roles. |

FlashAttention-4's pinned SM100 forward source likewise starts from a 16-warp layout with two four-warp softmax groups, four correction warps, and dedicated MMA/load/epilogue IDs, then adjusts roles for configuration choices such as one Q stage, non-TMA paths, and dynamic persistence. These are concrete attention schedules, not a GEMM-wide architectural template.

## Correct Synchronization Obligations

A warp-specialized implementation must prove each ownership handoff. For a reusable TMA-to-MMA stage:

1. Initialize the mbarrier objects with correct participant/transaction counts and publish them to the required threads and async proxies before use.
2. The producer acquires an empty stage, sets expected transaction bytes, and submits the TMA copies. The full/data-ready phase completes only after the expected async transactions complete.
3. The MMA controller waits for the full phase and applies the required cross-thread/proxy fence before issuing `tcgen05.mma` against that SMEM stage.
4. Because MMA is asynchronous, `__syncwarp()` or an ordinary `mbarrier.arrive` after issue does not make the operands reusable. Commit prior tcgen05 work to an mbarrier and wait for completion before releasing or overwriting the stage.
5. Before another role reads D from TMEM, complete the MMA sequence and apply the required `tcgen05.fence::before_thread_sync` / execution-ordering handoff / `tcgen05.fence::after_thread_sync` protocol. A fence orders operations; it is not a completion wait.
6. Keep TMEM allocated until every consumer finishes its `tcgen05.ld` sequence and associated waits, then deallocate with the required collective participation.

Pipeline wrappers in CUTLASS/CuTe encode parts of these rules, but participant counts, initial phases, tails, and producer/consumer states remain configuration-specific. A short role-dispatch sketch is not a substitute for the complete pipeline and TMEM lifecycle.

## Source-Reported Tutorial Result

For the author's exact `M=N=K=4096` Modal B200 setup with PyTorch 2.9.1 and CUDA 13, tutorial v3 reports 939.61 TFLOP/s and v4 reports 1208.83 TFLOP/s after introducing warp specialization. That is a 269.22-TFLOP/s, approximately 28.65% step between those source variants. It is not a portable Blackwell speedup or evidence that specialization is always profitable.

## When to Test Warp Specialization

Use it as a candidate when independent pipeline roles have enough steady-state work to overlap and their register needs differ materially. Compare it with a temporally pipelined version at the same tile shape, stage count, CTA-group mode, and scheduler. Measure:

- time and achieved throughput across representative shapes;
- per-role active/stall time and pipeline backpressure;
- register allocation, spills, shared memory, and occupancy;
- TMA/MMA/epilogue balance, including prologue and tail cost;
- 1-SM versus 2-SM mode as an independent legal-shape/resource choice.

More role warps are not automatically useful for a more complex epilogue, and one MMA-control warp does not limit the issuer to one outstanding asynchronous MMA operation. Choose participant counts from the concrete collective and measurements.

## Local Evidence Bundle

[`artifacts/kernels/warp-specialization/full/`](../../artifacts/kernels/warp-specialization/full/) contains the captured FlashInfer PR 1039 mainloop file; its bytes match the SHA-256 recorded in `PROVENANCE.yaml` and the duplicate captured PR key-file. The provenance declares upstream revision `9a05c92a`. That abbreviated historical object was not independently refetchable during this audit, so the bundle is locally hash-verified captured evidence, not a fresh upstream retrieval.

[`artifacts/kernels/warp-specialization/variants/`](../../artifacts/kernels/warp-specialization/variants/) is explicitly `derived` teaching material and must not be cited as upstream code or as a complete safe kernel.

Retrieve the page plus both artifact modes with:

```bash
conda run -n base python scripts/get_page.py technique-warp-specialization --include-code
```
