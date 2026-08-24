---
id: technique-warp-specialization
title: Warp Specialization on Hopper and Blackwell
type: technique
architectures: [sm100, sm90]
tags: [warp-specialization, tcgen05, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem, hw-tcgen05-mma]
related: [technique-persistent-kernels, technique-pipeline-stages, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, pr-flashinfer-1039, blog-colfax-cutlass]
blackwell_relevance: "Single-thread tcgen05 MMA issue permits role splits that differ from Hopper's warpgroup-collective WGMMA, but thread count and role assignment remain kernel choices."
artifact_dir: artifacts/kernels/warp-specialization
---

# Warp specialization

Warp specialization assigns different warps to producer, matrix, scheduling, softmax, or epilogue work so that independent pipeline phases can overlap. The number of warps and their roles are kernel configuration choices; Blackwell has no architectural “canonical 16-warp CTA.”

On Hopper, a WGMMA instruction is issued collectively by a four-warp warpgroup. On SM100, a `tcgen05.mma` operation is initiated by one thread, while TMEM allocation and TMEM load/store operations remain warp-collective. A practical role split therefore depends on all of the selected operations—not just MMA issue granularity.

## Source-backed pipeline fragment

This contiguous excerpt is from FlashInfer PR 1039’s captured `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`. It shows ownership transfer around one QK MMA step; it does not imply a universal warp count.

```cpp
q_index = pipeline_q_consumer_state.index();
pipeline_q.consumer_wait(pipeline_q_consumer_state);
++pipeline_q_consumer_state;
pipeline_s0.producer_acquire(pipeline_s0_producer_state);
gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);
pipeline_s0.producer_commit(pipeline_s0_producer_state);
```

## Design checks

- Name one owner for each pipeline state transition and prove that acquire/wait pairs cannot deadlock.
- Keep `.sync.aligned` instructions on converged, fully active warps.
- Budget registers, shared memory, TMEM columns, and resident CTAs for the actual role split.
- Measure overlap. Extra roles and barriers can cost more than they save when phases are short or imbalanced.
- Treat a concrete warp count as part of the cited kernel configuration, not as a property of SM100.
