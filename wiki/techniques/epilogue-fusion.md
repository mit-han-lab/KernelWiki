---
id: technique-epilogue-fusion
title: Epilogue Fusion
type: technique
architectures: [sm100, sm90]
tags: [epilogue-fusion, tmem, warp-specialization]
confidence: verified
evidence_basis:
  - {source_id: doc-cutlass-blackwell, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
reproducibility: snippet
prerequisites: [hw-tmem, technique-warp-specialization]
related: [technique-warp-specialization, hw-tmem, technique-double-buffering]
sources: [doc-cutlass-blackwell, doc-ptx-isa-sm100, blog-colfax-cutlass, pr-cutlass-2139, pr-vllm-16032]
blackwell_relevance: "SM100 epilogues read accumulators from TMEM and can fuse conversion, scaling, bias, activation, or quantization before the output store."
artifact_dir: artifacts/kernels/epilogue-fusion
---

# Epilogue Fusion

## Overview

An epilogue converts a completed accumulator tile into output values. Fusing scaling, bias, activation, residual addition, or quantization into that path can avoid materializing an intermediate tensor and launching another kernel.

On SM100, MMA destinations reside in TMEM. After MMA completion is established, participating threads load assigned TMEM fragments into registers, apply the output operation, and store directly or stage through shared memory for a TMA store. Some schedules pipeline multiple accumulator stages so an epilogue for one work unit overlaps other work, but overlap and TMEM partitioning are schedule choices—not guarantees of fusion itself.

## Role mapping is configuration-dependent

CUTLASS's current SM100 TMA warp-specialized GEMM kernel assigns one warp each to MMA, scheduling, mainloop load, and epilogue load; epilogue warps begin at warp index 4, and their count is `CollectiveEpilogue::ThreadCount / 32`. Therefore, “warps 2–15” or “14 epilogue warps” is not a general SM100 mapping.

The epilogue policy, tile shape, datatype, and whether a separate epilogue-load producer is needed determine the thread count and synchronization graph.

## Schematic lifecycle

```python
def fused_epilogue_pipeline(work_units, accumulator_stages):
    for work in work_units:
        stage = acquire_empty_accumulator_stage(accumulator_stages)
        issue_mma(work, stage)
        commit_mma_completion(stage)

        ready = wait_for_mma_completion(stage)
        fragment = load_assigned_tmem_fragment(ready)
        output = activation(scale(fragment) + load_bias(work))
        store_output(work, output)
        release_accumulator_stage(stage)
```

This is dependency pseudocode, not CUDA syntax. The implementation must use the exact `tcgen05.commit`/`mbarrier` completion mechanism and TMEM load forms required by its PTX target. `tcgen05.fence` orders accesses around thread synchronization; it does not by itself wait for an asynchronous MMA to finish.

## When fusion helps

Fusion is most attractive when the unfused path would write and reread a large intermediate or pay another launch. It can lose when the fused output operation increases registers, shared memory, barriers, or code size enough to reduce occupancy or throughput. Quantify the full kernel, including tails and small shapes.

## Correctness checklist

- Wait for MMA completion before reading its TMEM stage.
- Do not reuse a stage until all epilogue consumers have released it.
- Map each TMEM lane/column to the intended logical output coordinate.
- Preserve the requested accumulation, rounding, saturation, and activation semantics.
- Handle output bounds and residual loads independently of the fast-path alignment.

## Full Reference Implementation

Verbatim upstream code lives in [`artifacts/kernels/epilogue-fusion/full/`](../../artifacts/kernels/epilogue-fusion/full/). The unvalidated pseudo-API teaching variant and blog-extracted sketch were removed.
