---
id: technique-ping-pong-scheduling
title: Ping-Pong Scheduling
type: technique
architectures:
- sm100
tags:
- ping-pong-scheduling
- warp-specialization
- tmem
- pipeline-stages
confidence: source-reported
reproducibility: snippet
prerequisites:
- hw-tmem
- technique-warp-specialization
related:
- kernel-flash-attention-4
- technique-double-buffering
sources:
- blog-flash-attention-4
- doc-flash-attention-4
- blog-tcgen05-tutorial
artifact_dir: artifacts/kernels/ping-pong-scheduling
---

# Ping-Pong Scheduling

## Overview

FA4 pipelines two score/output tiles so Tensor Core work on one tile can overlap softmax, correction, or data movement for another when dependencies permit. Blackwell's faster Tensor Cores make this overlap important because shared-memory and exponential throughput scale more slowly.

## Pattern

```python
# Dependency pseudocode, not CuTe DSL.
def pipeline(tile, steady_state):
    compute_score(tile[0])
    compute_score(tile[1])
    for stage in steady_state:
        advance_mma(stage.mma_tile)
        normalize(stage.softmax_tile)
        handoff_probabilities_via_tmem(stage.softmax_tile)
        correct_output_scale(stage.correction_tile)
        rotate_stage_after_all_handoffs(stage)
```

This is dependency pseudocode, not CuTe DSL or CUDA syntax. FA4 uses two 128-thread softmax warpgroups, a correction warpgroup, and a Tensor Core/TMA-driving warpgroup; the full implementation owns the exact TMEM partition and barrier protocol.

## Why It Helps on Blackwell

- B200's FP16/BF16 Tensor Core throughput is much higher than H100's, while exponential and shared-memory resources did not scale proportionally
- Multiple in-flight tiles expose independent MMA and non-MMA work for overlap
- Actual utilization remains dependency- and resource-limited; the paper does not claim both units are always 100% busy
- FA4's v1 paper reports up to 1613 TFLOP/s on B200 BF16/FP16 (approximately 71% of theoretical throughput) for the full implementation; that result is not an isolated measurement of this scheduling technique.

## When To Use

- Attention pipelines with at least two independent tiles and enough TMEM/register capacity for their live state
- Kernels where non-MMA work can overlap an asynchronous MMA without violating data dependencies
- The concrete FA4 schedule targets Blackwell; ping-pong/double-buffering as a general technique is not Blackwell-exclusive

## Full Reference Implementation

Verbatim upstream code lives in [`artifacts/kernels/ping-pong-scheduling/full/`](../../artifacts/kernels/ping-pong-scheduling/full/). Its SHA-256 and upstream-pinning metadata are in `PROVENANCE.yaml`. The former sequential teaching sketch did not demonstrate the overlap it claimed and was removed.

Query via:

```bash
python3 scripts/get_page.py technique-ping-pong-scheduling --include-code
```
