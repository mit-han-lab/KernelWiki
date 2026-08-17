---
id: pattern-moe-load-imbalance
title: "MoE Expert Load Imbalance"
type: pattern
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm, blog-gpu-mode-reward-hack]
---

# MoE Expert Load Imbalance

## Symptom

Expert token counts or per-tile durations differ enough that some workers finish while others remain active. End-to-end latency may instead be dominated by dispatch/combine communication, so separate routing, communication, GEMM, and tail intervals.

## Likely causes

- Skewed router assignments.
- Small or irregular per-expert M dimensions.
- Static work partitioning that cannot adapt to cost variance.
- Padding/fixed-capacity layouts or graph-capture constraints.
- Cross-device expert placement and communication imbalance.

## Candidate techniques

| Technique | Boundary |
|---|---|
| CLC/persistent scheduler | Can reassign unlaunched cluster IDs; does not preempt running work |
| Contiguous grouped layout | Packs variable-M expert segments, subject to alignment constraints |
| Masked layout | Supports graph-friendly fixed allocations while implementation guards/skips invalid rows |
| K-grouped layout | Targets APIs such as weight gradients; not a generic M-imbalance fix |
| Expert placement/replication | Trades device memory and communication for cross-device balance |
| Fusion | May overlap/avoid intermediates but can enlarge the scheduling unit |

The GPU Mode Problem 4 “reward hack” reused/cached evaluation behavior; it is an evaluation-harness incident, not evidence that a particular MoE scheduling technique balances experts.

## Verification

Report the token histogram, group shapes, padding/skip behavior, worker timeline, and communication volume. Compare uniform and adversarial routing. Avoid fixed “80/20” or speedup claims unless the cited experiment supplies that distribution and environment.
