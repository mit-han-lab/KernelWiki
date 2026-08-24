---
id: pattern-moe-load-imbalance
title: "MoE Expert Load Imbalance"
type: pattern
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm]
---

# MoE Expert Load Imbalance

## Symptom

A grouped-GEMM trace shows unequal expert tile counts or runtimes, leaving some CTAs or SMs idle while other expert groups remain active. Confirm the imbalance from the actual routed row counts and per-tile timings; no fixed router-distribution ratio is assumed.

## Kernel-local causes

- routed row counts differ across experts;
- a static tile assignment cannot adapt to unequal tile runtimes;
- fixed-capacity masked layouts perform or carry padding work;
- very small expert matrices do not fill the selected MMA tile effectively.

## Candidate techniques

| Technique | Kernel-local effect |
|---|---|
| [CLC](../hardware/clc.md) | Lets a persistent kernel acquire remaining grid work dynamically on supported Blackwell targets. |
| [Persistent kernels](../techniques/persistent-kernels.md) | Reuse resident CTAs across a work queue. |
| [Contiguous grouped layout](../kernels/grouped-gemm.md) | Packs valid routed rows and uses offsets to identify expert boundaries. |
| [Masked grouped layout](../kernels/grouped-gemm.md) | Keeps graph-friendly static storage at the cost of masked capacity. |

Expert replication and placement across GPUs are cluster/system policies and are outside this kernel-only page. The earlier EPLB speedups and a generic “80/20” routing assertion were removed.

## Measurement boundary

Compare static and dynamic schedulers with identical routed inputs, tile shapes, correctness checks, and launch conditions. Dynamic acquisition has its own instruction/synchronization cost, so it should be justified by measured imbalance rather than assumed to win universally.
