---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2161]
---

# Low SM Utilization

## Symptom

Profiler timelines show fewer active SMs than the workload and resource limits appear able to use. Do not diagnose from a universal percentage threshold: “SM utilization” metrics differ, and a bandwidth-saturated kernel may be efficient without high issue activity.

## Likely causes

- Too few CTAs or clusters for the device.
- A partially filled final wave.
- Per-tile work variance or uneven grouped workloads.
- Resource limits allowing too few resident blocks.
- Dependencies that serialize nominally independent work.
- Host gaps, synchronization, or another kernel occupying resources.

## Candidate techniques

- Adjust tile/cluster shape or grid size when there is insufficient parallel work.
- Use a persistent software scheduler when many logical tiles can be processed by a smaller resident grid.
- On SM100, use CLC to let a running cluster cancel an **unlaunched** cluster and execute the work associated with its launch ID.
- Consider Stream-K or another partition only when reduction overhead is smaller than the tail loss.
- Reorder grouped work to reduce variance or improve locality.

CLC does not cancel running clusters and does not guarantee elimination of load imbalance or the final wave. Persistent kernels likewise trade launch/wave overhead against scheduler cost and long-lived resource residency.

## Verification

Compare full timelines, useful throughput, memory traffic, and tail duration. Preserve tutorial percentages as measurements of the cited kernel revision and hardware; they are not architecture-wide outcomes attributable solely to persistence or CLC.
