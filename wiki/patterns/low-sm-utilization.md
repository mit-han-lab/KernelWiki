---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161]
---

## Symptom

Profiling shows idle SMs during material portions of kernel execution even though resource occupancy is not the immediate limiter. The earlier fixed utilization threshold was only a local heuristic and has been removed.

## Likely Causes

1. **Tail effect**: Last wave of tiles leaves most SMs idle (see [tail-effect](tail-effect.md))
2. **Load imbalance**: Some tiles take longer than others (variable computation per tile)
3. **Static scheduling**: Fixed tile-to-SM assignment doesn't adapt to runtime conditions
4. **Grid too small**: Fewer threadblocks than SMs

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [CLC](../hardware/clc.md) | PTX target `sm_100` or higher | May redistribute IDs of clusters that have not launched |
| [Persistent kernels](../techniques/persistent-kernels.md) | Software design | Can amortize launch/setup work and redistribute tasks when enough work remains |
| [Tile scheduling](../techniques/tile-scheduling.md) | Software design | Can trade locality against load balance |

## Caveats
- Verify the selected toolkit and target before using CLC; current PTX lists
  `clusterlaunchcontrol.try_cancel` as requiring `sm_100` or higher.
- Persistent kernels complicate debugging and profiling
- A larger grid can reduce wave quantization, but it can also add overhead and
  does not correct unequal per-tile costs by itself.
