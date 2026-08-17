---
id: pattern-tail-effect
title: "Tail Effect — Last Wave Underutilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-low-sm-utilization]
sources: [doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2161]
---

# Tail Effect

## Symptom

Performance can drop when the number of independently schedulable CTAs or clusters does not fill the final residency wave. The relevant worker count is not always the number of SMs: cluster size, blocks resident per SM, and resource limits determine wave capacity.

## Likely causes

1. A grid of N workers executed with resident capacity R needs `ceil(N/R)` waves; the final wave may contain only `N mod R` workers.
2. Variable-duration workers leave a long straggler after peers complete.
3. A tile or cluster shape creates too few independent workers.

## Candidate techniques

| Technique | Effect |
|---|---|
| [CLC](../hardware/clc.md) | Running clusters can take work IDs from canceled, unlaunched clusters |
| [Persistent kernels](../techniques/persistent-kernels.md) | Resident workers iterate over logical work, trading wave loss for scheduler overhead |
| [Tile scheduling](../techniques/tile-scheduling.md) | Changes granularity/order and may improve locality or balance |

```text
Illustration: if resident worker capacity R=142 and the grid has 150 equal
workers, the second wave contains 8 workers. A persistent/CLC schedule may let
earlier workers take future launch IDs, but it cannot guarantee every SM stays
busy or remove all tail time.
```

The fractional loss is often amortized across many uniform waves, but one expensive tail worker can still dominate. Persistence, Stream-K, and smaller tiles add scheduler, reduction, or locality costs. Tutorial utilization changes are whole-kernel measurements, not isolated CLC guarantees.
