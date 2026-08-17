---
id: hw-2sm-cooperative
title: "Two-CTA Cooperative MMA"
type: hardware
architectures: [sm100, sm100a]
tags: [2sm-cooperative, tcgen05, cluster]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, doc-cutlass-blackwell, blog-colfax-cutlass, blog-modular-blackwell, pr-cutlass-2139]
aliases: ["2-SM cooperative", "dual CTA", "2CTA", "cta_group::2"]
---

# Two-CTA Cooperative MMA

## Overview

`tcgen05.mma.cta_group::2` lets a CTA pair in a two-CTA cluster cooperate on an MMA operation. The instruction's data-path and TMEM layouts distribute the work across the pair; issue, descriptor, synchronization, and allocation rules differ from `cta_group::1`.

The mode is not one fixed `m256 x n256 x k16` instruction. Valid M/N/K shapes depend on MMA kind and datatype. Current CUTLASS tables include both M=128 and M=256 two-SM legacy-type MMA tiles and narrower fixed combinations for NVFP4/MXFP4.

## Structural requirements

```python
def two_cta_mma(cluster_pair, instruction):
    assert cluster_pair.size == 2
    prepare_each_cta_operands_and_descriptors(instruction)
    coordinate_tmem_allocation_and_pipeline(cluster_pair, instruction)
    elected_issuer_follows_cta_group_2_rules(instruction)
    commit_and_wait_for_pair_completion()
    each_cta_drains_its_layout_defined_output_partition()
```

This is dependency pseudocode. Use the exact PTX issue rules and CUTLASS traits; simply issuing the same inline assembly independently from both CTAs is not a valid substitute.

## Tradeoffs

Two-CTA MMA can increase the cooperative tile size and enable multicast/reuse, but it binds two SMs into one schedulable cluster and can reduce the number of independent workers. It is useful only when the larger tile, datatype, shape, and surrounding pipeline offset cluster residency and tail costs.

Tutorial figures that improve after enabling 2CTA also reflect that tutorial's tile, pipeline, and hardware. They do not establish a universal 7.5% gain.
