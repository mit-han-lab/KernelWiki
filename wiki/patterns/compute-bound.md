---
id: pattern-compute-bound
title: "Not Reaching Peak FLOPS"
type: pattern
tags: [tcgen05, 2sm-cooperative, pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [hw-2sm-cooperative, technique-pipeline-stages, technique-warp-specialization, technique-epilogue-fusion, technique-software-exp]
related: [pattern-low-sm-utilization, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-flash-attention-4]
---

## Symptom

Measured arithmetic throughput is well below the relevant roof while global-memory bandwidth is not the limiting resource. “Below 70%” is not a universal cutoff: attainable utilization depends on dtype, instruction shape, sparsity, clocking, and the non-MMA work in the kernel.

## Likely Causes

1. **Pipeline bubbles**: MMA stalled waiting for data from TMA
2. **Non-matmul overhead**: Softmax, activation functions, reductions consuming cycles
3. **Single-SM MMA tiles too small**: Not fully utilizing available compute
4. **Epilogue blocking mainloop**: TMEM reads blocking next MMA

## Candidate Techniques

| Technique | Effect |
|---|---|
| [2-SM cooperative](../hardware/2sm-cooperative.md) | Change operand staging and tile partition across a CTA pair; may reduce shared-memory traffic for suitable shapes |
| [Pipeline stages](../techniques/pipeline-stages.md) | Overlap TMA load with MMA compute |
| [Warp specialization](../techniques/warp-specialization.md) | Separate pipeline roles so independent stages can overlap; synchronization stalls can remain |
| [Epilogue fusion](../techniques/epilogue-fusion.md) | Overlap epilogue with next tile's MMA |
| [Software exponential](../techniques/software-exp.md) | Distribute non-matmul ops across FMA units (FA4) |

## Example: FlashAttention-4

```
// Problem: Blackwell doubles tensor core throughput but SFU count unchanged
// SFU bottleneck: exp() for softmax
//
// Solution: Software 2^x via Cody-Waite + Horner polynomial
// Uses FMA capacity for a tuned subset of exp2 evaluations while the rest use MUFU
// FA4 paper result for its full kernel: up to 1613 TFLOP/s (~71%) on B200.
```

## Caveats
- 2-SM cooperative requires cluster configuration and identical SMEM layouts
- Pipeline depth tuning is workload- and resource-dependent; this page does not prescribe a universal stage count
- Software-emulated transcendentals trade accuracy for throughput
