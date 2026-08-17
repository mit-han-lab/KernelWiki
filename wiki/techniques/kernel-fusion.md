---
id: technique-kernel-fusion
title: "Kernel Fusion"
type: technique
architectures: [sm100, sm90]
tags: [kernel-fusion, fused-kernel, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem]
related: [kernel-fused-moe, kernel-nvfp4-gemm, technique-epilogue-fusion]
sources: [contest-gpumode-p3, contest-flashinfer-track-a, blog-tflops-gap-fp4-moe, pr-vllm-23696]
blackwell_relevance: "TMEM can hold layout-defined accumulator stages for fused SM100 pipelines, but capacity, synchronization, and residency constrain fusion."
---

# Kernel Fusion

## Overview

Fusion combines operations so an intermediate can remain in registers, shared memory, TMEM, or another on-chip representation. Potential benefits are fewer launches, fewer global-memory round trips, and producer/consumer overlap. Costs include larger live state, lower occupancy, more complex scheduling, compilation, and reduced reuse of optimized library kernels.

## Dependency sketch

```python
def fused_gate_up(x, w_gate, w_up):
    gate_acc = block_scaled_gemm(x, w_gate)
    up_acc = block_scaled_gemm(x, w_up)
    gate = load_completed_accumulator_fragment(gate_acc)
    up = load_completed_accumulator_fragment(up_acc)
    return silu(gate) * up
```

On SM100, the two accumulator layouts may occupy separate TMEM column regions. Their column counts are derived from the MMA traits; there is no universal 256-column total or assumption that two 256-column allocations fit alongside scale-factor TMEM.

## Fusion boundary

A production fused kernel must define:

- completion and ownership for every accumulator stage;
- scale-factor and output rounding semantics;
- tails, expert/group boundaries, and optional inputs;
- whether shared operands are actually reused;
- how the larger kernel affects registers, shared memory, TMEM, cluster residency, and code size.

End-to-end MoE launch counts and traffic reductions depend on the serving stack, routing representation, quantization, and communication backend. Do not present a fixed “5–7 launches” or percentage saving without a pinned trace.

## Decision rule

Fuse when profiling shows that removed traffic/launch gaps outweigh the resource and scheduling cost across representative shapes. Keep unfused fallbacks for cases where a library GEMM plus a small epilogue wins or where independent launches provide needed concurrency.
