---
id: kernel-gated-dual-gemm
title: Gated Dual GEMM (Gate-Up + SwiGLU Fusion)
type: kernel
architectures: [sm100, sm90]
tags: [gated-dual-gemm, gemm, fused-kernel, kernel-fusion, nvfp4, tmem]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, cute-dsl]
related: [kernel-nvfp4-gemm, kernel-fused-moe, technique-kernel-fusion, technique-epilogue-fusion]
sources: [contest-gpumode-p3, blog-deepgemm, blog-tflops-gap-fp4-moe, pr-vllm-23696]
performance_claims: []
blackwell_relevance: "SM100 can keep layout-defined gate and up accumulators in TMEM and fuse their epilogue, subject to the 512-column budget and scale-factor storage."
---

# Gated Dual GEMM

## Operation

```text
gate = X @ W_gate
up   = X @ W_up
out  = SiLU(gate) * up
```

A fused implementation may reuse an X tile, issue two independent GEMM streams, and apply SiLU/multiply while draining their completed accumulator fragments. It avoids materializing both full projections when the fused resource footprint remains efficient.

## SM100 dependency structure

```python
def gated_dual_tile(x_tile, gate_weight_tiles, up_weight_tiles):
    gate_stage = allocate_layout_defined_accumulator()
    up_stage = allocate_layout_defined_accumulator()
    for gate_w, up_w in zip(gate_weight_tiles, up_weight_tiles):
        issue_block_scaled_mma(gate_stage, x_tile, gate_w)
        issue_block_scaled_mma(up_stage, x_tile, up_w)
    wait_for_both_accumulators()
    gate = load_tmem_fragment(gate_stage)
    up = load_tmem_fragment(up_stage)
    return silu(gate) * up
```

The logical sketch omits pipeline reuse and scale layouts. Two 256-column regions are one possible TMEM partition, not a general consequence of a 128-by-256 output tile. NVFP4 scale factors use UE4M3 and their own layout; TMEM column needs must come from the selected traits.

## Tradeoffs

- X traffic can be shared only if both paths consume the same staged layout/lifetime.
- Weight traffic is still distinct.
- Two accumulators and a fused epilogue can increase TMEM, registers, shared memory, and barrier pressure.
- Numerical behavior must preserve the reference SiLU, scale application, accumulator type, and output rounding.

## Contest evidence boundary

The task file establishes the dual-GEMM numerical contract, four benchmark
shapes, and a geometric-mean ranking rule. The separate public API snapshot on
2026-08-16 reported a 13.123660-us leading score and placed the account `Simon` 4th at
14.054539 us. It does not publish contestant implementations, so prior
participant-specific fusion attributions remain omitted. The API reports this
leaderboard ended at its 2026-01-20 deadline; the snapshot is not an independent
award claim.

## Implementation evidence boundary

No local bundle is claimed as a gated-dual-GEMM implementation. The previously
linked vLLM PR 23696 patch implements fused MXFP4 MoE and is retained only in
the fused-MoE bundle; it is supporting MoE context, not this exact operation.
