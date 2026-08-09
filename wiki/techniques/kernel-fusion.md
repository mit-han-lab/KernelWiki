---
id: technique-kernel-fusion
title: "Kernel Fusion"
type: technique
architectures: [sm100, sm90]
tags: [kernel-fusion, fused-kernel, tmem]
confidence: verified
evidence_basis:
  - source_id: contest-gpumode-p3
    evidence_type: upstream-code
reproducibility: pseudocode
prerequisites: [hw-tmem]
related: [kernel-fused-moe, kernel-nvfp4-gemm, technique-epilogue-fusion]
sources: [contest-gpumode-p3, contest-flashinfer-track-a]
blackwell_relevance: "Blackwell TMEM can hold tcgen05 accumulators for fused schedules, but fusion legality and benefit still depend on the exact dataflow, live storage, synchronization, and output contract."
---

# Kernel Fusion

## Definition and evidence boundary

Kernel fusion implements two or more dependent logical operations inside one
GPU kernel launch. It can remove a launch or an intermediate global-memory
write/read only when the chosen unfused baseline would perform that work and
the fused implementation keeps the value on chip. Fusion is therefore an
implementation property, not a conclusion that follows from an operation's
name.

Report a benefit only for a defined fused/unfused pair with identical input and
output semantics. Count actual launches and bytes for the pinned implementation;
do not transfer framework-level launch counts or traffic percentages from an
unpinned configuration.

## Gated dual GEMM

GPU Mode NVFP4 Challenge 3 fixes this operation for each batch index:

```python
gate = scaled_mm(a, b1.T, sfa, sfb1)
up = scaled_mm(a, b2.T, sfa, sfb2)
output = silu(gate) * up
```

The two products share `a` and its scale tensor but use separate B operands and
scales. This operation graph permits reuse and fused pointwise output, but the
pinned correctness reference does not prescribe launch count, TMEM partition,
TMA pipeline, or instruction schedule. A concatenated gate/up projection or two
logical accumulators can both implement compatible observable semantics when
their tensor contracts match.

## Fused MoE

FlashInfer MLSys 2026 Track A includes sigmoid routing, grouped expert
selection, two grouped GEMMs, SwiGLU, and weighted expert accumulation in one
logical benchmark definition. The exact reference performs one concatenated
W13 projection, splits its 4096 columns into gate and up halves, applies
SwiGLU, then performs W2. The term “Fused MoE” does not establish that a
submission uses one launch or keeps every intermediate on chip.

## Implementation constraints

- **Semantics:** preserve routing, scale, bias, activation, reduction,
  accumulation, dtype, ordering, and output-edge behavior. A pointwise epilogue
  is easier to compose than a cross-CTA reduction or global routing decision.
- **Communication:** identify the producer and every consumer of each
  intermediate. Use the synchronization scope and memory space actually needed;
  fusion is not restricted to one CTA, but multi-CTA communication has its own
  legality and completion rules.
- **Resources:** record registers, spills, SMEM, TMEM, barriers, threads, cluster
  shape, and occupancy. Added live values can raise register or shared-memory
  pressure even when a global intermediate disappears.
- **TMEM:** SM100 TMEM is 128 lanes by 512 columns of 32-bit cells per SM.
  Simultaneously live regions must fit, use legal collective allocation sizes,
  remain live through every asynchronous access, and be collectively freed.
  Two 256-column halves are one possible allocation policy, not the total
  hardware capacity or a dual-GEMM requirement.
- **Fallbacks:** keep unfused or differently fused paths for shapes, dtypes,
  layouts, reductions, or resource footprints that the fused schedule cannot
  implement safely.

## Evaluation

For correctness, compare the fused and unfused paths on identical tensors and
exercise empty/small groups, partial tiles, extreme activations, routing ties,
and every supported scale/layout variant. Add negative tests for premature
buffer reuse, omitted completion, wrong routing weights, and invalid output
edges.

For performance, include preparation and layout conversion unless the real
producer already emits the required representation. Record hardware, software,
shapes, dtypes, warmup, synchronization, repeated trials, statistic, variance,
launch count, bytes, and resource usage. Saved traffic does not guarantee a
speedup: extra computation, synchronization, staging, spills, or lower occupancy
can outweigh it.

## Primary references

- [GPU Mode NVFP4 gated dual GEMM at `c5b2f7c`](https://github.com/gpu-mode/reference-kernels/tree/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm)
- [FlashInfer Track A definition](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048)
- [FlashInfer MoE reference at `7f614b8`](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/trace/templates/moe.py)
- [PTX ISA 9.0 Tensor Memory](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensor-memory)

## Related

- [fused MoE](../kernels/fused-moe.md)
- [epilogue fusion](epilogue-fusion.md)
