---
id: kernel-fused-moe
title: Fused MoE — NVFP4 Routing and Projections
type: kernel
architectures: [sm100, sm100a, sm90]
tags: [moe, fused-kernel, nvfp4, fp4, block-scale, kernel-fusion, grouped-gemm, gated-dual-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, fused-kernel, grouped-gemm, gated-dual-gemm]
languages: [cuda-cpp, cute-dsl, triton]
related: [kernel-grouped-gemm, kernel-deepgemm, technique-fine-grained-quantization, technique-tile-scheduling]
sources: [contest-flashinfer-track-a, blog-deepgemm, blog-tflops-gap-fp4-moe, pr-vllm-23696]
performance_claims:
  - gpu: B200
    software: "SGLang v0.5.5rc2, CUDA 13.0; compared with vLLM v0.11.0 and FlashInfer CuTe DSL from SGLang"
    dtype: nvfp4
    shape: GPT-OSS-20B, topk=4, experts=32, hidden=2880, intermediate=7680, batch=4096
    workload: "GPT-OSS-20B MoE layer at batch 4096"
    metric: TFLOPS
    value: 1262
    measurement_method: "third-party source-reported mean over 200 measured iterations after 20 warmups, with torch.cuda.synchronize after each iteration"
    baseline: "cross-framework table with FlashInfer FP4 and vLLM FP4"
    limitations: "PyTorch, driver, and standalone FlashInfer versions are not stated; result and source-specific MoE FLOPS convention are not reproduced locally"
    source_id: blog-tflops-gap-fp4-moe
blackwell_relevance: "The cited third-party performance note compares B200 NVFP4 MoE baselines for GPT-OSS-20B; the exact fusion boundary and launch count are implementation details, not consequences of SM100 alone."
artifact_dir: artifacts/kernels/fused-moe
---

# Fused MoE

An MoE layer routes tokens to experts, executes gate/up projections and an activation, executes a down projection, and combines expert outputs. Implementations may fuse adjacent stages or keep routing, grouped GEMMs, and combination separate. “Fused MoE” therefore names a family of boundaries rather than a guaranteed one-kernel implementation.

```python
def expert_mlp(tokens, gate_weight, up_weight, down_weight):
    gate = tokens @ gate_weight
    up = tokens @ up_weight
    return (silu(gate) * up) @ down_weight
```

Routing, quantization, scale application, and combination surround this per-expert numerical contract.

## Benchmark case

The Hugging Face performance note records a B200 NVFP4 GPT-OSS-20B case with top-4 routing, 32 experts, hidden size 2,880, and intermediate size 7,680. Its baseline table reports:

| Framework label | Batch 4096 | Batch 1 |
|---|---:|---:|
| SGLang FP4 | 1,262 TFLOP/s | 206.9 µs/layer |
| FlashInfer FP4 | 1,225 TFLOP/s | 481.9 µs/layer |
| vLLM FP4 | 1,117 TFLOP/s | 369.5 µs/layer |

These are third-party, source-reported baselines, not results reproduced by this repository or scores from the MLSys contest. The appendix identifies SGLang v0.5.5rc2, vLLM v0.11.0, FlashInfer CuTe DSL from SGLang, CUDA 13.0, B200 `sm_100a`, 20 warmup iterations, and 200 measured iterations with `torch.cuda.synchronize()` after each iteration. It does not identify the PyTorch/driver versions or a standalone FlashInfer release, and its MoE-specific FLOPS convention is retained as source-reported. The official contest winner page publishes placements but no numerical scores. The table alone does not establish each framework's launch count, attribute the difference to fusion, or support a universal memory-traffic saving.

## Optimization boundary

A gate/up fusion can reuse an activation tile and avoid writing both full intermediate projections before applying `SiLU(gate) * up`. It still reads two weight matrices and can increase accumulator, shared-memory, register, and synchronization pressure. Grouped scheduling must also handle variable expert token counts and empty or thin expert GEMMs.

SM100 implementations select legal `tcgen05` kinds and layout-defined TMEM allocations through library traits. The former pseudo-kernel used non-existent allocation helpers and an incorrect instruction kind, so it has been removed rather than presented as compilable CUDA.

## Reproduction

[`artifacts/kernels/fused-moe/full/`](../../artifacts/kernels/fused-moe/full/) contains a pinned vLLM PR patch and a pinned SGLang source file. They are upstream implementation evidence, not a contest submission.
