---
id: kernel-deepgemm
title: DeepGEMM — Fine-Grained-Scaled GEMM Library
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, fp4, fine-grained-quantization, block-scale]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, grouped-gemm]
languages: [cuda-cpp, ptx]
related: [technique-fine-grained-quantization, hw-tcgen05-mma, hw-nvfp4]
sources: [blog-deepgemm, pr-deepgemm-304, pr-cutlass-2139, pr-vllm-23696]
performance_claims:
  - gpu: H800
    software: "DeepGEMM README news item dated 2025-04-18; CUDA and library versions not stated"
    dtype: fp8
    shape: unspecified peak case in the 2025-04-18 README news item
    workload: "FP8 GEMM peak headline; exact operation variant not stated"
    metric: TFLOPS
    value: 1550
    measurement_method: "source-reported project benchmark; method not stated in the headline"
    baseline: "none; absolute throughput headline"
    limitations: "shape, clocks, CUDA version, and measurement method are not stated in the retained headline"
    source_id: blog-deepgemm
blackwell_relevance: "The current SM100 path uses tcgen05 block-scaled MMA, TMEM accumulators, and packed UE8M0 scale layouts; current releases also include FP4, BF16, attention/indexer, and fused MoE kernels."
artifact_dir: artifacts/kernels/deepgemm
---

# DeepGEMM

## Current scope

DeepGEMM is a runtime-compiled CUDA kernel library supporting SM90 and SM100. Its current upstream README includes dense and M-grouped FP8 GEMMs, K-grouped weight-gradient GEMMs, BF16 paths, FP8/FP4 attention-indexer kernels, and Mega MoE. PR 304 added major SM100 functionality including FP8-by-FP4 Mega MoE and FP4 indexer work.

The project uses a lightweight JIT module. NVCC is the documented default compiler; `DG_JIT_USE_NVRTC=1` is an optional faster-compilation path that may reduce performance in some cases. Therefore “JIT via NVRTC” is not an unconditional description of current DeepGEMM.

## Scaling and layouts

The upstream interface documentation distinguishes:

- SM90 scale tensors in FP32;
- SM100 scale tensors in a packed UE8M0 layout (four scale values per `torch.int`) for the FP8 paths described there;
- SM90 dense input layout restricted to NT;
- SM100 dense APIs supporting NT, TN, NN, and TT;
- M-grouped contiguous and masked layouts with fixed N/K, plus K-grouped APIs with their own constraints.

The historical fine-grained scheme commonly associated with DeepGEMM uses 1-by-128 activation and 128-by-128 weight scaling. Current APIs accept multiple recipes/granularities, so that pair is not a complete description of every kernel.

## SM100 implementation anchor

The shipped upstream SM100 source constructs a block-scaled instruction descriptor with UE8M0 scales, copies scale-factor tiles into TMEM, issues `SM100_MMA_MXF8F6F4_*`, and commits completion to barriers. It explicitly notes that `tcgen05.commit` supplies the relevant before-thread-sync fence for that sequence.

```python
def sm100_dependency_shape():
    tma_load_quantized_operands_and_packed_scales()
    wait_for_stage_and_order_cross_thread_tcgen_access()
    copy_scale_factor_fragments_to_tmem()
    issue_mxf8f6f4_block_scaled_mma()
    commit_mma_to_completion_barriers()
    wait_before_tmem_epilogue_load()
```

This is a structural summary, not inline PTX. The real descriptor contains shape, major order, scale IDs, and accumulation flags that a shortened assembly example would omit.

## Precision boundary

The SM90 implementation uses `BLOCK_K = 128`, issues four K=32 WGMMA operations for each such scale block, and then applies its CUDA-core promotion step before continuing the final FP32 accumulation. The per-128-K scaling interval and the promotion operation are related in that implementation but are not interchangeable terms. The presence of TMEM on SM100 removes distributed register storage for `tcgen05` destinations but does not, by itself, prove “full FP32” arithmetic or eliminate every numerical concern. Accumulator semantics come from the selected MMA kind and output type and should be tested against the application tolerance.

## Performance boundary

DeepGEMM's current README retains a historical “up to 1,550 TFLOP/s on H800” news item dated 2025-04-18 and links the relevant changes. It does not attach that headline to the 4096-by-4096-by-4096 shape or a universal 90% utilization figure. Preserve it as an unspecified peak, not a per-shape benchmark.

## Full Reference Implementation

Pinned upstream SM90/SM100 code is under [`artifacts/kernels/deepgemm/full/`](../../artifacts/kernels/deepgemm/full/). The former accumulation teaching variant was removed because its pseudo-API and variable naming mixed the SM90 promotion/scaling mechanism with SM100 behavior; the exact upstream SM90 mechanism is described above instead.
