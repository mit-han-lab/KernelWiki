---
id: technique-kernel-fusion
title: Kernel fusion
type: technique
architectures: [sm100, sm90]
tags: [kernel-fusion, fused-kernel, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-fused-moe, kernel-nvfp4-gemm, technique-epilogue-fusion]
sources: [contest-gpumode-p3, contest-flashinfer-track-a, blog-tflops-gap-fp4-moe, pr-TensorRT-LLM-11897]
blackwell_relevance: TMEM can hold accumulator state for some fused SM100 designs, but the legal and profitable fusion boundary is kernel-specific.
---

# Kernel fusion

Kernel fusion keeps producer/consumer operations within one launch to avoid
materializing selected intermediates in global memory. It can reduce launch and
traffic costs, but it may increase live state, registers, shared memory, TMEM,
code size, and synchronization. A workload's operation graph—not a fixed
number of kernels—defines the legal boundary.

TensorRT-LLM PR 11897 provides a retained example: a Blackwell NVFP4 dense GEMM
fused with SwiGLU. This contiguous call-site excerpt is the BF16-output path:

```python
output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
    act_fp4, module.weight, act_sf, module.weight_scale, alpha,
    module.dtype)
```

The source has separate contracts for shapes, alignment, input scales, and
output type. It does not establish that routing, two expert GEMMs, activation,
and combine should all be fused into one device kernel.

Evaluate fusion by holding numerics and workload constant, then measuring
launch count, intermediate bytes, occupancy/resources, and end-to-end latency.
The former local example was removed because `tmem_alloc` and `tcgen05_mma`
were invented pseudo-APIs presented as CUDA.
