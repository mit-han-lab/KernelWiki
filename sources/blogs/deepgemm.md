---
id: blog-deepgemm
title: DeepGEMM
author: DeepSeek AI
url: https://github.com/deepseek-ai/DeepGEMM
source_category: benchmark-blog
architectures: [sm100, sm90]
tags: [gemm, fp8, fine-grained-quantization, block-scale, jit-compilation, tcgen05, wgmma]
retrieved_at: 2026-08-16
---

# DeepGEMM

DeepGEMM is DeepSeek's runtime-compiled CUDA library for dense/grouped GEMM and related LLM primitives on SM90 and SM100. The checked README states:

- SM90 GEMM uses FP32 scale factors; SM100 uses packed UE8M0 scale factors;
- SM90 supports the NT layout, while the described SM100 interfaces cover NT, TN, NN, and TT;
- M-grouped contiguous GEMM varies M while N and K remain fixed; a separate K-grouped API serves weight gradients;
- masked grouped GEMM handles decode cases where valid expert-token counts remain device-side;
- the default JIT compiler is NVCC; `DG_JIT_USE_NVRTC=1` opts into faster NVRTC compilation with possible performance loss; and
- the README's historical 1,550-TFLOP/s H800 headline is an unspecified peak tied to April 2025 changes, not a shape-independent guarantee.

As of the checked record, the project also describes lightning-indexer MQA scoring and Mega MoE. Those interfaces have their own layouts, requirements, and benchmark links.

The former local source contained synthesized pseudo-kernels, asserted “full FP32” TMEM accumulation, and called NVRTC the default. Those claims are removed. Use the pinned files under kernel artifact bundles or the upstream repository for implementation evidence.
