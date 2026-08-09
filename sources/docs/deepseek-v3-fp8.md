---
id: doc-deepseek-v3-fp8
title: 'DeepSeek-V3 Technical Report: FP8 Training'
author: DeepSeek-AI
url: https://arxiv.org/abs/2412.19437v2
source_category: paper
architectures:
- sm90
tags: [fp8, fine-grained-quantization, block-scale, gemm]
retrieved_at: 2026-08-08
---

## Fine-Grained Quantization

Sections 3.3.1-3.3.2 define the DeepSeek-V3 FP8 mixed-precision framework. Forward activations are scaled per `1x128` tile and weights per `128x128` block so smaller groups can better accommodate outliers. The paper also documents phase-specific exceptions, such as `128x1` activation grouping for backward use.

## Hopper Accumulation

Sections 3.3.3 and 3.5.2 describe promotion on Hopper. A 128-element K interval—equivalent to four WGMMAs in the cited configuration—is accumulated and then combined with scaling factors in FP32 CUDA-core registers. The paper characterizes the Tensor Core alignment/addition precision relevant to the limitation as 14 bits and discusses the extra scale handling rather than claiming zero overhead.

This paper establishes the model/training format and numerical motivation. DeepGEMM's exact scale tensor layouts, supported recipes, and architecture-specific kernel implementations are pinned separately.
