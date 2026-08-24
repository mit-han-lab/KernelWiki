---
id: blog-tflops-gap-fp4-moe
title: 'TFLOPS Gap: Why FP4 MoE Kernel Engineering Matters on Blackwell'
author: Konstantin (apsys)
url: https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison
source_category: benchmark-blog
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, moe, warp-specialization, tma, kernel-fusion, tile-scheduling, block-scale, grouped-gemm]
retrieved_at: 2026-08-18
---

# TFLOPS Gap: FP4 MoE backends on Blackwell

This Hugging Face community benchmark compares vLLM, SGLang, and FlashInfer
CuteDSL grouped MoE paths on one B200 for a GPT-OSS-20B-like configuration
with 32 experts, top-four routing, hidden size 2880, and intermediate size
7680.

## Source-reported results

At batch size 4096, the article reports 1262 TFLOP/s for SGLang FP4,
1225 TFLOP/s for FlashInfer FP4, and 1117 TFLOP/s for vLLM FP4. At batch size
128, it reports 0.433 ms per layer for SGLang and 0.604 ms for vLLM. At batch
size one, it reports 206.9, 481.9, and 369.5 microseconds per layer for SGLang,
FlashInfer, and vLLM respectively.

Those are author-run effective-throughput measurements, not independently
reproduced hardware peaks.

## Author's implementation analysis

The article attributes differences to three areas:

- fewer global-memory passes through fusion;
- a Blackwell NVFP4 CUTLASS schedule with TMA and warp specialization; and
- an adaptive launch heuristic that reduces block size and increases grid size
  for small workloads.

It shows the SGLang schedule name
`KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` and a block-scale offset rounding
expression using 128-token boundaries. Those details describe the cited
implementation; they do not establish a universal TMA alignment rule or prove
that one mechanism alone caused the measured end-to-end gap.

For a scaled 256-expert, top-eight configuration at batch size 4096, the article
reports 1132 TFLOP/s for FlashInfer, 993 TFLOP/s for SGLang, and 968 TFLOP/s for
vLLM. The local map retains these only as source-reported results and does not
extend them to distributed DeepSeek deployments.
