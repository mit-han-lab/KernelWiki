---
id: blog-tflops-gap-fp4-moe
title: 'TFLOPS Gap: Why FP4 MoE Kernel Engineering Matters on Blackwell'
author: apsys (Hugging Face)
url: https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison
source_category: benchmark-blog
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, moe, warp-specialization, tma, kernel-fusion, tile-scheduling, persistent-kernel, block-scale, gemm, grouped-gemm, fine-grained-quantization]
retrieved_at: 2026-08-16
---

# Blackwell NVFP4 backend comparison

This third-party Hugging Face community post compares vLLM, SGLang, and FlashInfer on B200 for an NVFP4 GPT-OSS-20B MoE configuration: 32 experts, top-4 routing, hidden size 2,880, and intermediate size 7,680. It gives:

| Backend label | Batch 4096 | Batch 1 |
|---|---:|---:|
| SGLang FP4 | 1,262 TFLOP/s | 206.9 µs/layer |
| FlashInfer FP4 | 1,225 TFLOP/s | 481.9 µs/layer |
| vLLM FP4 | 1,117 TFLOP/s | 369.5 µs/layer |

It also reports, for batch 128, 0.433 ms/layer (157.1 TFLOP/s) for SGLang and 0.604 ms/layer (112.6 TFLOP/s) for vLLM. These are source-reported benchmark snapshots; KernelWiki has not reproduced them.

The appendix states the environment as B200 (`sm_100a`) on Nebius, vLLM v0.11.0, SGLang v0.5.5rc2, FlashInfer CuTe DSL from SGLang, and CUDA 13.0. Its method uses 20 warmup iterations, 200 measured iterations, mean latency, `torch.cuda.synchronize()` after each iteration, and a stated MoE FLOPS formula. PyTorch, driver, and standalone FlashInfer versions are not identified.

The post discusses fusion, CUTLASS schedule selection, and launch sizing as explanations. The table alone does not isolate those causes, prove exact launch counts for every compared version, or make a 21.9% memory-traffic reduction universal. The former local summary presented those interpretations as established causal facts and embedded an unverified one-line alignment fragment; those overclaims are removed.
