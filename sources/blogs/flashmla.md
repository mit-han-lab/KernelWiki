---
id: blog-flashmla
title: FlashMLA upstream README
author: DeepSeek AI
url: https://github.com/deepseek-ai/FlashMLA
source_category: benchmark-blog
architectures: [sm100, sm90]
tags: [mla, attention, decode, prefill, fp8, sparse-attention]
retrieved_at: 2026-08-18
source_commit: 15f13e5030374295491c5ce31b02d7e63a7772c6
---

# FlashMLA upstream README

FlashMLA's README at commit
`15f13e5030374295491c5ce31b02d7e63a7772c6` lists four supported families:
dense SM90 MLA decode, sparse SM90/SM100 MLA decode with FP8 KV cache, dense
SM100 MHA prefill, and sparse SM90/SM100 MLA prefill.

## Source-reported results

- Dense decode on H800: up to 3000 GB/s in the memory-bound configuration and
  660 TFLOPS in the compute-bound configuration, CUDA 12.8.
- Sparse decode: 410 TFLOPS on H800 and up to 350 TFLOPS on B200; the README
  explicitly says the B200 path was not yet really optimized.
- Dense MHA prefill on B200: up to 1460 TFLOPS forward and 1000 TFLOPS backward,
  reported by NVIDIA.
- Sparse MLA prefill: up to 640 TFLOPS on H800 and 1450 TFLOPS on B200, CUDA
  12.9 for the B200 result.

These are separate benchmark modes, not one comparable table of identical
shapes.

## FP8 decode layout

For FP8 KV-cache decode, the README defines 656 bytes per token: 512 E4M3
values, four FP32 scales occupying 16 bytes, and 64 BF16 RoPE values occupying
128 bytes. The kernel dequantizes the cache to BF16, performs attention in
BF16, and returns BF16 output.

Sparse decode accepts an `indices` tensor. Invalid entries are `-1`; the encoded
indices already identify page and offset, so the sparse path does not require a
separate block table according to the README.

The former local CUDA blocks were removed because they were synthesized and did
not come from FlashMLA.
