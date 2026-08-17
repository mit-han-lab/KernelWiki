---
id: blog-flashmla
title: FlashMLA
author: DeepSeek AI
url: https://github.com/deepseek-ai/FlashMLA
source_category: benchmark-blog
architectures: [sm100, sm90]
tags: [mla, attention, decode, prefill, fp8, sparse-attention, tcgen05, tmem]
retrieved_at: 2026-08-16
---

# FlashMLA

The checked upstream README lists dense decode (SM90), sparse decode (SM90/SM100), dense MHA prefill (SM100), and sparse MLA prefill (SM90/SM100). It reports the following peak headlines without a complete shape table in the headline paragraphs:

- H800 dense decode: up to 3,000 GB/s in a memory-bound case and 660 TFLOP/s in a compute-bound case;
- sparse decode: 410 TFLOP/s on H800 and up to 350 TFLOP/s on a then-unoptimized B200 path;
- B200 dense **MHA** prefill: 1,460 TFLOP/s forward and 1,000 TFLOP/s backward, explicitly labeled “as reported by NVIDIA”; and
- sparse MLA prefill: 640 TFLOP/s on H800 with CUDA 12.8 and 1,450 TFLOP/s on B200 with CUDA 12.9.

The README headline paragraphs do not state the compute dtype for either prefill result, so this record does not infer one from the project name or neighboring decode discussion.

The FP8 sparse-decode cache uses 656 bytes per token: 512 E4M3 values, four FP32 scales, and 64 BF16 RoPE values. The README explicitly says the cache is dequantized and attention computation/output use BF16.

Sparse kernels consume an `indices` tensor. Decode accepts `-1` for invalid entries; sparse prefill documents `-1` or values at least `s_kv`, has no batch dimension, and requires `h_kv=1` in its equivalent reference.

The former local summary called 656 bytes “massive compression to 70 KB/token” and embedded invented CUDA. Those statements were not upstream and are removed.
