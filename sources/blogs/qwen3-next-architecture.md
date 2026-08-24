---
id: blog-qwen3-next-architecture
title: "NVIDIA Qwen3-Next Architecture Announcement"
author: Anu Srivastava (NVIDIA)
url: https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/
source_category: community-note
architectures: [sm90, sm100]
tags: [gated-delta-net, moe, linear-attention, attention]
retrieved_at: 2026-08-18
---

# NVIDIA Qwen3-Next Architecture Announcement

NVIDIA's announcement describes the Qwen3-Next 80B-A3B Instruct and Thinking
models as hybrid sparse-MoE models intended for long input contexts.

The article reports:

- 80 billion total parameters and 3 billion activated per token;
- 512 routed experts plus one shared expert, with 10 experts activated per
  token;
- 48 layers, with every fourth layer using GQA and the other layers using
  linear attention;
- input contexts longer than 260,000 tokens; and
- Gated Delta Networks as the linear-attention mechanism.

It also states that the model can run on NVIDIA Hopper and Blackwell, describes
SGLang, vLLM, and NIM deployment paths, and quotes 1.8 TB/s for Blackwell's
fifth-generation NVLink. The article connects that fabric to expert-routing
communication, but it does not isolate a kernel-level latency or throughput
benefit from NVLink. This local source map therefore does not infer such a
speedup or treat total/active parameter counts as equivalent to dense-model
inference cost.
