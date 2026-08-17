---
id: blog-qwen3-next-architecture
title: Qwen3-Next hybrid Gated Delta Net and MoE architecture
author: NVIDIA / Alibaba
url: https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/
source_category: community-note
architectures: [sm100, sm100a]
tags: [gated-delta-net, moe, linear-attention, attention, sparse-attention, cluster]
retrieved_at: 2026-08-16
---

# Qwen3-Next architecture announcement

The announcement describes Qwen3-Next-80B-A3B as a hybrid model with 80B total parameters and about 3B activated per token, combining Gated Delta Net layers, periodic full-attention layers, and sparse MoE. It records a 3:1 Gated-Delta-Net-to-full-attention pattern and serving support in NVIDIA and open-source stacks at publication time.

These model-level properties do not imply that inference costs equal a dense 3B model: inactive weights still affect storage and communication, routing has overhead, and full-attention layers retain sequence-dependent KV state. Similarly, an NVLink bandwidth headline does not by itself quantify expert-routing latency or end-to-end throughput. The former local source made both causal leaps and is corrected here.

Use the model's own technical report/configuration for exact layer/expert counts; this page remains a launch-announcement source rather than kernel benchmark evidence.
