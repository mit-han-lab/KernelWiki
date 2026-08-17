---
id: kernel-gated-delta-net
title: Gated Delta Net — Linear Attention
type: kernel
architectures: [sm100, sm90]
tags: [gated-delta-net, linear-attention, attention]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-delta-net, linear-attention, decode, prefill, attention]
languages: [triton, cuda-cpp]
related: [technique-chunk-parallelism, technique-kernel-fusion]
sources: [doc-gated-delta-net-paper, blog-gated-delta-net, doc-tfla, pr-vllm-37303]
performance_claims: []
blackwell_relevance: "The cited implementations include recurrent decode and chunk-parallel prefill paths; SM100 support and performance must be established per implementation and shape."
artifact_dir: artifacts/kernels/gated-delta-net
---

# Gated Delta Net

Gated Delta Networks are recurrent linear-attention layers. A learned decay gate controls retention, while a delta-rule update corrects the value associated with the current key. Decode carries a fixed-shape state forward instead of appending a conventional per-token attention KV cache.

## Execution modes

- **Decode:** update the recurrent state for one token and read the output from that state. Work per token is independent of prior sequence length, although state traffic and launch overhead still depend on the implementation.
- **Prefill:** partition the sequence into chunks, use matrix operations within a chunk, and propagate summary state between chunks.
- **Hybrid models:** architectures such as Qwen3-Next interleave Gated Delta Net layers with full-attention layers. It is therefore incorrect to say that the whole model eliminates KV-cache growth.

```python
def gated_delta_step(state, key, value, decay, update_rate):
    decayed_state = decay * state
    prediction = key @ decayed_state
    correction = value - prediction
    return decayed_state + update_rate * outer(key, correction)
```

The decay participates in both the carried state and the delta prediction, matching the paper's gated delta recurrence. This mathematical sketch fixes the orientation convention locally; production implementations batch heads and choose their own layouts.

The exact state shape follows the model's head and expansion dimensions; a single `128 × 128` matrix or a repository-independent memory-size estimate is not a general property.

## Evidence boundary

The earlier `10×` performance entry combined a cross-model Qwen serving comparison with an unrelated H100 kernel shape. It was not a kernel-to-kernel benchmark and has been removed. Likewise, contest-status statements and Qwen3.5 adoption claims are time-sensitive and are not used here as architectural proof.

The pinned artifact contains an upstream SGLang fused-projection file. The code under `sources/blogs/gated-delta-net.md` was an explanatory sketch, not the NVlabs or FLA implementation, and is not treated as reproducible kernel evidence.

## Reproduction

See [`artifacts/kernels/gated-delta-net/full/`](../../artifacts/kernels/gated-delta-net/full/) for the pinned upstream file and its hash. The former teaching implementations were removed because they simplified the gated delta-rule recurrence into a different update.
