---
id: kernel-gated-delta-net
title: Gated Delta Net — Linear Attention
type: kernel
architectures:
- sm100
- sm90
tags:
- gated-delta-net
- linear-attention
- attention
confidence: source-reported
reproducibility: snippet
kernel_types:
- gated-delta-net
- linear-attention
- decode
- prefill
- attention
languages:
- cute-dsl
- triton
- python
related:
- technique-pipeline-stages
sources:
- blog-gated-delta-net
- contest-flashinfer-track-c
- blog-qwen3-next-architecture
- doc-tfla
- pr-vllm-37303
performance_claims: []
blackwell_relevance: FlashInfer commit 7f614b8 implements an SM100/SM103 CuTe
  DSL prefill path that requires CUDA 13 or newer and head size 128.
artifact_dir: artifacts/kernels/gated-delta-net
---

# Gated Delta Net — Linear Attention

## Verified mechanism

Gated DeltaNet is a recurrent linear-attention architecture published at ICLR 2025. For one head with state `S` shaped `[K,V]`, the FlashInfer reference at commit `7f614b8` implements the update in this equivalent form:

```python
def gdn_step(S, q, k, v, A_log, a, dt_bias, b, scale):
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)
    decayed = g * S
    read = k @ decayed
    S_new = decayed + outer(k, beta * (v - read))
    output = scale * (q @ S_new)
    return output, S_new
```

The subtraction is the delta-rule correction: the update moves the value retrieved at `k` toward `v`, while `g` independently decays the previous state. The recurrent state has no prior-token axis, so a single GDN decode step has work and storage fixed with respect to context length. Its per-head state update is still proportional to `K*V`; “constant” here means constant in the number of earlier tokens.

## Qwen3-Next architecture

The immutable `Qwen3-Next-80B-A3B-Instruct` configuration at revision `9c7f2fbe` records:

| Field | Value |
| --- | --- |
| Layers | 48 |
| Hybrid layout | `12 * (3 * Gated DeltaNet -> MoE, 1 * Gated Attention -> MoE)` |
| GDN heads | 16 QK heads, 32 value heads, head dimension 128 |
| MoE | 512 experts; 10 routed experts plus one shared expert |
| Parameters | 80B total, 3B activated |
| Native context | 262,144 tokens |

Thus 36 layers use GDN and 12 use full Gated Attention. The fixed GDN state does not eliminate cache growth for the whole hybrid model: the full-attention layers retain their own context-dependent cache.

Qwen's architecture article attributes attention-sink and massive-activation mitigation to the output gate in its full Gated Attention path and says the gate *helps* address those effects. That is separate from the GDN recurrence and is not a requirement imposed by the GDN kernel contract.

## FlashInfer-Bench Track C contracts

The MLSys 2026 organizer identifies Gated Delta Net as Track C and links separate verified definitions captured from Qwen3-Next with tensor parallelism four.

| Axis | Decode | Prefill |
| --- | --- | --- |
| Q heads / K heads / V heads | `4 / 4 / 8` | `4 / 4 / 8` |
| Head size | `128` | `128` |
| Token axis | `seq_len=1`, variable `batch_size` | variable `total_seq_len` and `num_seqs` |
| Q/K/V dtype | BF16 | BF16 |
| State | FP32 `[B,8,128,128]` | FP32 `[N,8,128,128]` |
| Variable-length metadata | none | `cu_seqlens[N+1]` |

Decode exposes `A_log`, `a`, `dt_bias`, and `b` for the decay and update gates, plus an optional scale. It returns BF16 output `[B,1,8,128]` and the updated state. Prefill returns BF16 output `[total_seq_len,8,128]` and one final state per sequence.

For this exact geometry, one value-head state contains `128*128 = 16,384` FP32 values, or 64 KiB. All eight value heads contain 131,072 FP32 values, or 512 KiB, per sequence and layer. The heads are independent states; they must not be flattened into one `512x1024` matrix.

## Implementations and architecture scope

- NVlabs commit `b53d6d3` is the authors' PyTorch/Triton research implementation. Its README recommends FLA for faster kernels and variable-length functionality.
- FlashInfer commit `7f614b8` supplies the exact decode/prefill reference contracts used above. Its prefill wrapper dispatches SM90 and SM100 implementations; the SM100/SM103 path uses CuTe DSL, requires CUDA 13 or newer, and fixes head size 128.
- vLLM merge `e1d85e5c` gives its recurrent-attention backend uniform-batch CUDA-graph support for decode. CUDA graphs reduce CPU launch setup cost, but their benefit remains workload-dependent.

TFLA (`arXiv:2503.14376v3`) is related linear-recurrence work, but its published application and official code are for mLSTM. They use a second level of sequence parallelization to permit arbitrarily large chunks. They do not establish a GDN implementation or an inline WGMMA/tcgen05 path, so no TFLA assembly is presented here.

## Performance boundary and use

GDN sequence mixing scales linearly with sequence length, but asymptotic complexity is not a measured speedup. Qwen reports a 10x whole-model inference-throughput comparison against Qwen3-32B for contexts over 32K, while also warning that efficiency depends strongly on implementation. That result does not isolate this kernel, name a GPU, or describe the Track C qk4/v8/d128 workloads, so it is not stored as a kernel performance record.

Use an exact backend measurement for the intended batch, sequence distribution, dtype, state layout, software revision, and GPU. Also account for the context-growing full-attention cache in a hybrid model and for the GDN layer's learned projections, gates, convolution, and state; it is not a weight-compatible runtime substitute for a trained softmax-attention layer.

## Local artifacts

The `full/` bundle contains one byte-pinned SGLang file from merge `5bdc07d974f6cf236fa765a685453ea5e587a838`. It fuses projection-output split/reshape/concatenation for Qwen3-Next/Qwen3.5; it is adjacent preprocessing, not a GDN recurrence, prefill, or decode implementation.

The `variants/` bundle contains a standard-library, one-head recurrence check derived from the FlashInfer reference. It checks the compact update against the independently expanded remove/write form and rejects an additive-only negative control. It is not a tuned GPU kernel.

## Primary sources

- [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464)
- [NVlabs implementation at `b53d6d3`](https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921)
- [Qwen3-Next model card and configuration at `9c7f2fbe`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/tree/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7)
- [MLSys 2026 organizer](https://mlsys26.flashinfer.ai/)
- [Exact decode definition](https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last)
- [Exact prefill definition](https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last)
- [FlashInfer GDN trace at `7f614b8`](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/trace/templates/gdn.py)
- [FlashInfer prefill dispatch at `7f614b8`](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/gdn_prefill.py)
- [CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [TFLA v3](https://arxiv.org/abs/2503.14376v3)

Query via:

```bash
conda run -n base python scripts/get_page.py kernel-gated-delta-net --include-code
```
