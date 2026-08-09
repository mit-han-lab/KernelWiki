---
id: contest-flashinfer-track-c
title: 'FlashInfer MLSys 2026 Track C: Gated Delta Net'
source_category: contest-report
architectures:
- sm100
- sm100a
tags:
- gated-delta-net
- linear-attention
- chunk-parallelism
techniques:
- chunk-parallelism
hardware_features: []
kernel_types:
- gated-delta-net
- linear-attention
- decode
- prefill
languages:
- cute-dsl
- python
url: https://mlsys26.flashinfer.ai/
benchmark_decode_url: https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last
benchmark_prefill_url: https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last
captured_at: 2026-08-08
flashinfer_sha: 7f614b86470180bab2d22e36fd1775791c6bf3e6
---

# Track C: Gated Delta Net

## Primary scope

The MLSys 2026 FlashInfer organizer identifies Gated Delta Net as Track C, targets NVIDIA Blackwell B200, and links separate decode and prefill definitions. Both definitions were captured from Qwen3-Next linear-attention layers at tensor parallelism four and are tagged `status:verified`.

## Fixed geometry

| Axis | Value |
| --- | ---: |
| `num_q_heads` | 4 |
| `num_k_heads` | 4 |
| `num_v_heads` | 8 |
| `head_size` | 128 |

The heads remain separate. The recurrent state is `[batch_or_sequences,8,128,128]`, not a flattened `512x1024` matrix.

## Decode definition

The decode definition fixes `seq_len=1` and varies batch size. Q, K, and V are BF16; the state is FP32 in K-last layout `[B,H,V,K]`. The public inputs also include `A_log`, `a`, `dt_bias`, `b`, and an optional scale. Output is BF16 `[B,1,8,128]`, accompanied by the updated FP32 state.

At FlashInfer commit `7f614b8`, the reference computes:

```text
g = exp(-exp(A_log) * softplus(a + dt_bias))
beta = sigmoid(b)
decayed = g * state
read = k @ decayed
new_state = decayed + outer(k, beta * (v - read))
output = scale * (q @ new_state)
```

## Prefill definition

The prefill definition varies total sequence length and sequence count, accepts `cu_seqlens[num_seqs+1]`, and returns one final state per sequence. At commit `7f614b8`, FlashInfer dispatches SM90 and SM100 implementations. The SM100/SM103 CuTe DSL path requires CUDA 13 or newer and head size 128.

## Results boundary

The current organizer page lists Track C winners separately:

- Agent-assisted: Kachua, UW SyFI, and LLM-CUDA.
- Full-agent: UW SyFI, LLM-CUDA, and HAN Lab Kernel Mafia.

The former local Gemini/GPT/Claude ranking was a stale agent-baseline snapshot and is not reproduced as a final result. No latency or speedup record is retained here because a revision-pinned result table with workload, environment, timing protocol, repetitions, statistic, and variance was not captured.

## Primary sources

- [Organizer and winners](https://mlsys26.flashinfer.ai/)
- [Decode definition](https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last)
- [Prefill definition](https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last)
- [Pinned FlashInfer reference](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/trace/templates/gdn.py)
- [Pinned FlashInfer prefill dispatch](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/gdn_prefill.py)
