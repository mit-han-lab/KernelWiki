---
id: contest-flashinfer-track-a
title: 'FlashInfer MLSys 2026 Track A: FP8 Block-Scale MoE'
source_category: contest-report
architectures:
- sm100
- sm100a
tags:
- moe
- fp8
- block-scale
- grouped-gemm
techniques:
- kernel-fusion
- tile-scheduling
- fine-grained-quantization
hardware_features:
- fp8
- block-scale
kernel_types:
- moe
- grouped-gemm
- fused-kernel
languages:
- cuda-cpp
- cute-dsl
- triton
url: https://mlsys26.flashinfer.ai/
benchmark_url: https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
captured_at: 2026-08-08
starter_kit_sha: 75ccd05cafceb0fd1f86be4cd0f2117249463c66
flashinfer_sha: 7f614b86470180bab2d22e36fd1775791c6bf3e6
---

# Track A: FP8 Block-Scale MoE

## Primary Scope

The organizer page identifies Fused MoE as Track A of the MLSys 2026 FlashInfer AI Kernel Generation Contest. The contest challenge targets NVIDIA Blackwell B200 GPUs, and Track A links directly to this definition:

`moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

The definition describes an FP8 block-scale MoE operation with routing and two grouped GEMMs included. It is tagged for DeepSeek-V3 and DeepSeek-R1 and records expert parallelism `ep:8`.

## Definition Axes

| Axis | Value |
| --- | ---: |
| `seq_len` | variable |
| `num_experts` | 256 |
| `num_local_experts` | 32 |
| `hidden_size` | 7168 |
| `intermediate_size` | 2048 |
| `gemm1_out_size` | 4096 |
| `num_hidden_blocks` | 56 |
| `num_intermediate_blocks` | 16 |
| `num_gemm1_out_blocks` | 32 |

The definition name and reference fix `top_k=8`, `n_group=8`, and `topk_group=4`. Thus `e32` denotes local experts; routing logits still span 256 global experts.

## Signature

| Input | Dtype | Shape |
| --- | --- | --- |
| `routing_logits` | FP32 | `[seq_len, num_experts]` |
| `routing_bias` | BF16 | `[num_experts]` |
| `hidden_states` | FP8 E4M3FN | `[seq_len, hidden_size]` |
| `hidden_states_scale` | FP32 | `[num_hidden_blocks, seq_len]` |
| `gemm1_weights` | FP8 E4M3FN | `[num_local_experts, gemm1_out_size, hidden_size]` |
| `gemm1_weights_scale` | FP32 | `[num_local_experts, num_gemm1_out_blocks, num_hidden_blocks]` |
| `gemm2_weights` | FP8 E4M3FN | `[num_local_experts, hidden_size, intermediate_size]` |
| `gemm2_weights_scale` | FP32 | `[num_local_experts, num_hidden_blocks, num_intermediate_blocks]` |
| `local_expert_offset` | INT32 | scalar |
| `routed_scaling_factor` | FP32 | scalar |

The output is BF16 `[seq_len, hidden_size]`.

## Reference Semantics

The official reference uses sigmoid routing. It adds `routing_bias` only for expert selection, scores each of eight groups by the sum of its two largest selection scores, retains four groups, and selects eight global experts. Combine weights come from the corresponding unbiased sigmoid values, normalized per token and multiplied by `routed_scaling_factor`.

Each rank computes only global experts in its 32-expert local interval. GEMM1 produces the concatenated 4096-column W13 result; the reference splits it into two 2048-column halves, applies SwiGLU, performs GEMM2, and accumulates each expert result with its routing weight.

The reference semantics specify an operation, not a GPU launch count or required tcgen05/TMEM decomposition.

## Official Evaluation Contract

At starter-kit commit `75ccd05cafceb0fd1f86be4cd0f2117249463c66`, `EVALUATION.md` records:

| Field | Value |
| --- | --- |
| GPU | Bare-metal NVIDIA B200 (`sm_100a`) |
| Locked clocks | `nvidia-smi -ac 3996,1965` |
| Container | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| CUDA | 13.2 |
| Python | 3.12 |
| PyTorch | 2.12.0+cu132 |
| Triton | 3.6.0 |
| MoE correctness | `atol=1`, `rtol=0.3`, required matched ratio `0.9` |

The baseline solution is `flashinfer_wrapper_9sdjf3`. For the single-definition MoE track, the score is the arithmetic mean of per-workload `baseline_latency / candidate_latency`; any failing workload zeros that kernel's score.

## Performance Boundary

No framework TFLOPS/latency/launch-count table is retained from this source. The current definition page exposes per-solution, per-`seq_len` traces, but a reusable performance record would need a named solution and revision, exact workload, environment, timed region, synchronization, warmup, repetitions, statistic, and variance. The former local 1262-TFLOPS record did not supply those fields.

## Contest Dates and Results

The current organizer page records a January 22, 2026 public launch, February 9 baseline release, April 24 kernel submission deadline, May 1 writeup deadline, May 12 winner notification, and May 22 award ceremony. It now lists the actual Track A winners separately for agent-assisted and full-agent approaches; stale pre-contest AI-baseline rankings are not reproduced here.

## Primary Sources

- [Organizer page](https://mlsys26.flashinfer.ai/)
- [Exact benchmark definition](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048)
- [Starter kit at `75ccd05`](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/tree/75ccd05cafceb0fd1f86be4cd0f2117249463c66)
- [Evaluation contract at `75ccd05`](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/75ccd05cafceb0fd1f86be4cd0f2117249463c66/EVALUATION.md)
- [FlashInfer reference template at `7f614b8`](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/trace/templates/moe.py)
