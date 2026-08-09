---
id: kernel-fused-moe
title: FlashInfer Track A FP8 Block-Scale MoE
type: kernel
architectures:
- sm100
- sm100a
tags:
- moe
- fp8
- block-scale
- grouped-gemm
- kernel-fusion
confidence: source-reported
reproducibility: snippet
kernel_types:
- moe
- grouped-gemm
- fused-kernel
languages:
- cuda-cpp
- cute-dsl
- triton
related:
- kernel-grouped-gemm
- kernel-deepgemm
- technique-fine-grained-quantization
- technique-tile-scheduling
sources:
- contest-flashinfer-track-a
performance_claims: []
blackwell_relevance: The MLSys 2026 Track A definition and official evaluation
  target NVIDIA B200 (sm_100a); this page documents the logical benchmark
  contract, not a particular launch decomposition.
artifact_dir: artifacts/kernels/fused-moe
---

# FlashInfer Track A FP8 Block-Scale MoE

## Scope

FlashInfer's MLSys 2026 contest calls Track A **Fused MoE** with FP8 support and targets NVIDIA B200. Its exact benchmark definition is `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`. The definition says that DeepSeek-style routing and two grouped GEMMs are included. That operation-level scope does **not** establish that an implementation uses one GPU launch.

## Fixed Benchmark Geometry

| Field | Value | Meaning |
| --- | ---: | --- |
| `seq_len` | variable | Number of input tokens; this is not labeled request batch size |
| global experts | 256 | Width of `routing_logits` |
| local experts | 32 | Experts whose weights are resident on one EP rank |
| expert parallelism | 8 | `256 / 32` ranks in the published definition |
| `top_k` | 8 | Selected global experts per token |
| `n_group` | 8 | Groups of 32 global experts |
| `topk_group` | 4 | Groups retained before global top-k selection |
| hidden size | 7168 | Input and output width |
| intermediate size | 2048 | Per-expert SwiGLU width |
| GEMM1 output | 4096 | Concatenated W13 output, `2 * 2048` |
| scale block | 128 | Fixed granularity for this DeepSeek-FP8 trace |

The `e32` suffix means **32 local experts**, not 32 total experts.

## Tensor Contract

For `T = seq_len`, the published benchmark signature is:

| Input | Dtype | Shape |
| --- | --- | --- |
| `routing_logits` | FP32 | `[T, 256]` |
| `routing_bias` | BF16 | `[256]` |
| `hidden_states` | FP8 E4M3FN | `[T, 7168]` |
| `hidden_states_scale` | FP32 | `[56, T]` |
| `gemm1_weights` | FP8 E4M3FN | `[32, 4096, 7168]` |
| `gemm1_weights_scale` | FP32 | `[32, 32, 56]` |
| `gemm2_weights` | FP8 E4M3FN | `[32, 7168, 2048]` |
| `gemm2_weights_scale` | FP32 | `[32, 56, 16]` |
| `local_expert_offset` | INT32 | scalar |
| `routed_scaling_factor` | FP32 | scalar |

The output is BF16 `[T, 7168]`. These scale tensors are explicit storage inputs; this page does not infer their runtime cost from their existence.

This derived helper makes the fixed shape arithmetic executable without pretending to encode FlashInfer's physical layouts:

```python
def track_a_shapes(tokens: int) -> dict[str, tuple[int, ...]]:
    assert tokens > 0
    return {
        "routing_logits": (tokens, 256),
        "hidden_states": (tokens, 7168),
        "hidden_states_scale": (7168 // 128, tokens),
        "gemm1_weights": (32, 2 * 2048, 7168),
        "gemm1_weights_scale": (32, (2 * 2048) // 128, 7168 // 128),
        "gemm2_weights": (32, 7168, 2048),
        "gemm2_weights_scale": (32, 7168 // 128, 2048 // 128),
    }


assert track_a_shapes(7)["gemm1_weights_scale"] == (32, 32, 56)
```

At FlashInfer commit `7f614b86470180bab2d22e36fd1775791c6bf3e6`, the corresponding public entry point is `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`. Its complete call includes the eight tensors above plus `num_experts=256`, `top_k=8`, `n_group=8`, `topk_group=4`, `intermediate_size=2048`, the local expert interval, the routed scaling factor, and DeepSeek-V3 routing mode.

## Reference Semantics

The official reference performs these steps:

1. Convert routing logits to `s = sigmoid(logits)` and form selection scores `s + routing_bias`.
2. Reshape 256 scores into eight groups of 32. Sum the top two selection scores in each group, then retain four groups.
3. Select eight global experts from the retained groups using the biased selection scores.
4. Form combine weights from the **unbiased** sigmoid values for those eight experts, normalize per token, and multiply by `routed_scaling_factor`.
5. For global expert IDs in `[local_expert_offset, local_expert_offset + 32)`, dequantize the relevant activation and weight blocks, compute one W13 projection, split its 4096 columns into two 2048-column halves, apply SwiGLU, and compute W2.
6. Accumulate each local expert result into the token output using that expert's combine weight. Experts outside the local interval contribute nothing on that rank.

The W13 representation permits one logical `A @ W13.T` followed by a split. It does not require two separately allocated TMEM accumulators, and the benchmark contract does not prescribe a tcgen05 instruction sequence.

## Small Derived Reference

[`01-routing-plus-fusion-skeleton.py`](../../artifacts/kernels/fused-moe/variants/01-routing-plus-fusion-skeleton.py) is a CPU-checkable, parameterized reference for grouped selection and local W13/SwiGLU/W2 accumulation. It is derived KernelWiki code, not an optimized kernel and not an upstream contest solution.

## Official Evaluation Boundary

The starter-kit evaluation document at commit `75ccd05cafceb0fd1f86be4cd0f2117249463c66` records:

- bare-metal NVIDIA B200 (`sm_100a`) with clocks locked to `3996,1965`;
- container `flashinfer/flashinfer-ci-cu132:20260401-2c675fb`;
- CUDA 13.2, Python 3.12, PyTorch 2.12.0+cu132, and Triton 3.6.0;
- correctness gates `atol=1`, `rtol=0.3`, and matched ratio `0.9` for the MoE command; and
- an arithmetic mean of per-workload `FlashInfer baseline latency / candidate latency` as the single-definition MoE score.

The current primary sources do not support the former framework TFLOPS/latency table, its launch counts, or the structured 1262-TFLOPS record. No performance result is retained here. The official trace axis is `seq_len`; relabeling its endpoints as prefill/decode or batch size requires a separate serving experiment.

## Implementation Risks That Must Be Measured

- Routing produces different token counts per expert, hence different grouped-GEMM M dimensions. A scheduler can mitigate that imbalance; no single bottleneck is universal across token counts and tactics.
- Small M or partially filled GEMM tiles can reduce utilization. This is a workload-dependent risk, not a retained performance result.
- At the pinned FlashInfer revision, runtime autotuning enumerates valid tactics and selects GEMM1 and GEMM2 tactics over token buckets. Record the exact revision and tuning state in any measurement.
- The FP8 element type alone is insufficient for compatibility. Routing method, global/local expert mapping, scale dtype and granularity, tensor layout, activation, and output semantics must all match.
- CUDA-graph layout requirements are backend-specific. DeepGEMM's masked grouped layout is one documented decode case when the CPU does not know expert token counts; it is not a universal graph requirement.
- TMA alignment depends on the tensor-map configuration. CUDA 13.2.1 requires a 64-byte-aligned tensor-map object and ordinarily a 16-byte-aligned global base, with additional requirements for selected interleave, dtype, and swizzle modes; it does not require every expert pointer to be 128-byte aligned.

## Adjacent Local Artifacts

The files under [`full/`](../../artifacts/kernels/fused-moe/full/) are mixed **adjacent references**, not a full implementation of this benchmark:

- `vllm-PR-23696-dual-gemm.patch` is the aggregate merged diff for vLLM's MXFP4-weight fused expert-compute integration (BF16 activations on Hopper and MXFP8 activations on Blackwell).
- `flashinfer_cutedsl.py` is byte-identical to an SGLang FP4 CuteDSL runner at commit `c554dc5c64b661f2c53225b03a76359eaddc39e4`.
- `moe-grouped-gemm-launch.cpp` is local illustrative pseudocode for varying per-expert M segments.

None supplies the exact Track A FP8 implementation or substantiates performance/launch claims. Their hashes and exact scopes are recorded in the bundle provenance.

## Sources

- [MLSys 2026 FlashInfer contest](https://mlsys26.flashinfer.ai/)
- [Exact FP8 MoE benchmark definition](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048)
- [Starter-kit evaluation contract at `75ccd05`](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/75ccd05cafceb0fd1f86be4cd0f2117249463c66/EVALUATION.md)
- [FlashInfer trace reference at `7f614b8`](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/trace/templates/moe.py)
- [FlashInfer API implementation at `7f614b8`](https://github.com/flashinfer-ai/flashinfer/blob/7f614b86470180bab2d22e36fd1775791c6bf3e6/flashinfer/fused_moe/core.py)
- [CUDA 13.2.1 tensor-map requirements](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [CUTLASS efficient-GEMM small-dimension discussion](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [DeepGEMM grouped-layout documentation at `891d57b4`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/README.md)

Query the page and its explicitly labeled artifacts with:

```bash
python3 scripts/get_page.py kernel-fused-moe --include-code
```
