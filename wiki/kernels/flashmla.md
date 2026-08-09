---
id: kernel-flashmla
title: FlashMLA — Multi-head Latent Attention
type: kernel
architectures:
- sm100
- sm90
tags:
- mla
- attention
- decode
- prefill
- fp8
- sparse-attention
confidence: source-reported
reproducibility: snippet
kernel_types:
- mla
- attention
- decode
- prefill
- sparse-attention
languages:
- cuda-cpp
- python
related:
- hw-tcgen05-mma
- hw-tmem
- kernel-nsa
sources:
- blog-flashmla
- doc-deepseek-v2-mla
- pr-cutlass-2466
- pr-cutlass-2472
performance_claims: []
blackwell_relevance: FlashMLA supports sparse MLA decode and prefill on SM100 and
  bundles NVIDIA's dense MHA prefill path; its pinned snapshot requires CUDA 12.9+
  for SM100.
artifact_dir: artifacts/kernels/flashmla
---

# FlashMLA -- Multi-head Latent Attention

## Scope

FlashMLA is DeepSeek's attention-kernel library for DeepSeek-V3 and DeepSeek-V3.2-Exp. At pinned commit [`71c7379`](https://github.com/deepseek-ai/FlashMLA/tree/71c737929f2567bd0a094ae140f8f60f390b1232), the library contains MLA-mode decode and sparse-prefill operators plus a dense **MHA** prefill operator contributed for SM100. The repository's term “MLA mode” distinguishes MQA-shaped `d_qk=576, d_v=512` kernels from MHA-shaped `d_qk=192/128, d_v=128` kernels; it does not mean every operator in the package is MLA.

## MLA Cache Reduction Versus a Concrete Cache ABI

The DeepSeek-V2 paper describes the model-level reduction in elements cached per layer and token:

- MHA: `2 * n_h * d_h`
- MLA: `d_c + d_h^R`, approximately `4.5 * d_h` for its `d_c=4*d_h` and `d_h^R=d_h/2` configuration

These formulas are independent of storage dtype and layer count. They should not be converted to whole-model KB/token figures without naming those additional assumptions.

FlashMLA's **656-byte** layout is narrower: it is the DeepSeek-V3-family FP8 sparse-decode cache ABI, not the definition of an MLA cache. One token contains:

| Region | Representation | Bytes |
|---|---|---:|
| NoPE latent data | 512 `float8_e4m3` values | 512 |
| NoPE group scales | four `float32` values, one per 128 values | 16 |
| RoPE data | 64 unquantized `bfloat16` values | 128 |
| **Total** |  | **656** |

Dense decode uses a BF16 cache. The pinned quantization tests also contain a separate 512-dimensional sparse layout, so code must select the model/layout contract rather than assuming 656 bytes universally. `page_block_size` is taken from the cache tensor; 64 is a test default, not a fixed API rule.

## Supported Paths at `71c7379`

| Operator | Architecture | Attention mode | Documented cache/input format |
|---|---|---|---|
| Dense decode | SM90 | MQA (`576/512`) | BF16 paged KV |
| Sparse decode | SM90, SM100 | MQA (`576/512`) | FP8 KV, dequantized for BF16 MMA; BF16 output |
| Dense prefill | SM100 | MHA (`192/128` or `128/128`) | BF16 Q/K/V |
| Sparse prefill | SM90, SM100 | MQA | BF16 Q and KV |

CUDA 12.8 or newer and PyTorch 2.0 or newer are required; the pinned README requires CUDA 12.9 or newer for SM100.

## Sparse Contracts

Sparse decode receives `indices[batch, s_q, topk]`. Each nonnegative value already encodes a physical page and offset:

```text
encoded = physical_page * page_block_size + offset_in_page
```

Because the physical page is already encoded, sparse decode does not use `block_table`; `-1` marks an invalid entry. The kernel consumes these indices but does not produce the top-k selection, so an indexing stage outside the attention call must supply them.

Sparse prefill is a different interface. It receives BF16 `q[s_q,h_q,d_qk]`, BF16 `kv[s_kv,h_kv,d_qk]`, and `indices[s_q,h_kv,topk]`; it has no batch dimension, requires `h_kv=1` in the documented equivalence, and accepts `-1` or values at least `s_kv` as invalid. It returns `(out, max_logits, lse)`.

## Source-Reported Performance

The following are maxima reported by the pinned first-party README. They were not reproduced here, and the README does not provide complete shape, timing, sample-count, or variance cells, so they are intentionally excluded from structured `performance_claims`.

| Operator | Environment stated by source | Precision scope | Author-reported observation |
|---|---|---|---|
| Dense MLA decode | H800 SXM5, CUDA 12.8 | BF16 cache | Up to 3000 GB/s in a memory-bound configuration; up to 660 TFLOPS in a compute-bound configuration |
| Sparse MLA decode | H800 SXM5, CUDA 12.8 | FP8 KV, BF16 MMA | 410 TFLOPS in a compute-bound configuration |
| Sparse MLA decode | B200; software version not stated in the row | FP8 KV, BF16 MMA | Up to 350 TFLOPS; the author says it was not well optimized |
| Dense MHA prefill | B200; NVIDIA-reported | BF16 inputs | Up to 1460 TFLOPS forward and 1000 TFLOPS backward |
| Sparse MLA prefill | H800 SXM5, CUDA 12.8 | BF16 inputs | Up to 640 TFLOPS forward |
| Sparse MLA prefill | B200, CUDA 12.9 | BF16 inputs | Up to 1450 TFLOPS forward |

The numbers compare different operators, phases, shapes, precision scopes, and machines. In particular, `1460` is dense MHA prefill, not a replacement for the `660` dense-MLA-decode observation.

## SM100 Implementation Notes

The pinned DeepSeek SM100 sources use TMA, `tcgen05` tensor-core operations, and TMEM in specialized sparse-decode/prefill and dense-MHA-prefill code. Those mechanisms are implementation-specific: they do not make the CUTLASS and FlashInfer files in this repository copies of DeepSeek FlashMLA.

The local [`full/`](../../artifacts/kernels/flashmla/full/) bundle contains two byte-verified **adjacent implementations**:

- NVIDIA CUTLASS Example 77 MLA forward at merge `9baa06dd`
- FlashInfer's SM100 FMHA-MLA header at commit `9a05c92a`

Their exact per-file origins are recorded in `full/PROVENANCE.yaml`. The [`variants/`](../../artifacts/kernels/flashmla/variants/) directory contains a small KernelWiki-derived layout/index helper, explicitly marked as non-upstream. For the DeepSeek implementation itself, use commit `71c7379` linked above.

## Selection and Validation Checklist

- Match the operator family, `d_qk/d_v`, architecture, CUDA version, cache dtype, and index shape exactly.
- Treat the 656-byte layout as a V3-family FP8 sparse-decode ABI, not a generic MLA property.
- Generate sparse indices before invoking FlashMLA and validate invalid-index/page encoding rules.
- Benchmark the target decode or prefill workload; do not transfer TFLOPS or bandwidth across the table's distinct regimes.
- Validate output and LSE against the repository reference before relying on throughput.

## Sources and Local Query

- [DeepSeek FlashMLA at audited commit `71c7379`](https://github.com/deepseek-ai/FlashMLA/tree/71c737929f2567bd0a094ae140f8f60f390b1232)
- [DeepSeek-V2 MLA paper, v5](https://arxiv.org/html/2405.04434v5)
- [CUTLASS PR 2466, SM100 MLA-shape backward](https://github.com/NVIDIA/cutlass/pull/2466)
- [CUTLASS PR 2472, SM100 MLA-shape forward](https://github.com/NVIDIA/cutlass/pull/2472)

Query the page and its labeled artifacts with:

```bash
conda run -n base python scripts/get_page.py kernel-flashmla --include-code
```

The mode-specific byte arithmetic can be checked without a GPU:

```python
# KernelWiki-derived contract check; not upstream FlashMLA code.
def v3_fp8_sparse_bytes() -> int:
    nope_bytes = 512
    scale_bytes = 4 * 4
    rope_bytes = 64 * 2
    return nope_bytes + scale_bytes + rope_bytes

assert v3_fp8_sparse_bytes() == 656
```
