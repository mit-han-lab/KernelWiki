---
id: lang-triton
title: "Triton on Blackwell"
type: language
tags: [triton, attention, moe, gated-delta-net]
related: [kernel-nsa, kernel-gated-delta-net, kernel-fused-moe, lang-cute-dsl]
sources: [doc-triton-3.3-blackwell, doc-triton-3.6-blackwell, pr-vllm-34597, pr-vllm-29339, pr-sglang-22079, pr-sglang-21019, pr-sglang-5390, pr-sglang-21595]
reproducibility: snippet
architectures: [sm100, sm100a, sm120, sm90]
confidence: verified
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-triton-3.6-blackwell
  - evidence_type: upstream-code
    source_id: doc-triton-3.3-blackwell
  - evidence_type: upstream-code
    source_id: pr-vllm-34597
version_sensitive:
  id: vs-triton-3.3-blackwell-tcgen05
blackwell_relevance: "Native TCGen5/TMEM compiler support enters between Triton v3.2.0 and v3.3.0. Later releases expand the user-visible surfaces, while exact instruction selection remains configuration-dependent."
---

## Verified release boundary

Native Blackwell TCGen5/TMEM compiler support enters between Triton v3.2.0 and v3.3.0. In the exact tag comparison, the corresponding TCGen5 MMA, TMEM, and MMAv5-lowering symbols are absent at v3.2.0 (`9641643d`) and present at v3.3.0 (`819e9c8c`). The v3.3 conversion suite checks concrete `tcgen05.mma`, commit, TMEM, scaled-MMA, and `cta_group::2` output. See the pinned files in [`doc-triton-3.3-blackwell`](../../sources/docs/triton-3.3-blackwell.md).

| Release | Evidence-scoped milestone |
|---|---|
| v3.2.0 | Checked negative side of the native-backend boundary. |
| v3.3.0 | First checked tag after v3.2.0 with TCGen5/TMEM operations, allocation and lowering passes, and concrete conversion tests. |
| v3.5.0 | Tagged tree includes an explicit Gluon TCGen5/TMEM tutorial and the Blackwell block-scaled matmul tutorial; release notes also document warp-specialization work. |
| v3.6.0 | Generalizes TCGen5 copies/layouts and MMA handling, advances aref-style warp specialization, and adds initial multi-CTA/2-CTA Gluon work. It is not the introduction boundary. |
| v3.7.0 / v3.7.1 | v3.7.0 continues 2-CTA, multicast, and TMA work; v3.7.1 is a two-regression patch with no advertised new API or feature. |

The compiler evidence proves that native paths exist. It does not prove that every plain `tl.dot` shape selects TCGen5: instruction selection can depend on architecture, dtype, shape, layout, and compiler configuration. A claim about one kernel's emitted instruction needs matching IR or PTX.

## User-visible surfaces

The v3.5.0 tree provides two useful pinned examples:

- [`python/tutorials/gluon/06-tcgen05.py`](https://github.com/triton-lang/triton/blob/c3c476f357f1e9768ea4e45aa5c17528449ab9ef/python/tutorials/gluon/06-tcgen05.py) explicitly allocates, loads, and stores TMEM and invokes TCGen5 MMA through Gluon.
- [`python/tutorials/10-block-scaled-matmul.py`](https://github.com/triton-lang/triton/blob/c3c476f357f1e9768ea4e45aa5c17528449ab9ef/python/tutorials/10-block-scaled-matmul.py) demonstrates `tl.dot_scaled` for Blackwell block-scaled matmul.

Triton v3.6.0 expands those foundations with broader layouts and copies and initial multi-CTA/2-CTA Gluon support. “Initial” is deliberate: v3.7.0 contains follow-on end-to-end 2-CTA, multicast, and TMA changes. See [`doc-triton-3.6-blackwell`](../../sources/docs/triton-3.6-blackwell.md) and the compact release matrix in [`data/triton-3.6-evidence.md`](../../data/triton-3.6-evidence.md).

## What downstream code establishes

- [`pr-vllm-34597`](../../sources/prs/vllm/PR-34597.md), pinned at `a1257fd1`, adds FP8 KV-cache handling to the Triton MLA decode backend. Its verbatim kernel contains `tl.dot`, but no target guard, Triton-version requirement, TCGen5/TMEM name, or emitted PTX. The primary PR specifically motivates the backend as the MLA option on SM120; it is not an SM100-only lowering demonstration.
- [`pr-vllm-29339`](../../sources/prs/vllm/PR-29339.md), pinned at `c17610e2`, gates MXFP4 `triton_kernels` dispatch to SM90 and SM100. It changes dispatch logic, not a kernel or compiler lowering.
- [`pr-sglang-21019`](../../sources/prs/sglang/PR-21019.md), pinned at `5bdc07d9`, provides a Triton GatedDeltaNet projection rearrangement using loads and stores, with no `tl.dot`.
- [`pr-sglang-22079`](../../sources/prs/sglang/PR-22079.md), pinned at `5638d40f`, provides an extend-attention Triton kernel with real `tl.dot` operations. The source does not contain an emitted-PTX witness for a particular MMA instruction.

These are verified downstream Triton examples. They must not be combined with the separate compiler-version evidence to infer an unobserved lowering.

## Scoped ecosystem results

SGLang [`pr-sglang-5390`](../../sources/prs/sglang/PR-5390.md) reports 10,447.34 total tok/s for its CUTLASS MLA run and 8,227.35 total tok/s for its Triton run, 26.98% higher under the PR's recorded DeepSeek-R1, 3,000-prompt, TP8/DP8, float16, 1,000-input/1,000-output-token scope. This is one reported comparison, not a universal language ranking.

SGLang [`pr-sglang-21595`](../../sources/prs/sglang/PR-21595.md) changes the SM100 datacenter multimodal-attention default from `triton_attn` to FA4. That routing decision is likewise workload- and architecture-scoped.

The live [FlashInfer-Bench leaderboard](https://bench.flashinfer.ai/), retrieved 2026-08-08, reports these author rows across 660 workloads each:

| Author | Average speedup | Resolved |
|---|---:|---:|
| Gemini 2.5 Pro | 0.628x | 73.1% |
| GPT-5 | 0.467x | 92.3% |
| Claude Opus 4.1 | 0.456x | 73.1% |

The leaderboard does not attach those rows to a Triton release or identify them as a Triton-only language subset.

## Launch overhead and CUDA Graphs

The CUDA Programming Guide explains that CPU setup and launch overhead can be significant for short kernels and that CUDA Graphs reduce repeated launch costs by preparing work in advance. At vLLM commit `a1257fd1`, `FULL_AND_PIECEWISE` is the v1 CUDA-graph default, and the GatedDeltaNet backend implements uniform-batch graph support plus decode-only full graph capture. This is a launch-path observation, independent of which MMA instruction a Triton kernel selects.

## Provenance-pinned examples

Each linked file is stored verbatim under its bundle's recorded merge SHA:

| File | Verified role |
|---|---|
| [`triton_decode_attention.py`](../../artifacts/prs/vllm/PR-34597/key-files/vllm/v1/attention/ops/triton_decode_attention.py) | Triton MLA decode kernels with `tl.dot`; PR 34597 adds FP8 cache handling. |
| [`triton_mla.py`](../../artifacts/prs/vllm/PR-34597/key-files/vllm/v1/attention/backends/mla/triton_mla.py) | Backend wrapper and supported FP8 cache dtypes. |
| [`format_conversion.py`](../../artifacts/prs/flashinfer/PR-1025/key-files/flashinfer/triton/format_conversion.py) | Triton FP8/FP16 format-conversion kernels. |
| [`norm.py`](../../artifacts/prs/sglang/PR-20910/key-files/python/sglang/jit_kernel/norm.py) | Triton normalization kernels. |
| [`gdn_fused_proj.py`](../../artifacts/prs/sglang/PR-21019/key-files/python/sglang/jit_kernel/triton/gdn_fused_proj.py) | Triton GatedDeltaNet projection rearrangement using loads/stores. |
| [`extend_attention.py`](../../artifacts/prs/sglang/PR-22079/key-files/python/sglang/srt/layers/attention/triton_ops/extend_attention.py) | Triton extend-attention kernels containing `tl.dot`. |

The complete tracked PR universe and its captured/skipped flags are recorded in [`data/triton-universe.yaml`](../../data/triton-universe.yaml); this page does not duplicate that changing count.

This verbatim excerpt from the pinned vLLM decode kernel demonstrates the page's deliberately limited downstream claim—FP8 rescaling followed by `tl.dot`, without proving a particular emitted MMA instruction:

```python
if k.dtype.is_fp8():
    k = (k.to(tl.float32) * ks).to(q.dtype)
qk = tl.dot(q, k.to(q.dtype))
if BLOCK_DPE > 0:
    offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs
```
