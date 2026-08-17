---
id: lang-triton
title: "Triton on Blackwell"
type: language
tags: [triton, attention, moe, gated-delta-net]
related: [kernel-nsa, kernel-gated-delta-net, kernel-fused-moe, lang-cute-dsl]
sources: [doc-triton-3.7-blackwell, doc-triton-3.6-blackwell, pr-vllm-34597, pr-vllm-29339, pr-sglang-22079, pr-sglang-21019, pr-sglang-5390, pr-sglang-21595, blog-nsa, blog-gated-delta-net, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm90]
confidence: verified
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-triton-3.7-blackwell
  - evidence_type: official-doc
    source_id: doc-triton-3.6-blackwell
  - evidence_type: upstream-code
    source_id: pr-sglang-22079
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
blackwell_relevance: "Triton's source-defined tcgen05/TMEM dialect operations predate 3.6 and are present by 3.3. Triton 3.6 materially generalized and hardened SM100 lowering and exposed broader warp-specialization, block-scaled, and Gluon/2CTA paths; 3.7 added follow-on work."
---

# Triton on Blackwell

## Versioned status

Triton's release/3.3.x source already defines `tc_gen5_mma`, `tc_gen5_mma_scaled`, and TMEM allocation/load/store/copy operations. Triton 3.6.0, released on 2026-01-21, materially generalized and hardened those Blackwell lowering paths and added broader user-facing warp-specialization, block-scaled, and Gluon/2CTA work. Triton 3.7.0 was released on 2026-05-07, before this repository's 2026-05-20 refresh cutoff, and added material follow-on work for 2CTA, TMA/`tcgen05.mma` multicast, validation/fixes, and nested-loop warp specialization.

Consequently, the old blanket statement “Triton emits WGMMA and cannot use TMEM on Blackwell” is false. Version-specific claims must distinguish the early source-defined operations from the broader, more mature 3.6+ paths.

## Verified surfaces

- The official persistent-matmul tutorial uses descriptor/TMA inputs, `tl.range(..., warp_specialize=True)`, and `tl.dot`; the tutorial marks the warp-specialized mode as Blackwell-specific at that release.
- `tl.dot_scaled` and the Blackwell dialect provide block-scaled tensor-core paths and TMEM-related operations for supported formats/configurations.
- Gluon exposes explicit warp-specialization and multi-CTA/cluster surfaces; release notes describe the initial 2CTA work and later 3.7 hardening.
- Captured downstream code includes an SM100/GB200 Triton attention path in SGLang PR 22079 and a separate SM120 MLA decode path in vLLM PR 34597.

These facts establish that the infrastructure and adoption are real. They do **not** establish that every arbitrary `tl.dot`, dtype, or shape emits the same `tcgen05` form, nor that it matches CUTLASS/CuTe performance.

## Practical selection

Triton is attractive for portable/custom memory-bound transforms, rapid iteration, and attention/recurrent algorithms already expressed in its programming model. CUTLASS, CuTe DSL, or vendor libraries may lead for a compute-bound shape with a mature specialized kernel. Make this a dispatch/benchmark decision, not a language hierarchy.

SGLang PR 5390 reports 10,447.34 total tok/s for its `cutlass_mla` backend versus 8,227.35 tok/s for the Triton backend in its 3,000-request DeepSeek-R1 serving command—about 1.27×, or 27% higher. The PR does not provide the exact GPU SKU, clocking, or complete software environment, so this is a scoped author-reported serving comparison, not a general Triton-versus-CUTLASS result. PR 21595 separately changes one Blackwell multimodal-attention default from `triton_attn` to FA4; a routing choice is not itself a benchmark.

## Minimal semantic example

```python
@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.program_id(0) * n + tl.arange(0, n)
    values = tl.load(x + offsets) + tl.load(y + offsets)
    tl.store(out + offsets, values)
```

Real attention and recurrent kernels need shape-correct block pointers, masks, reductions, state ordering, and autotuning. Pseudocode that loads an entire multidimensional recurrent state into one Triton program or uses Python `range(topk)` on a runtime value is not a valid reference kernel.

## Evidence boundaries

No captured downstream PTX dump proves a universal mapping of plain `tl.dot` to `tcgen05`. Likewise, historical FlashInfer-Bench model tables from a Triton 3.5-era snapshot are not evidence for Triton 3.6/3.7 and are omitted here pending a pinned rerun.

## Verbatim downstream examples

| File | Evidence |
|---|---|
| [`triton_decode_attention.py`](../../artifacts/prs/vllm/PR-34597/key-files/vllm/v1/attention/ops/triton_decode_attention.py) | SM120 vLLM MLA decode with FP8 KV-cache changes and `tl.dot` paths |
| [`triton_mla.py`](../../artifacts/prs/vllm/PR-34597/key-files/vllm/v1/attention/backends/mla/triton_mla.py) | Backend wrapper for that decode kernel |
| [`extend_attention.py`](../../artifacts/prs/sglang/PR-22079/key-files/python/sglang/srt/layers/attention/triton_ops/extend_attention.py) | SGLang SM100/SM90 attention with `tl.dot` |
| [`gdn_fused_proj.py`](../../artifacts/prs/sglang/PR-21019/key-files/python/sglang/jit_kernel/triton/gdn_fused_proj.py) | Memory-layout/fused projection kernel; not a matmul-lowering proof |
| [`format_conversion.py`](../../artifacts/prs/flashinfer/PR-1025/key-files/flashinfer/triton/format_conversion.py) | FP8/FP16 format conversion |

The full classified universe is in `data/triton-universe.yaml`; `data/triton-3.6-evidence.md` records the evidence limits.
