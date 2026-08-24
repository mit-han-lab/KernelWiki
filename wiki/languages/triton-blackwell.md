---
id: lang-triton
title: Triton on Blackwell
type: language
tags: [triton, attention, moe, gated-delta-net]
related: [kernel-nsa, kernel-gated-delta-net, kernel-fused-moe, lang-cute-dsl]
sources: [doc-triton-3.6-blackwell, pr-vllm-34597, pr-sglang-22079, pr-sglang-21019]
reproducibility: snippet
architectures: [sm100, sm90]
confidence: verified
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-triton-3.6-blackwell
  - evidence_type: upstream-code
    source_id: pr-vllm-34597
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
blackwell_relevance: Triton 3.6 release notes document Blackwell TMEM, tcgen05, warp-specialization, scaled-MMA, and initial multi-CTA backend work; actual lowering and performance remain shape- and surface-dependent.
---

# Triton on Blackwell

Triton 3.6 includes compiler infrastructure for Blackwell TMEM and `tcgen05`, plus Gluon multi-CTA and scaled-MMA work. This makes the earlier blanket statement that Triton has no TMEM or `tcgen05` path obsolete for 3.6. It does not mean every `tl.dot` kernel lowers identically or performs like a specialized CuTe/CUDA kernel.

## Verified boundary

- The official 3.6 release notes establish the compiler/backend features.
- vLLM PR 34597 provides an inspectable downstream `@triton.jit` MLA decode kernel using `tl.dot` on a source page classified for SM100.
- SGLang PR 22079 is another retained attention example with `tl.dot` on SM100/SM90.
- SGLang PR 21019 is a memory-rearrangement Triton kernel and therefore must not be cited as evidence of tensor-core lowering.

The retained downstream sources demonstrate use of Triton kernels on Blackwell paths. They do not contain an in-repository PTX dump proving a particular `tcgen05.mma` form for every call below.

## Verbatim call-site excerpt

The following non-contiguous call sites are copied from the vLLM PR 34597 artifact to show the relevant Triton surface; consult the pinned full file for control flow and shapes:

```python
qk = tl.dot(q, k.to(q.dtype))
qk += tl.dot(qpe, kpe.to(qpe.dtype))
acc += tl.dot(p.to(v.dtype), v)
```

Pinned file: [`triton_decode_attention.py`](../../artifacts/prs/vllm/PR-34597/key-files/vllm/v1/attention/ops/triton_decode_attention.py).

## Selection guidance

Use Triton when its programming model and supported operations fit the kernel and validate the generated code on every target SKU. For peak-sensitive compute kernels, compare against appropriate vendor/CuTe/CUDA implementations rather than assuming either backend wins. For memory-bound or irregular kernels, measure launch, memory, and scheduling costs separately.

## Evidence ledger

The version claim is pinned in [`data/version-claims.yaml`](../../data/version-claims.yaml), and the retained-source boundary is summarized in [`data/triton-3.6-evidence.md`](../../data/triton-3.6-evidence.md).
