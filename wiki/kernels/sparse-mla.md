---
id: kernel-sparse-mla
title: Sparse MLA
type: kernel
architectures: [sm100, sm90]
tags: [sparse-attention, mla, fp8, attention, decode, prefill]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, mla, attention, decode, prefill]
languages: [cuda-cpp, python]
related: [kernel-flashmla, kernel-nsa]
sources: [blog-flashmla, blog-vllm-deepseek-v3-sparse, doc-nsa]
performance_claims:
  - gpu: B200
    dtype: bf16
    shape: sparse MLA prefill; upstream benchmark suite
    metric: TFLOPS
    value: 1450
    source_id: blog-flashmla
    source_locator: "FlashMLA README, Test & benchmark MLA prefill (Sparse), plus flash_mla/flash_mla_interface.py::flash_mla_sparse_fwd dtype contract"
blackwell_relevance: FlashMLA supports token-sparse MLA decode and prefill on SM100; the caller supplies selected token indices.
---

# Sparse MLA

FlashMLA's sparse paths compute attention only for caller-supplied token
indices. Selection/indexer construction is upstream of the attention kernel and
must not be conflated with the sparse MLA call itself.

For sparse prefill, the upstream README defines the following equivalent
PyTorch dataflow:

```python
focused_kv = kv[indices]
P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * math.log2(math.e)
max_logits = P.max(dim=-1)
lse = log2sumexp2(P, dim=-1, base=2)
S = exp2(P - lse)
out = S @ focused_kv
```

The full reference also specifies shapes, a single KV head, invalid-index
handling, and returned `(out, max_logits, lse)` values. The excerpt is not a
drop-in replacement for its test.

Sparse decode uses an FP8-with-scale KV-cache format and BF16 attention
arithmetic. Sparse prefill's B200 README maximum is 1450 TFLOPS; it is a
source-reported result for that suite, not evidence for an assumed top-k value,
page size, or separate Lightning Indexer implementation. Those unsupported
constants and the former hand-written tcgen skeletons were removed.
