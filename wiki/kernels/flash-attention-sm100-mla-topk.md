---
id: kernel-flash-attention-sm100-mla-topk
title: FlashAttention SM100 MLA TopK Sparse Forward
type: kernel
architectures:
- sm100
tags:
- attention
- flash-attention
- mla
- sparse-attention
- tma
- tile-scheduling
- top-k-selection
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- flash-attention
- mla
- sparse-attention
- topk
languages:
- cute-dsl
- python
related:
- kernel-flash-attention-4
- kernel-sparse-mla
- technique-tile-scheduling
- technique-external-source-map-research
sources:
- pr-flash-attention-2441
- pr-flash-attention-1236
performance_claims:
- gpu: sm100-class (specific SKU not stated in PR)
  dtype: not stated in PR
  shape: batch=512, seqlen_q=1, seqlen_k=16384, nheads=128, topk=2048
  metric: latency_ms
  value: 0.31
  source_id: pr-flash-attention-2441
  source_locator: https://github.com/Dao-AILab/flash-attention/pull/2441 (PR description, "DSA, no bitmask")
blackwell_relevance: PR-grade CuTe DSL SM100 MLA code is directly relevant to DSA sparse attention and top-k KV-gather routing on SM100-class GPUs.
artifact_dir: artifacts/prs/flash-attention/PR-2441
---

## Shape

FlashAttention PR 2441 adds an SM100 CuTe DSL forward path for MLA shapes with
top-k sparsity. It is useful when an attention candidate has to combine page/KV
layout handling, sparse top-k selection, and tiled forward scheduling.

```python
for i in cutlass.range_constexpr(entries_per_thread):
    topk_idx = rTopk[i]
    if const_expr(not self.disable_bitmask):
        row_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        tPrRowValid[i] = row_valid
    if const_expr(not transpose):
        tPrXPtr[i] = utils.elem_pointer(mX, (topk_idx, 0)).toint()
    else:
        tPrXPtr[i] = utils.elem_pointer(mX, (0, topk_idx)).toint()
```

This is a contiguous excerpt from `topk_gather_kv.py` in the retained PR
snapshot. It is implementation context, not a standalone benchmark. The full
snapshot and PR tests are required to exercise the path.

## Transfer Notes

- Treat top-k gather and tiled attention scheduling as separate evidence paths.
- Profile memory traffic separately from tensor-pipe utilization; sparse top-k
  routing can improve arithmetic work while worsening gather locality.
- Keep full-workload validation because the useful path is shape-specific.
