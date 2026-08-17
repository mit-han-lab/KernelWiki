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
performance_claims:
- gpu: "SM100 target; exact GPU SKU not stated"
  software: "FlashAttention PR 2441 at merge SHA f219c89c; CUDA/CuTe versions not stated"
  dtype: bf16
  shape: batch=512, seqlen_q=1, seqlen_k=16384, nheads=128, topk=2048
  workload: "initial saturating SM100 MLA sparse decode timing, DSA without bitmask"
  metric: latency_ms
  value: 0.31
  measurement_method: "initial author-reported PR-body timing; method not stated"
  baseline: "same-shape vanilla MLA: 1.98 ms and 1180.70 TFLOP/s; DSA no-bitmask reports 955.47 TFLOP/s"
  limitations: "exact GPU SKU, software environment, warmup, and timing method are not stated"
  source_id: pr-flash-attention-2441
blackwell_relevance: PR-grade CuTe DSL SM100 MLA code is directly relevant to DSA sparse attention and top-k KV-gather routing; the PR does not identify the exact GPU SKU used for its measurements.
---

## Shape

FlashAttention PR 2441 adds an SM100 CuTe DSL forward path for MLA shapes with
top-k sparsity. It is useful when an attention candidate has to combine page/KV
layout handling, sparse top-k selection, and tiled forward scheduling.

For batch 512, query length 1, KV length 16,384, 128 heads, and top-k 2,048, the PR calls these “initial saturating decode” measurements:

| PR-body variant | Latency | Reported TFLOP/s |
|---|---:|---:|
| Vanilla MLA | 1.98 ms | 1,180.70 |
| DSA without bitmask; selected indices assumed in bounds | 0.31 ms | 955.47 |
| DSA with bitmask | 0.33 ms | 898.08 |

The sparse variants reduce executed work and latency at this decode-shaped `seqlen_q=1` point while reporting lower TFLOP/s; the table should not be read as a tensor-core-throughput increase. These are author-reported SM100-targeted numbers without an exact GPU SKU or a complete measurement-method/software-environment record, so they should not be generalized beyond that PR context.

```python
# Query pattern before borrowing implementation details:
# open PR page, then inspect the source snapshot or upstream files listed there.
from pathlib import Path

provenance = Path("artifacts/prs/flash-attention/PR-2441/PROVENANCE.yaml")
text = provenance.read_text()
assert "flash_fwd_mla_sm100.py" in text
assert "topk_gather_kv.py" in text
```

## Transfer Notes

- Treat top-k gather and tiled attention scheduling as separate evidence paths.
- Profile memory traffic separately from tensor-pipe utilization; sparse top-k
  routing can improve arithmetic work while worsening gather locality.
- Keep full-workload validation because the useful path is shape-specific.
