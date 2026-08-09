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
performance_claims: []
evidence_basis:
- source_id: pr-flash-attention-2441
  evidence_type: upstream-code
blackwell_relevance: PR 2441 adds a pinned SM100 CuTe DSL MLA forward path for
  64/512 query/value dimensions, MQA with 128 query heads per KV head, and
  caller-supplied top-k KV indices.
---

# FlashAttention SM100 MLA Top-K Sparse Forward

## Verified Scope

[Dao-AILab/flash-attention PR 2441](https://github.com/Dao-AILab/flash-attention/pull/2441), merged as [`f219c89c886c6ccbf9d3dbd9fe41b11ac64e9df8`](https://github.com/Dao-AILab/flash-attention/commit/f219c89c886c6ccbf9d3dbd9fe41b11ac64e9df8), adds an SM100 CuTe DSL forward path for the DeepSeek-style MLA dimensions `head_dim=64` and `head_dim_v=512`, with MQA packing of 128 query heads per KV head and caller-supplied top-k indices.

The top-k path is deliberately narrower than a generic paged sparse-attention API. It requires packed GQA/MQA, `qhead_per_kvhead == 128`, and the cp.async K/V gather path. Its constructor defaults to top-k length 2048 and requires the configured length to be divisible by 256. The merged kernel rejects a non-null page table with `page table tbd for MLA`; page-table support was explicitly outside this PR.

## Gather and Scheduling Paths

The pinned merge keeps three responsibilities distinguishable:

- [`topk_gather_kv.py`](../../artifacts/prs/flash-attention/PR-2441/key-files/flash_attn/cute/topk_gather_kv.py) loads caller-provided top-k indices, forms indexed K/V addresses, issues cp.async copies, and optionally constructs validity bitmasks for out-of-range indices.
- [`tile_scheduler.py`](../../artifacts/prs/flash-attention/PR-2441/key-files/flash_attn/cute/tile_scheduler.py) supplies the tile-scheduler implementations selected by the MLA kernel.
- [`flash_fwd_mla_sm100.py`](../../artifacts/prs/flash-attention/PR-2441/key-files/flash_attn/cute/flash_fwd_mla_sm100.py) integrates gather, scheduling, two-CTA tcgen05 MMA, TMEM accumulators, softmax, and the output path.

This separation matters for evaluation. Arithmetic work follows the effective top-k length, whereas memory addresses follow the caller's index set and use an indexed gather/optional-bitmask path. Reduced attention FLOPs therefore do not prove a proportional latency or locality improvement.

## Source-Reported Performance

The PR author reports an initial saturating-decode comparison for batch 512, `seqlen_q=1`, `seqlen_k=16384`, 128 query heads, the 64/512 MLA shape, and top-k length 2048:

| Variant | Author-reported latency | Author-reported throughput |
|---|---:|---:|
| DSA, no bitmask; indices assumed in bounds | 0.31 ms | 955.47 TFLOPS |
| DSA, validity bitmask | 0.33 ms | 898.08 TFLOPS |
| Vanilla MLA baseline | 1.98 ms | 1180.70 TFLOPS |

These PR-description observations are not reproduced here. The PR body does not name the exact GPU model, dtype, clocks/power state, software environment, timing protocol, or run-to-run variation for the rows. They therefore remain prose with exact source qualifications rather than structured `performance_claims`.

## Transfer Checklist

- Confirm the exact MQA packing, 64/512 dimensions, top-k divisibility, index layout, and bounds contract used by the target revision.
- Treat no-bitmask results as valid only when the caller guarantees every index is in bounds; otherwise preserve the validity path.
- Do not infer paged-KV compatibility from the exposed parameter name: merge `f219c89c` rejects page tables in this MLA path.
- Profile indexed-gather traffic and tensor arithmetic separately, then validate end-to-end accuracy and latency on the full workload.

## Reproducible Source Lookup

The local PR record and byte-verified bundle identify the integration and its dedicated gather file:

```python
from pathlib import Path

pr_page = Path("sources/prs/flash-attention/PR-2441.md").read_text()
provenance = Path("artifacts/prs/flash-attention/PR-2441/PROVENANCE.yaml").read_text()
assert "flash_fwd_mla_sm100.py" in pr_page
assert "topk_gather_kv.py" in provenance
```
