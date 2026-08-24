---
id: kernel-gated-delta-net
title: Gated Delta Network kernels
type: kernel
architectures: [sm100, sm90]
tags: [gated-delta-net, linear-attention, triton, kernel-fusion]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-delta-net, linear-attention]
languages: [triton, python]
related: [technique-kernel-fusion, pattern-memory-bound]
sources: [blog-gated-delta-net, contest-flashinfer-track-c, pr-sglang-21019]
performance_claims: []
blackwell_relevance: The FlashInfer contest targeted B200 Gated Delta Net decode and prefill; the retained SGLang projection-fusion example was reported on H200 and is not presented as a Blackwell-specific kernel.
artifact_dir: artifacts/kernels/gated-delta-net
---

# Gated Delta Network kernels

Gated Delta Networks combine a gated recurrent state update with a delta-rule
correction. Implementations commonly separate or fuse projections, local
convolution, recurrence/chunk processing, normalization, and output projection.
The exact recurrence must come from the paper or implementation; a simple gated
outer-product update is not an equivalent substitute.

## Retained implementation excerpt

SGLang PR 21019 fuses the split/reshape/concatenate work around Qwen3.5's GDN
projection. It is not the recurrent update itself. This contiguous excerpt from
the captured Triton file shows its interleaved-input stores:

```python
tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
```

The full upstream file supports both the interleaved Qwen3-Next layout and the
contiguous Qwen3.5 layout. Layout identity is therefore a correctness condition,
not a performance-only choice.

## Evidence and reproduction boundary

- The NVlabs repository provides the research reference and points to FLA for
  a faster variable-length implementation.
- The FlashInfer MLSys 2026 page identifies a B200 Gated Delta Net track and
  winner names, but publishes no latency or throughput table.
- The retained SGLang PR description reports an H200 projection-fusion
  benchmark; it does not validate a `tcgen05.mma` form or a universal GDN
  mainloop.

The former synthetic prefill/decode kernels and invalid tcgen inline PTX were
removed. Reproduction requires an upstream implementation plus its model
layout, recurrence parameters, state initialization, dtype, and tolerance.
