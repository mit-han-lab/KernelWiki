---
id: technique-ping-pong-scheduling
title: Ping-Pong Scheduling
type: technique
architectures: [sm100]
tags: [ping-pong-scheduling, warp-specialization, tmem, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem, technique-warp-specialization]
related: [kernel-flash-attention-4, technique-double-buffering]
sources: [blog-flash-attention-4, doc-flash-attention-4]
---

# Ping-Pong Scheduling

FlashAttention-4 processes two query tiles per CTA and alternates their matrix-multiply and softmax work. While one tile advances through tensor-core work, a dedicated softmax warpgroup can process the other tile's intermediate state. Synchronization and a separate correction stage keep dependencies explicit.

The technique is useful because FA4's forward pass must overlap resources whose throughput scaled differently on Blackwell. It aims to improve overlap; it does not prove that either resource remains busy continuously.

## Requirements

- separate live storage for both tiles, including their TMEM-backed intermediates;
- producer/consumer barriers whose phases cannot be confused across iterations;
- an epilogue/correction schedule that does not overwrite a tile before its consumers finish;
- enough loop work to amortize the extra storage and synchronization.

Use the FA4 implementation for exact code. The earlier local “full implementation” was an unrelated CUTLASS backward-attention file, and the teaching skeleton was not upstream FA4 code; both were removed.

At the implementation commit pinned by `doc-flash-attention-4`, the SM100
forward kernel dispatches two softmax roles to distinct stage arguments:

```python
if warp_idx < self.softmax1_warp_ids[0]:
    softmax_loop(stage=0, tStS=tStS)
if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
    softmax_loop(stage=1, tStS=tStS)
```

This is a contiguous excerpt from `flash_attn/cute/flash_fwd_sm100.py`; the
surrounding pipeline and barriers define when each stage may run.
