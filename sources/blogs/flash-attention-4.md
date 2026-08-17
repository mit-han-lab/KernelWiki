---
id: blog-flash-attention-4
title: FlashAttention-4 Blog
author: Tri Dao
url: https://tridao.me/blog/2026/flash4/
source_category: benchmark-blog
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling, conditional-rescaling, cute-dsl]
retrieved_at: 2026-08-16
---

# FlashAttention-4 blog

Tri Dao's post describes the Blackwell FlashAttention-4 design and links the paper/code. Its source-reported points include the ping-pong schedule, partial software exponential evaluation, conditional rescaling, two-CTA backward design, and a roughly 20–30× compilation comparison.

The more specific 10–25% emulation share and the 2.5/1.4-second versus
55/45-second per-kernel compilation table come from the associated paper, not
from the blog text. They are retained on the paper source record and synthesized
wiki page with that attribution.

The associated paper reports up to 1,613 TFLOP/s on B200 BF16/FP16 (about 71% of its theoretical reference), up to 1.3× over cuDNN 9.13, and up to 2.7× over its Triton baseline. Those maxima occur on particular plotted configurations.

The former local page embedded explanatory pseudo-CUDA and an incomplete `tcgen05.mma` form under extracted-code provenance. Those were not verbatim blog code and are removed. The local CUTLASS PR-2466 bundle is analogous FMHA code, not the FA4 implementation.
