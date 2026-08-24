---
id: kernel-flash-attention-4
title: FlashAttention-4
type: kernel
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp]
confidence: source-reported
reproducibility: snippet
kernel_types: [attention, flash-attention]
languages: [cute-dsl]
related: [technique-warp-specialization, technique-software-exp, hw-tcgen05-mma, hw-tmem]
sources: [doc-flash-attention-4, blog-flash-attention-4]
performance_claims:
  - gpu: B200
    dtype: bf16
    shape: up to, across paper benchmark settings
    metric: TFLOPS
    value: 1613
    utilization: 71%
    source_id: doc-flash-attention-4
    source_locator: https://arxiv.org/html/2603.05451v1#S5 (§5 Empirical Evaluation)
---

# FlashAttention-4

FlashAttention-4 is a Blackwell attention implementation written in CuTe DSL. Its design responds to asymmetric scaling from Hopper to Blackwell: tensor-core throughput increased more quickly than shared-memory bandwidth and the special-function resources used by softmax.

## Forward pipeline

- A CTA processes two query tiles in a ping-pong schedule to overlap matrix multiplication, softmax, and memory operations.
- Dedicated softmax warpgroups operate on attention intermediates held in TMEM.
- Exponential work is distributed across the hardware MUFU path and a Cody-Waite/Horner software approximation on FMA units.
- A correction warpgroup performs conditional online-softmax rescaling outside the main critical stage.

## Backward pipeline

- Intermediate results are retained in TMEM to reduce shared-memory traffic.
- The two-CTA MMA mode lets paired CTAs cooperate and reduces the shared-memory traffic and global atomic reduction described by the authors.
- The implementation also supports a deterministic scheduling mode for reproducible training.

## Reported performance

The paper reports up to **1613 TFLOPS** in BF16 on B200 across its benchmark sweep, corresponding to **71%** utilization in that reported setting. The author blog separately reports up to 1605 TFLOPS and comparisons of up to 1.3× over cuDNN 9.13 and up to 2.7× over the tested Triton implementation. These are source-reported benchmark maxima, not guarantees for an arbitrary attention shape.

## Reproduction boundary

Use the authors' linked implementation and benchmark configuration for reproduction. Earlier local pseudo-code and the local “full implementation” bundle were removed because they were not a verbatim FA4 implementation: the purported full file came from an unrelated CUTLASS MLA-backward PR.

At the implementation commit pinned by `doc-flash-attention-4`, the public API
README gives this minimal invocation:

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

This is a verbatim usage excerpt, not a self-contained benchmark. Shapes,
dtypes, installation, and correctness checks still have to follow the pinned
upstream implementation.

## Sources

- [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451)
- [FlashAttention-4 author blog](https://tridao.me/blog/2026/flash4/)
