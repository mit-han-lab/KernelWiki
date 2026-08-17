---
id: doc-flash-attention-4
title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"
url: https://arxiv.org/abs/2603.05451
source_category: paper
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling]
retrieved_at: 2026-08-16
---

## Summary

FlashAttention-4 paper — algorithm-kernel co-design for Blackwell's asymmetric hardware scaling (tensor core throughput doubles but SFU count unchanged).

## Key Contributions

### Forward Pass
- Two 128-thread softmax warpgroups, one correction warpgroup, and one warpgroup driving Tensor Cores/TMA; two score tiles are pipelined through TMEM
- Partial software exponential: Cody-Waite-style `floor(x)` range reduction to `[0,1)` plus a Horner polynomial on FMA units for 10–25% of entries; remaining entries use `MUFU.EX2`
- Conditional rescaling at a typical base-2 threshold of `log2(256)=8`, while retaining a consistent active scale and applying the true final normalization

### Backward Pass
- 2-CTA MMA with `M=256, N=K=128` for most backward MMAs; each CTA stages half of operand B
- dQ exchanges half of dS through distributed shared memory to give each CTA a 128-row output with a 256-wide reduction
- Halves dQ global atomic reductions; the paper's analyzed total shared-memory cycles fall from 3328 to 2688 rather than being halved overall

### Implementation
- Written in CuTe DSL embedded in Python
- Per-kernel compile-time table: FA4 forward/backward 2.5 s/1.4 s versus FA3 C++ 55 s/45 s (22x/32x; summarized as 20–30x)

## Performance
- Up to 1613 TFLOP/s on B200 BF16/FP16 (approximately 71% of theoretical throughput)
- Up to 1.3x over cuDNN 9.13 and up to 2.7x over the evaluated Triton baseline
- For the plotted forward head-dimension-128 sweep: 1.1–1.3x over cuDNN 9.13.0 and 2.1–2.7x over Triton
- Benchmark sweep: sequence lengths 1k–32k, head dimensions 64/128/(192,128), causal and non-causal, total tokens fixed at 32k; the prose does not identify the peak as sequence length 8192
