---
id: doc-flash-attention-4
title: "FlashAttention-4: Algorithm and Kernel Co-design for Blackwell GPUs"
url: https://arxiv.org/abs/2603.05451v1
source_category: paper
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling]
retrieved_at: 2026-08-08
---

## Evidence Scope

FlashAttention-4 paper v1 (2026-03-05), an author-primary description of the algorithm, CuTe DSL implementation, and B200 evaluation. This entry is a summary, not executable source or an independently reproduced benchmark.

## Forward Pass

- One CTA pipelines two 128-row output tiles through one MMA warp, two softmax warpgroups, and a correction warpgroup, with score/output accumulators in TMEM.
- Roughly 10-25% of exponential entries are evaluated with a Cody-Waite-style floor reduction and degree-3 FMA polynomial; the remainder use hardware MUFU `ex2`.
- Conditional rescaling permits the retained row maximum to lag, typically by `tau=log2(256)=8.0`, then performs final renormalization.

## Backward Pass

- Five backward GEMMs use two-CTA MMA with `M=256, N=128, K=128` in the described configuration.
- Pairing roughly halves shared-memory reads for operand B of those GEMMs; it does not halve all backward shared-memory traffic.
- dQ uses a distributed-shared-memory exchange of half-dS tiles and a doubled reduction width that halves the described global atomic reductions.

## Implementation and Performance

- The implementation is written in CuTe DSL. The paper reports single-kernel compilation of 2.5 seconds versus 55 seconds for forward and 1.4 seconds versus 45 seconds for backward compared with FA3.
- Paper v1 reports up to 1613 TFLOPS/s on B200 BF16, labeled 71% under the authors' peak convention, and up to 1.3x over cuDNN 9.13 and 2.7x over Triton.
- The benchmark suite spans several sequence lengths and head-dimension pairs. The text does not establish one `seqlen=8192, headdim=128` row containing all those maxima.
