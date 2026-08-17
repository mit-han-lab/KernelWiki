---
id: kernel-flash-attention-4
title: FlashAttention-4
type: kernel
architectures:
- sm100
tags:
- attention
- flash-attention
- tcgen05
- tmem
- 2sm-cooperative
- software-exp
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- flash-attention
languages:
- cute-dsl
related:
- technique-warp-specialization
- technique-software-exp
- hw-tcgen05-mma
- hw-tmem
sources:
- doc-flash-attention-4
- blog-flash-attention-4
- pr-flashinfer-1850
performance_claims:
- gpu: B200
  software: "FlashAttention-4 paper v1 CuTe DSL implementation; exact dependency versions not stated in the retained result"
  dtype: bf16
  shape: paper benchmark sweep; peak configuration not identified in prose
  workload: "attention benchmark sweep; exact peak operation/configuration not identified in the retained prose"
  metric: TFLOPS
  value: 1613
  measurement_method: "paper-reported benchmark"
  baseline: "none for the absolute peak; paper comparisons are documented separately"
  limitations: "source-reported ceiling; exact peak shape and full software environment are not reconstructed"
  utilization: 71%
  source_id: doc-flash-attention-4
---

# FlashAttention-4

## Overview

FlashAttention-4 (FA4) is a Blackwell-focused attention implementation that co-designs the attention algorithm and software pipeline for asymmetric scaling: B200 Tensor Core throughput increased much more than shared-memory bandwidth and exponential-unit throughput. The paper addresses those bottlenecks with asynchronous MMA pipelines, partial software exponential evaluation, conditional rescaling, TMEM reuse, and a 2-CTA backward mode.

The implementation is entirely in CuTe DSL embedded in Python. In the paper's per-kernel compilation comparison, FA4 takes 2.5 s forward / 1.4 s backward versus FlashAttention-3's 55 s / 45 s, reported as 22× and 32×. “20–30× faster compilation” refers to this comparison with FA3 C++ templates, not to every CUTLASS program.

## Forward pipeline

Blackwell holds MMA accumulators in TMEM and uses 128×128 accumulator tiles for this workload. FA4 assigns two 128-thread softmax warpgroups, a correction warpgroup, and a warpgroup that drives Tensor Cores and TMA. It retains two output tiles and pipelines two score tiles so MMA, softmax, correction, and data movement can overlap.

The exact pipeline is more involved than alternating two independent Python `with warpgroup(...)` blocks. The authoritative implementation is linked from the paper and project; the former teaching variant was removed because it misstated the software-exponential mechanism.

## Partial software `exp2`

FA4 supplements the hardware MUFU exponential path with a Cody–Waite-style polynomial path on FMA units. Its reduction is:

```python
# Algorithm structure only; coefficients and bit reconstruction are omitted.
def exp2_structure(x, horner_polynomial):
    x = max(x, -127)
    n = math.floor(x)
    f = x - n  # f is in [0, 1)
    return math.ldexp(horner_polynomial(f), n)
```

This corrects two common misstatements:

- The paper uses `floor(x)` with the fractional interval `[0,1)`, not `round(x)` with `[-0.5,0.5]`.
- It does not replace every exponential or claim a universal 4× speedup. Only 10–25% of entries in a softmax row use polynomial emulation; the rest use hardware `MUFU.EX2`. The fraction is tuned to the tile's MMA/exponential throughput balance to avoid register pressure and spills.

For degree 3, the paper reports one-BF16-ULP agreement with hardware for 99% of tested inputs, while higher degrees reduce raw FP32 error at the cost of more FMAs.

## Conditional online-softmax rescaling

Let `m_prev` be the retained running maximum and `m_new` the maximum including the current block. FA4 rescales only when `m_new - m_prev > τ`, typically `τ = log2(256) = 8` in the paper's base-2 formulation. Otherwise it keeps `m_prev` as the active scale and accumulates the new block using that scale. The final normalization uses the true final maximum and normalizer.

This is not equivalent to merely skipping `O *= scale` while always updating the running maximum; that would mix values expressed at different scales.

## 2-CTA backward

For most backward MMAs, the paper uses a 2-CTA tile with `M=256` and `N=K=128`. Each CTA stages half of operand B and owns its accumulator slice, reducing operand-B shared-memory traffic. The dQ step is special: the pair exchanges half of dS through distributed shared memory so each CTA forms a 128-row operand with a doubled 256-wide reduction. This also halves the number of dQ global atomic reductions relative to the 1-CTA counterpart.

The benefit is therefore more specific than “the two CTAs share one accumulator and halve all dQ/dK/dV traffic.” The paper's roofline table reports total backward shared-memory cycles falling from 3328 (1 CTA) to 2688 (2 CTA) for the analyzed tile.

## Paper-reported performance

The v1 paper evaluates BF16/FP16 attention on B200 over sequence lengths 1k–32k, head dimensions 64, 128, and (192,128), causal/non-causal modes, with total tokens fixed at 32k. It reports:

- up to **1613 TFLOP/s**, approximately **71%** of theoretical B200 BF16/FP16 throughput;
- up to **1.3×** over cuDNN 9.13; and
- up to **2.7×** over the evaluated Triton baseline.

For forward head dimension 128 across the plotted sequence lengths, the paper narrows the ranges to 1.1–1.3× over cuDNN 9.13.0 and 2.1–2.7× over Triton. The prose does not identify 1613 TFLOP/s as specifically `seqlen=8192`, so this page does not attach that unsupported shape to the peak.

The 71% is an achieved-to-theoretical ratio. The paper does not partition the remaining 29% into an exact softmax/rescale/memory accounting.

## Applicability and caveats

- The reported implementation and results target Blackwell B200/GB200 and FP16/BF16 attention; do not infer an SM90 fallback or FP8 result from this paper.
- Medium and long sequences (4k and above) are where the paper reports consistent gains over its baselines. It does not establish a universal `seqlen >= 1024` rule.
- CuTe DSL and the open-source FA4 code are the reproducibility path; the snippets on this page are explanatory, not API-compatible kernel code.

## Sources

- [FlashAttention-4 paper v1](https://arxiv.org/html/2603.05451v1)
- [Tri Dao's FA4 blog](https://tridao.me/blog/2026/flash4/)

## Implementation evidence boundary

The local bundle under `artifacts/kernels/flash-attention-4/full/` is a
byte-pinned CUTLASS example-77 SM100 FMHA-backward/MLA kernel from PR 2466. It
is useful as an analogous Blackwell FMHA implementation, but it is CUDA C++ and
is **not** the CuTe DSL Python FlashAttention-4 implementation. The authoritative
FA4 code is linked by the [paper/project](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
