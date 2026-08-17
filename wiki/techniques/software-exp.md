---
id: technique-software-exp
title: "Software-Assisted Exponential"
type: technique
architectures: [sm100]
tags: [software-exp, attention]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-flash-attention-4, technique-warp-specialization]
sources: [blog-flash-attention-4, doc-flash-attention-4]
---

# Software-Assisted Exponential

## What FlashAttention-4 does

FlashAttention-4 addresses exponential throughput in the softmax path by **partially** emulating `exp2` with ordinary floating-point instructions. The paper does not replace every hardware exponential with a four-FMA polynomial, nor claim an architecture-wide 8x speedup. Its decomposition uses an integer part and a fractional part in `[0, 1)`; only roughly 10–25% of the exponential work is shifted to FMA-based emulation, with the rest still using the MUFU path.

That distinction matters: the technique balances two execution resources. The best fraction depends on their instruction throughput, dependencies, the attention configuration, and other overlapped work.

## Structural sketch

```python
import math

def exp2_structure(x):
    """Mathematical structure only; not FA4's production approximation."""
    integer = math.floor(x)
    fraction = x - integer       # 0 <= fraction < 1
    fractional_power = approximate_exp2_fraction(fraction)
    return math.ldexp(fractional_power, integer)
```

The production approximation requires the paper/code's actual coefficients, range handling, rounding behavior, and target-specific instruction sequence. A Taylor series presented without an error bound is not a drop-in substitute.

## Why partial emulation can help

Attention performs many exponentials while tensor-core work, reductions, and scalar arithmetic overlap. If profiling shows the MUFU pipeline limiting progress while FMA capacity is available, moving a measured fraction of work can improve balance. Moving all work can instead create a dependent-FMA bottleneck.

## Numerical requirements

- Use the same log base and scale as the softmax algorithm.
- Define behavior for large negative inputs, underflow, infinities, and NaNs.
- Bound approximation error over the **actual reduced interval**.
- Test probability normalization and final output error, not just scalar `exp2` error.
- Preserve online-softmax rescaling semantics. FA4's conditional rescale is a separate optimization and its threshold is configuration-dependent (the paper says it is typically `log2(256) = 8`).

## Evidence boundaries

The FA4 paper reports up to 1,613 TFLOP/s (71% of B200 BF16/FP16 peak) for its complete attention kernel and comparisons up to 1.3x versus cuDNN 9.13 and 2.7x versus Triton in the tested sweep. Those end-to-end results cannot be assigned to partial exponential emulation alone.

Use this technique only after instruction-level or stall profiling identifies exponential throughput as relevant, and compare hardware-only, partial-emulation, and alternative approximation mixes under identical shapes and accuracy checks.
