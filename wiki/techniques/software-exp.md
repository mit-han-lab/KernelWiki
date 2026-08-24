---
id: technique-software-exp
title: "Software-Emulated Exponential"
type: technique
architectures: [sm100]
tags: [software-exp, attention]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-flash-attention-4, technique-warp-specialization]
sources: [blog-flash-attention-4, doc-flash-attention-4]
---

# Software-Emulated Exponential

FlashAttention-4 uses a software approximation of `exp2` on FMA units alongside Blackwell's hardware MUFU exponential path. The goal is to raise effective exponential throughput during softmax by using FMA capacity that would otherwise be underused while attention's tensor-core work is in flight.

## Method reported by FlashAttention-4

The implementation described by the authors has three parts:

1. Cody-Waite range reduction decomposes the exponent into integer and fractional components.
2. A Sollya-optimized polynomial approximates the fractional power of two and is evaluated in Horner form on FMA units.
3. The integer component is applied through the floating-point exponent representation.

The FA4 schedule distributes work across both the hardware MUFU path and the software FMA path; it is not simply a wholesale replacement of every hardware exponential.

## Kernel integration

The technique matters in combination with FA4's forward pipeline:

- two query tiles are processed in a ping-pong schedule;
- dedicated softmax warpgroups coordinate so their exponential phases do not collide;
- a correction warpgroup handles conditional online-softmax rescaling outside the critical stage;
- TMEM holds intermediate attention data used across the pipeline.

## Porting guidance

This is a kernel-specific numerical approximation, not a generic instruction substitution. A port must preserve the approximation range, exceptional-value behavior, and softmax error tolerance of the tested implementation. It should also confirm by profiling that the exponential path is the relevant bottleneck and validate outputs against an appropriate reference.

No universal exponential speedup, FMA latency, SFU latency, or accuracy bound is asserted here; the cited FA4 sources do not establish those earlier local figures as portable hardware facts.

The authors' implementation pinned by `doc-flash-attention-4` contains the
following contiguous core of `ex2_emulation` in `flash_attn/cute/utils.py`:

```python
x_clamped = cute.arch.fmax(x, -127.0)
x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
x_rounded_back = x_rounded - fp32_round_int
x_frac = x_clamped - x_rounded_back
x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)
```

It relies on surrounding helpers, coefficient tables, CuTe DSL types, and the
documented input-domain assumptions; this excerpt is not independently usable.

## Sources

- [FlashAttention-4 author blog](../../sources/blogs/flash-attention-4.md)
- [FlashAttention-4 paper](../../sources/docs/flash-attention-4.md)
