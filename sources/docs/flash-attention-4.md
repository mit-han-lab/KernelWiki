---
id: doc-flash-attention-4
title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"
url: https://arxiv.org/abs/2603.05451
source_category: paper
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling]
retrieved_at: 2026-04-16
implementation_url: https://github.com/Dao-AILab/flash-attention/tree/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute
implementation_commit: 0251105a2fb19d2957484b7f023cd8c115286ced
---

## Summary

FlashAttention-4 paper — algorithm-kernel co-design for Blackwell's asymmetric hardware scaling (tensor core throughput doubles but SFU count unchanged).

## Key Contributions

### Forward Pass
- Ping-pong scheduling with two 128-token query tiles per CTA
- Dedicated softmax warpgroups handle S=QK^T accumulator in TMEM
- Software-emulated exponential via Cody-Waite range reduction + Horner polynomial
- Conditional softmax rescaling (only when max jump is large)

### Backward Pass
- 2-CTA backward spanning paired CTAs in a cluster, sharing TMEM
- Halves shared memory traffic and global atomic reductions for dQ

### Implementation
- Written in CuTe-DSL (Python), 20-30x faster compilation than C++ templates

The implementation was independently rechecked at commit
`0251105a2fb19d2957484b7f023cd8c115286ced` on 2026-08-18. Its public README gives
this minimal invocation:

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

The implementation's `utils.py` contains the software exponential helper. The
following is a contiguous excerpt; it is not a standalone kernel:

```python
x_clamped = cute.arch.fmax(x, -127.0)
x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
x_rounded_back = x_rounded - fp32_round_int
x_frac = x_clamped - x_rounded_back
x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)
```

The SM100 forward kernel uses two softmax roles when `q_stage == 2`. This
contiguous scheduler excerpt shows how the implementation gives the two roles
different stage arguments:

```python
if warp_idx < self.softmax1_warp_ids[0]:
    softmax_loop(stage=0, tStS=tStS)
if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
    softmax_loop(stage=1, tStS=tStS)
```

## Performance
- Up to 1613 TFLOPS on B200 BF16 (71% utilization)
- Up to 1.3x over cuDNN 9.13
- Up to 2.7x over Triton
