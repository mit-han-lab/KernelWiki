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
performance_claims: []
evidence_basis:
- source_id: doc-flash-attention-4
  evidence_type: paper
- source_id: blog-flash-attention-4
  evidence_type: source-reported
blackwell_relevance: The paper's SM100 design overlaps tcgen05 MMA with non-matmul
  work, stores large intermediates in TMEM, partially emulates exp2 on FMA units,
  and uses two-CTA MMA in backward.
artifact_dir: artifacts/kernels/flash-attention-4
---

# FlashAttention-4

## Verified Scope

FlashAttention-4 is an attention algorithm and CuTe DSL implementation designed around Blackwell's asymmetric throughput: B200 tensor-core throughput grows much more than its special-function and shared-memory resources. This page separates two scopes:

- the [FA4 paper v1](https://arxiv.org/abs/2603.05451v1), which studies the SM100/B200 algorithm and reports the authors' measurements; and
- the public implementation at Dao-AILab/flash-attention commit [`a369df707e1980fb328abcc1733e3457ec10155f`](https://github.com/Dao-AILab/flash-attention/tree/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute), which is the source snapshot used for implementation statements below.

The paper implementation is written in CuTe DSL. Its compilation comparison is against corresponding FA3 CUTLASS kernels: forward compiles in 2.5 seconds instead of 55 seconds, and backward in 1.4 seconds instead of 45 seconds. Those are single-kernel source-reported compile measurements, not an end-to-end installation or runtime speedup.

## Ping-Pong Forward Schedule

The forward kernel assigns two output tiles of 128 query rows to one CTA. One MMA warp issues the matrix products, two four-warp softmax groups serve the two output tiles, and a correction warpgroup handles accumulator corrections. The score and output accumulators live in disjoint TMEM regions. Pipeline barriers allow softmax work for one output tile to overlap MMA work that advances the other tile.

This is more than ordinary K/V double buffering: the two alternating objects are output tiles with separate softmax state. The exact synchronization and stage ownership are implementation details; the invented one-loop pseudocode formerly on this page did not preserve those dependencies.

## Partial Software Exponential

FA4 does not replace every hardware exponential. The paper selects only about 10-25% of entries for software evaluation on FMA units and leaves the rest on the hardware MUFU `ex2` path, allowing both resources to contribute.

For a software-selected value, the published range reduction writes `x = n + f` with `n = floor(x)` and `f` in `[0, 1)`, evaluates a degree-3 polynomial for `2**f`, and reconstructs the scale from `n`. The function below is a scalar reference using the rounded coefficients printed in the first-party blog. It illustrates that formula; it is not the CuTe kernel's vector selection, clamping, or scheduling code.

```python
import math

def fa4_blog_exp2_reference(x: float) -> float:
    n = math.floor(x)
    f = x - n
    polynomial = 1.0 + f * (0.6951 + f * (0.2276 + f * 0.0771))
    return math.ldexp(polynomial, n)
```

No standalone four-times software-versus-hardware exponential result is asserted here. The paper evaluates the combined kernel and its ablations rather than establishing that former page claim.

## Conditional Softmax Rescaling

Ordinary online softmax updates the row maximum and rescales accumulated state as each score block arrives. FA4 permits its retained maximum to lag: it resynchronizes only when the new block maximum exceeds the retained maximum by more than a threshold. The paper's typical threshold is `tau = log2(256) = 8.0` in the exponent's base-2 units.

When a rescale is skipped, subsequent probabilities are still evaluated relative to the retained old maximum, and auxiliary statistics track the delayed normalization. The algorithm performs final renormalization at the end. Comparing absolute changes in LSE, changing the maximum anyway, or simply skipping the accumulator multiply is not equivalent.

## Two-CTA Backward

The paper maps five backward GEMMs to two-CTA tcgen05 MMA with `M=256, N=128, K=128`. For those operations, the paired CTAs can share operand B, which the authors describe as roughly halving the shared-memory reads for that operand. This is not a claim that all shared-memory traffic for dQ, dK, and dV is halved.

For dQ, each CTA computes a half of dS and exchanges that half through distributed shared memory so both CTAs can form the required dQ product. The two-CTA organization also doubles the dQ reduction tile along N and thereby halves the number of global atomic reductions described by the paper. It does not assign dK exclusively to CTA 0 and dV exclusively to CTA 1.

## Pinned Implementation Notes

At commit `a369df7`:

- [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_fwd_sm100.py) builds tcgen05 operations through `make_trivial_tiled_mma`, allocates TMEM through `TmemAllocator`, and uses explicit TMEM column offsets for score and output accumulators.
- The forward path constructs TMA atoms for Q, K, and V where its configuration enables them. It also has non-TMA Q and paged-K/V copy paths, so TMA use is not unconditional.
- [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_bwd_sm100.py) constructs the five backward MMA operations and the two-CTA exchange/reduction pipelines rather than splitting dK and dV by cluster rank.
- The package README describes CuTe DSL attention for Hopper and Blackwell, and the tree contains SM90, SM100, and SM120 dispatch modules. The paper's FA4 result remains SM100/B200-specific; package architecture coverage should not be used to broaden that performance result.
- The pinned [`pyproject.toml`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/pyproject.toml) requires `nvidia-cutlass-dsl==4.6.0.dev0`. That is a property of this source snapshot, not a timeless minimum.
- Forward and backward accept FP16/BF16. An FP8 benchmark/bring-up script is present, but at this revision it explicitly expects the FA4 FP8 call to fail until support is implemented.

## Performance Evidence

The first-party sources contain two different source-reported peak values. Paper v1 reports **up to 1613 TFLOPS/s on B200 BF16, or 71% of the peak convention used by the authors**. Tri Dao's blog reports **up to 1605 TFLOPS/s, also labeled 71%**, plus up to 1.3x over cuDNN 9.13 and up to 2.7x over Triton.

The paper's benchmark suite spans sequence lengths from 1K through 32K and multiple query/value head-dimension pairs under a fixed total-token convention. Neither textual source establishes the former single row that attached 1605 TFLOPS, 71%, and both speedup ranges specifically to `seqlen=8192, headdim=128`. The structured performance record is therefore empty, and no complement-of-71% time breakdown is inferred.

## Practical Boundaries

- Choose the package path only for a supported architecture, dtype, head-dimension pair, masking mode, and feature set at the exact revision in use.
- Do not treat sequence length 1024 or head dimension 128 as universal crossover or optimum values. Compare against the relevant cuDNN, framework, or other kernel path on the actual workload.
- Compilation speed and runtime speed are separate measurements. The paper's compile comparison does not prove an equivalent runtime factor.
- Treat source-reported B200 numbers as unreproduced unless the same software, clock/power settings, tensor shapes, timing region, warmup, and FLOP convention are available.

## Pinned Sources

- [FlashAttention-4 paper, arXiv v1](https://arxiv.org/abs/2603.05451v1)
- [Tri Dao's FlashAttention-4 blog](https://tridao.me/blog/2026/flash4/)
- [FA4 CuTe DSL package at commit `a369df7`](https://github.com/Dao-AILab/flash-attention/tree/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute)
- [Pinned SM100 forward source](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_fwd_sm100.py)
- [Pinned SM100 backward source](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_bwd_sm100.py)

## Local Code References

The local [`full/`](../../artifacts/kernels/flash-attention-4/full/) bundle is a byte-verified, verbatim **adjacent NVIDIA CUTLASS SM100 FMHA backward MLA example**, pinned to CUTLASS commit `0e026982`. It is not the Dao-AILab FA4 implementation. The local [`variants/`](../../artifacts/kernels/flash-attention-4/variants/) bundle contains explicitly derived teaching material, including the scalar software-exp formula; it is not upstream code. Exact FA4 implementation evidence is linked to the immutable Dao-AILab commit above.

Query the page and its attached local references with:

```bash
python3 scripts/get_page.py kernel-flash-attention-4 --include-code
```
