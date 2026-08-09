---
id: technique-software-exp
title: "Software-Emulated Exponential"
type: technique
architectures: [sm100]
tags: [software-exp, attention]
confidence: verified
evidence_basis:
  - source_id: doc-flash-attention-4
    evidence_type: official-doc
reproducibility: concept
prerequisites: []
related: [kernel-flash-attention-4, technique-warp-specialization]
sources: [blog-flash-attention-4, doc-flash-attention-4, doc-ptx-isa-sm100]
blackwell_relevance: "FlashAttention-4 uses a configuration-dependent hybrid of hardware MUFU.EX2 and a degree-3 FMA polynomial for selected forward-softmax entries on SM100; it does not replace every exponential."
---

# Software-emulated exponential

## What FlashAttention-4 implements

FlashAttention-4 (FA4) combines two paths for base-2 exponential in its Blackwell forward softmax. Most selected configurations retain hardware `exp2` for some entries and evaluate a tunable fraction with a software polynomial on general-purpose FMA pipelines. The paper describes roughly 10–25% software evaluation, while the exact fraction is a configuration choice rather than an architecture constant.

This hybrid is part of a larger schedule: two 128-thread softmax warpgroups alternate 128-row query tiles, synchronize to limit simultaneous exponential contention, stage probabilities through TMEM, and hand conditional rescaling to a correction warpgroup. A standalone polynomial is not equivalent to that pipeline.

## Scoped bottleneck model

For the authors' `M=N=D=128` B200 feeds-and-speeds model, one SM supplies 8192 BF16 tensor-core operations per cycle, 16 exponential operations per cycle, and 128 shared-memory bytes per cycle. Their forward-tile accounting assigns 1024 cycles to two MMAs, 1024 cycles to 128×128 exponentials, and 768 cycles to shared-memory traffic.

Those are analytical inputs for that tile and schedule. They do not imply that every kernel with one exponential per MMA output is exponential-bound, nor do functional-unit counts give the throughput or latency of a software approximation. Range reduction, polynomial dependencies, reconstruction, instruction issue, and overlap all remain part of the measured implementation.

## Pinned software path

At Dao-AILab/flash-attention revision `a369df707e1980fb328abcc1733e3457ec10155f`, [`flash_attn/cute/utils.py`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/utils.py) defines the default software path and [`flash_attn/cute/softmax.py`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/softmax.py) performs fragment-level hardware/software selection.

For each software-selected pair, the pinned implementation:

1. Assumes each input is no greater than 127 and clamps values below `-127`.
2. Uses a rounding-down addition with the float32 constant `2^23 + 2^22` to recover the integer floor and a fractional value in `[0,1)`.
3. Evaluates a degree-3 polynomial in Horner form with packed float32 FMA operations.
4. Combines the integer contribution with the polynomial's float32 representation by exponent-field integer operations; the software branch does not call `ex2.approx` for reconstruction.

The degree-3 float32 coefficients in that revision are:

| Term | Coefficient |
|---:|---:|
| `p0` | `1.0` |
| `p1` | `0.695146143436431884765625` |
| `p2` | `0.227564394474029541015625` |
| `p3` | `0.077119089663028717041015625` |

The authors state that Sollya selected these coefficients to minimize relative error for `2^f` over `f in [0,1)`. For a natural exponential, the caller uses the identity `e^z = 2^(z * log2(e))`.

## Numerical contract

The polynomial is approximate. On a deterministic host grid of 1,000,002 evenly spaced points over `[0,1]`, evaluating the rounded degree-3 coefficients against `2^x` produced a maximum sampled relative error of approximately `8.763e-5`. This is a regression observation, not a proof of the continuous maximum and not an end-to-end attention tolerance.

BF16 conversion does not by itself prove that approximation errors are harmless. Per-entry error changes both the softmax numerator and row sum, and masking, all-`-inf` rows, conditional rescaling, underflow clamping, and output accumulation introduce separate boundary cases. Validate the scalar approximation and the complete attention output for the exact dtype and features.

## Reproduction and decision procedure

Use the pinned implementation as the starting point; do not reconstruct it from rounded blog coefficients. For each target configuration:

1. Build an all-hardware control and one or more hybrid selections from the same source revision, compiler, target, and launch configuration.
2. Inspect generated PTX/SASS to confirm which entries use hardware `MUFU.EX2`, which use the expected FMA/range-reduction sequence, and whether register use or spills changed.
3. Test ordinary, masked, all-masked, causal, variable-length, large-gap, and underflow-heavy rows against a declared higher-precision reference. Record maximum absolute/relative output error and row-sum behavior, not only BF16 agreement on random inputs.
4. Profile exponential-pipeline pressure, FMA/ALU issue, tensor-core overlap, registers, and spills. A high hardware-exp metric is a lead, not proof that moving more entries to FMA improves the schedule.
5. Benchmark identical shapes, inputs, warmup, synchronization, clock policy, and repeated-trial statistics. Sweep the software fraction because both zero emulation and mixed settings appear in the pinned target/configuration table.

Keep the hybrid only where both the numerical contract and declared end-to-end metric pass. Do not transfer the choice to another architecture or transcendental function without repeating the reduction, approximation, special-value, generated-code, and performance checks.

## Performance provenance

The FA4 blog reports complete forward-pass speedups of 1.1–1.3× over cuDNN 9.13 on its evaluated B200 BF16 configurations. That comparison includes the full FA4 co-design—pipelining, hybrid exponentials, conditional rescaling, TMEM staging, scheduling, and other choices. It is not a software-exponential-only ablation and must not be used as the isolated speedup of this technique.
