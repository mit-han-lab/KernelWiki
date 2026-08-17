---
id: technique-swizzling
title: "Shared Memory Swizzling"
type: technique
architectures: [sm100, sm90]
tags: [swizzling, shared-memory-optimization, tma]
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
reproducibility: snippet
prerequisites: [hw-tma]
related: [hw-tma, technique-pipeline-stages, pattern-memory-bound]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, blog-modular-blackwell, pr-cutlass-2139]
blackwell_relevance: "SM100 TMA and matrix descriptors support multiple swizzled and unswizzled layouts; the correct mode is selected for the operand layout and access pattern."
---

# Shared Memory Swizzling

## Overview

Shared-memory swizzling permutes address bits so that a collective access is less likely to concentrate requests on the same memory bank. It is a layout transformation: the producer and every consumer must use the same mapping.

TMA tensor maps expose no swizzle, 32-byte, 64-byte, and multiple 128-byte modes. The current CUDA Driver API defines the exact chunk size, span, alignment, and bounding-box restrictions for each mode. The PTX matrix-descriptor tables separately specify which modes are legal for each `tcgen05` operand layout.

## No universal 128-byte rule

128-byte swizzling is common for wide BF16/FP16 tiles, but it is neither mandatory for every Blackwell tensor-core operand nor always optimal. The PTX ISA includes legal no-swizzle, 32-byte, 64-byte, and 128-byte cases. Narrow tiles, scale-factor layouts, and different major orders can require or benefit from different encodings.

An incorrectly encoded descriptor can address the wrong elements. That is different from saying that all unswizzled data is incorrect: no-swizzle is a documented mode when the selected matrix layout permits it.

## Descriptor-first workflow

```python
def choose_smem_layout(consumer_layout, dtype, inner_bytes):
    """Schematic policy; query the PTX/Driver-API tables for legal modes."""
    legal_modes = legal_swizzles(consumer_layout, dtype, inner_bytes)
    candidates = [mode for mode in legal_modes if alignment_is_satisfied(mode)]
    return benchmark_bank_conflicts_and_runtime(candidates)
```

For TMA, encode the chosen mode in the `CUtensorMap`. For a `tcgen05` consumer, construct the matrix descriptor using the corresponding swizzle mode, leading dimension, stride, base offset, and major order. Do not replace this process with a hand-written XOR rule: the official layouts include mode-specific atomicity and base-offset behavior that a single formula does not capture.

## Verification

Check both correctness and performance:

- Compare boundary and odd-size outputs with a trusted reference.
- Confirm descriptor alignment and bounding-box rules at map-creation time.
- Profile shared-memory bank-conflict metrics for the actual access path.
- Benchmark all legal candidate layouts; a zero-conflict synthetic access does not prove the end-to-end kernel is faster.

Any throughput progression in a tutorial is evidence for that tutorial's kernel, shapes, software revision, and GPU. It must not be attributed to swizzling alone unless the experiment controls every other change.
