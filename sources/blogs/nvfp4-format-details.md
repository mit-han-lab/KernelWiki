---
id: blog-nvfp4-format-details
title: NVFP4 format details
author: Harold Benoit
url: https://haroldbenoit.com/notes/ml/engineering/precision/nvfp4-format
source_category: community-note
architectures: []
tags: [nvfp4, fp4, block-scale, fine-grained-quantization, quantization]
retrieved_at: 2026-08-16
---

# NVFP4 format details

This community note explains the E2M1 element format and a two-level NVFP4 quantization recipe. E2M1 has 16 bit patterns: positive and negative zero plus signed magnitudes 0.5, 1, 1.5, 2, 3, 4, and 6. Because the two zeros compare numerically equal, that is 15 distinct numerical values, not 16.

The SM100 instruction-level NVFP4 format uses one **unsigned UE4M3** block scale per 16 dense K elements (32 in the documented sparse case). The former local summary called this a signed E4M3 value with range `[-448,448]`; that is not the PTX scale type and is corrected here. Many higher-level recipes additionally use an FP32 tensor/global scale, but that extra scale is outside the `tcgen05` operand format.

Packed E2M1 data costs one byte per two values. A byte-sized block scale per 16 values adds 1/16 byte per element, or 12.5% relative to the packed FP4 payload (6.25% of one byte per logical element). These denominators should not be conflated.

Use `doc-ptx-isa-sm100` and `doc-cutlass-blackwell` for hardware/layout truth; use the community note for its quantization recipe and intuition.
