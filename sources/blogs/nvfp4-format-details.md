---
id: blog-nvfp4-format-details
title: "NVFP4 Format Details"
author: Harold Benoit
url: https://haroldbenoit.com/notes/ml/engineering/precision/nvfp4-format
source_category: community-note
architectures: []
tags: [nvfp4, fp4, block-scale, fine-grained-quantization, quantization]
retrieved_at: 2026-08-18
---

# NVFP4 format details

Harold Benoit's note explains E2M1 values and contrasts MXFP4 with NVIDIA's
hierarchically scaled FP4 representation.

The note states that E2M1 uses one sign bit, two exponent bits, and one mantissa
bit, representing signed zero and magnitudes 0.5, 1, 1.5, 2, 3, 4, and 6.
MXFP4 groups 32 values under a UE8M0 scale. NVFP4 instead groups 16 values under
an E4M3 scale and applies another FP32 scale at tensor level.

For each block, the note chooses the E4M3 block scale from its maximum magnitude
after applying the tensor scale, then quantizes each element by the product of
the tensor and block scales. Dequantization multiplies the decoded E2M1 value by
both scales. Exact rounding, saturation, and storage layout must still follow
the library or hardware interface being used.

Packed FP4 consumes half a byte per element. From the note's one-byte scale and
block sizes, the scale is a 12.5% byte overhead for NVFP4 (one byte per eight
bytes of packed data) and 6.25% for MXFP4 (one byte per sixteen bytes of packed
data). These percentages are derived here rather than stated by the note and
exclude tensor metadata, padding, and layout-specific storage.
