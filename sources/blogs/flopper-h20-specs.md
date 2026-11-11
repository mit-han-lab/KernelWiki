---
id: blog-flopper-h20-specs
title: "Flopper.io H20 96GB specification aggregation (secondary)"
author: Flopper.io
url: https://flopper.io/gpu/nvidia-h20-96gb/spec-sheet
source_category: community-note
architectures: [sm90]
tags: [fp8, gemm]
retrieved_at: 2026-11-11
---

# Flopper.io H20 96GB specification aggregation

This is a **secondary specification table**, not an NVIDIA datasheet or a
kernel benchmark. Its [product overview](https://flopper.io/gpu/nvidia-h20-96gb)
explains that no NVIDIA datasheet is linked and that specifications circulate
through server makers and resellers. The site's own confidence label is not
independent evidence of official verification.

The retrieved table reports Hopper/SXM, 96GB HBM3, 4.0 TB/s memory bandwidth,
148 TFLOPS dense BF16/FP16 Tensor Core throughput, 296 TFLOPS dense FP8 Tensor
Core throughput, and 400W TDP.

Use these as labeled, theoretical secondary values only. The H20 wiki page
independently cites NVIDIA for architecture, compute capability, and listed
96GB capacity, but does not claim NVIDIA verification of the other numbers.
FP32 is omitted because secondary public tables disagree. The site's estimated
maximum power is not a board-power measurement and is not adopted.
