---
id: doc-transformer-engine-2.13-nvfp4
title: "Transformer Engine 2.13: NVFP4"
url: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html
source_category: official-doc
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, block-scale]
retrieved_at: 2026-08-08
version: "2.13"
---

# Transformer Engine 2.13 NVFP4

## Evidence-scoped summary

- NVFP4 uses E2M1 payloads whose largest finite magnitude is 6.
- Its 1D recipe combines one E4M3 scale per 16 consecutive values with a per-tensor FP32 scale.
- Its weight-oriented 2D mode assigns a scale to each 16-by-16 block.
- Compared with MXFP4, the documented recipe uses a smaller group and a fractional rather than power-of-two local scale.
- The supported-hardware table associates training with compute capabilities 10.0 and 10.3 and inference with compute capability 10.0 and later.

These format facts do not, by themselves, establish a universal accuracy or throughput advantage for a particular kernel or workload.

## Primary reference

- [Version-pinned Transformer Engine 2.13 NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
