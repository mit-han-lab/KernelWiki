---
id: hw-nvfp4
title: NVFP4 and block-scaled narrow precision
type: hardware
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, block-scale, fp8, fp6]
confidence: source-reported
related: [technique-fine-grained-quantization, kernel-nvfp4-gemm, kernel-nvfp4-gemv, hw-tcgen05-mma]
sources: [doc-ptx-isa-sm100, contest-gpumode-p1, contest-gpumode-p2, pr-cutlass-2139]
aliases: [NVFP4, E2M1, FP4 E2M1, nv_float4]
---

# NVFP4 and block-scaled narrow precision

NVFP4 stores signed E2M1 values and supplies finer-grained scale metadata. The
GPU Mode task prose labels each per-16 scale `fp8(e4m3fnuz)`, while the
organizer's executable reference constructs the same scale tensors as
`torch.float8_e4m3fn`. Both contest input tuples expose only those per-16 scale
tensors (plus permuted views for the executable path); they do not expose a
separate tensor-level scale input. The organizer artifacts disagree on the
signed FP8 encoding name. Neither name is a synonym for PTX `ue4m3`, which PTX
9.3 defines as a 7-bit unsigned format, and the contest's per-16 scaling is
also distinct from MXFP4's block-32 UE8M0 scaling.

SM100 PTX exposes separate block-scale MMA kinds rather than one generic FP4
form:

- `kind::mxf8f6f4.block_scale` for MX FP8/FP6/FP4 combinations;
- `kind::mxf4.block_scale` for MXFP4;
- `kind::mxf4nvf4.block_scale` for the supported MXFP4/NVFP4 combinations.

The instruction descriptor, operand descriptors, scale-factor descriptor, and
scale layouts must all agree with the chosen kind. An NVFP4 scale must not be
silently renamed or rounded to another encoding and then described as
equivalent NVFP4 arithmetic.

The former per-opcode “4× versus Hopper” table was removed. Product-level peak
throughput comparisons do not establish the throughput of every instruction
shape or an application speedup.
