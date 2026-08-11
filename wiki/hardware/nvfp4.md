---
id: hw-nvfp4
title: "NVFP4 and Block-Scaled Narrow Precision"
type: hardware
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, block-scale]
confidence: verified
evidence_basis:
  - source_id: doc-transformer-engine-2.13-nvfp4
    evidence_type: official-doc
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
related: [technique-fine-grained-quantization, kernel-nvfp4-gemm, kernel-nvfp4-gemv, hw-tcgen05-mma]
sources: [doc-transformer-engine-2.13-nvfp4, doc-ptx-isa-sm100]
aliases: [NVFP4, E2M1, "FP4 E2M1", "nv_float4"]
blackwell_relevance: "Blackwell tensor cores support block-scaled E2M1 MMA; NVFP4 combines block-of-16 fractional local scales with a per-tensor FP32 scale."
---

# NVFP4

## Format and recipe

NVFP4 is a quantization recipe, not just another name for its 4-bit payload. It reconstructs a value as:

`x_hat_i = e2m1_i * s_block * s_global`

The three components are:

- **E2M1 payload:** one sign bit, two exponent bits, and one mantissa bit. Its numeric values are 0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, and +/-6. Two E2M1 encodings fit in one byte.
- **Local scale:** one E4M3 scale shared by 16 consecutive payload values. PTX names the corresponding unsigned scale element type `.ue4m3`; its most-significant storage bit is zero.
- **Global scale:** one FP32 scale per tensor. This second level lets the block scales use their range effectively.

Transformer Engine 2.13 also defines a 2D scaling mode for weights in which a scale covers a 16-by-16 block. The 1D mode groups 16 consecutive values.

## NVFP4 and MXFP4

The recipes make different scale tradeoffs:

| Property | NVFP4 | MXFP4 |
|---|---|---|
| Payload | E2M1 | E2M1 |
| Local group in the standard 1D recipe | 16 values | 32 values |
| Local scale | E4M3 (`.ue4m3` in PTX) | UE8M0 |
| Scale values | Fractional values are available | Powers of two |
| Additional recipe scale | Per-tensor FP32 | Not part of the MXFP4 microscaling format |

Finer groups and fractional local scales give NVFP4 more scaling freedom, at the cost of more scale metadata. They do not guarantee strictly lower error for every tensor, scale-selection algorithm, or error metric. Measure accuracy with the exact quantizer and workload.

## PTX block-scaled MMA

In PTX ISA 9.0, the relevant `tcgen05.mma` combinations for E2M1 inputs include:

| Instruction qualifiers | Block | Local scale | Scale-vector qualifier |
|---|---:|---|---|
| `.kind::mxf4.block_scale.block32` | 32 | `.ue8m0` | `.scale_vec::2X` |
| `.kind::mxf4nvf4.block_scale.block16` | 16 | `.ue4m3` | `.scale_vec::4X` |

The `.kind::mxf4nvf4` family also admits documented UE8M0 modes, so the kind name by itself does not select the NVFP4 recipe. Use the complete block and scale-vector qualifiers and follow the type-combination tables. These forms target architecture-specific Blackwell targets such as `sm_100a`; a kernel using them must use the matching compile and runtime target.

PTX ISA 9.0 provides the packed conversion form:

```ptx
cvt.rn.f16x2.e2m1x2 d, a;
```

Here `a` is a byte-sized packed pair and `d` is a 32-bit `f16x2` result. PTX also permits `mov.b32` to unpack a 32-bit scalar into four byte-sized vector destinations when the declarations satisfy its type rules. Neither syntax establishes that one packing strategy is faster than masks and shifts; inspect generated machine code and benchmark the target GPU.

## Performance claims

There is no architecture-independent "4x versus Hopper" result for these instructions. Hopper has no native FP4 tensor-core path, so any comparison depends on the emulation or higher-precision baseline as well as GPU SKU, clocks, matrix shape, layouts, scale preparation, epilogue, and achieved occupancy. Record those conditions with any throughput number.

## References

- [Transformer Engine 2.13: NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [PTX ISA 9.0: `tcgen05.mma`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma)
- [PTX ISA 9.0: `cvt`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- [Fine-grained quantization](../techniques/fine-grained-quantization.md)
- [NVFP4 GEMM](../kernels/nvfp4-gemm.md)
- [NVFP4 GEMV](../kernels/nvfp4-gemv.md)
