---
id: technique-fine-grained-quantization
title: "Fine-Grained FP8/FP4 Quantization"
type: technique
architectures: [sm100, sm90]
tags: [fine-grained-quantization, fp8, fp4, nvfp4, block-scale]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-nvfp4]
related: [hw-nvfp4, kernel-deepgemm]
sources: [blog-deepgemm, doc-ptx-isa-sm100, doc-transformer-engine-2.13-nvfp4]
blackwell_relevance: "Blackwell tcgen05 has native block-scaled MMA combinations using UE8M0 or UE4M3; the legal scale type and vector size depend on the selected MMA kind."
---

## Overview

Fine-grained quantization associates low-precision payloads with scales for
subsets of a tensor. Smaller groups let scale selection respond more locally,
but they do not guarantee a particular error or speed result. Keep four choices
separate when transferring a kernel: payload type, scale type, scale geometry,
and physical scale layout.

Three verified recipes in this wiki illustrate why the full contract matters:

- **DeepSeek FP8 training geometry:** one scale for each token row and 128
  activation channels (`1 x 128`), and one scale for each 128-input-channel by
  128-output-channel weight block (`128 x 128`).
- **NVFP4 1D recipe:** E2M1 payloads, one E4M3 local scale per 16 consecutive
  values, and one FP32 global scale per tensor. Transformer Engine 2.13 also
  defines a weight-oriented 2D mode with one local scale per `16 x 16` block.
- **MXFP4:** E2M1 payloads with one UE8M0 power-of-two scale per 32 values; the
  NVFP4 per-tensor global scale is not part of this microscaling format.

Scale-count and storage overhead are different quantities. A `128 x 128` block
has one factor per 16,384 payloads, a factor-count ratio of about 0.0061%. If
the payload is one byte and the factor is FP32, the byte ratio is instead
`4 / 16384`, about 0.0244%. Include factor width, payload width, padding, and
layout transformations in any storage or bandwidth claim.

## DeepGEMM FP8 implementations

At commit `891d57b4db1071624b5c8fa0d1e51cb317fa709f`, DeepGEMM uses
architecture-specific scale representations and accumulation paths.

### SM90

The pinned SM90 1D1D kernel consumes FP32 A/B factors and fixes
`BLOCK_K == 128`. The DeepSeek-V3 report characterizes the H800 FP8 Tensor Core
path as retaining about 14 bits and describes this interval as four WGMMAs in
its configuration. The exact kernel accumulates one K block in `float accum`
and then applies both factors into a separate `float final_accum`:

```cpp
final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
```

This is a source-specific implementation, not a rule for every Hopper FP8
kernel. The cited sources do not establish the deleted 0.1% error bound or a
universal ranking of `Nc=32`, `64`, `128`, and `256`.

### SM100

The pinned SM100 1D1D template accepts K scale granularities of 32 or 128 for
each operand. Its public interface packs four UE8M0 factors in each 32-bit
container. TMA moves factor blocks to shared memory, UTCCP copies them into
dedicated TMEM SFA/SFB columns, and block-scaled UMMA consumes the TMEM scale
addresses. The SM90 `final_accum` promotion loop is absent.

PTX ISA 9.0 expresses the corresponding FP8 block-scaled operand classes with
this grammar-level form:

```ptx
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
    [d_tmem], a_desc, b_desc, idesc,
    [scale_a_tmem], [scale_b_tmem], enable_input_d;
```

This line is an instruction-shape reference, not a complete kernel: declarations,
legal descriptors, collective participation, scale layouts, ordering, completion,
and an architecture-specific target are still required. UE8M0 uses power-of-two
finite values and reserves encoding `0xff` for NaN.

## Blackwell FP4 block-scale combinations

PTX ISA 9.0 distinguishes the native formats by the complete instruction
combination, not by “FP4” alone:

| Instruction qualifiers | Payload | Local scale | Group |
|---|---|---|---:|
| `.kind::mxf4.block_scale.block32` / `.scale_vec::2X` | E2M1 | UE8M0 | 32 |
| `.kind::mxf4nvf4.block_scale.block16` / `.scale_vec::4X` | E2M1 | UE4M3 | 16 |

The `mxf4nvf4` family also has documented UE8M0 modes, so the kind name alone
does not select the NVFP4 recipe. NVFP4-compatible UE4M3 scaling is native on
the documented Blackwell targets; it is not necessarily a software-decode path.

## Transfer checklist

1. Match the model/checkpoint recipe: payload encoding, local scale type,
   optional global scale, and scale geometry.
2. Match the kernel ABI: logical factor shape is not necessarily its TMA- or
   TMEM-ready physical layout. Account for packing, padding, transposition, and
   swizzling.
3. Match the target instruction. For native Blackwell block scaling, use the
   complete kind, block/scale-vector qualifier, scale type, and target rules.
4. Keep preparation in the timed region unless the producer already emits the
   required layout. Otherwise report preprocessing separately.
5. Validate output accuracy with the exact quantizer, workload, reference,
   tolerance, and accumulation path; then measure end-to-end latency or
   throughput for the actual shapes.

## Evidence boundaries

- More groups require more scale elements, but padding and layout determine the
  actual traffic and storage cost.
- Smaller groups and fractional scales provide more representational freedom;
  they do not guarantee lower error for every tensor or scale-selection method.
- DeepGEMM's Nc=128 path is evidence for that pinned SM90 implementation, not a
  universal optimum for Hopper.
- Native instruction support does not make two recipes ABI-compatible. NVFP4,
  MXFP4, and DeepGEMM FP8 differ in payload, scale type, grouping, and layout.

## Primary references

- [DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2)
- [DeepGEMM at commit `891d57b`](https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f)
- [Transformer Engine 2.13 NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [PTX ISA 9.0 `tcgen05.mma`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma)
