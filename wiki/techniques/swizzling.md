---
id: technique-swizzling
title: "Shared Memory Swizzling"
type: technique
architectures: [sm100, sm90]
tags: [swizzling, shared-memory-optimization, tma]
confidence: verified
evidence_basis:
  - source_id: doc-cuda-13-0-2-tma
    evidence_type: official-doc
reproducibility: concept
prerequisites: [hw-tma]
related: [hw-tma, technique-pipeline-stages, pattern-memory-bound]
sources: [doc-cuda-13-0-2-tma, doc-ptx-isa-sm100, blog-tcgen05-tutorial]
blackwell_relevance: "SM100 TMA and tcgen05 support multiple shared-memory mappings, including no swizzle and 32B, 64B, and 128B swizzles; correctness requires the producer and consumer to describe the same legal layout, not universal 128B use."
---

# Shared-memory swizzling

## Bank-conflict model

Shared memory has 32 banks, and successive 32-bit words map to successive banks. For one warp memory request, accesses to different words in the same bank require serialized wavefronts; same-word reads can broadcast and do not create that conflict. The result therefore depends on the exact byte address requested by every participating lane.

A swizzle permutes address chunks so a target access pattern may distribute its requests over banks more evenly. It can reduce conflicts for that pattern, but it does not make every access to the tile conflict-free. A layout favorable to an MMA operand can still be unfavorable to a CUDA-core row, column, transpose, or epilogue access.

## Three mappings must agree

For a TMA-to-tcgen05 path, keep these layers consistent:

1. The tensor map describes how TMA places the global-memory box into shared memory, including interleave and swizzle.
2. The shared-memory allocation and base address satisfy the alignment and span constraints of that mapping.
3. The tcgen05 shared-memory descriptor describes the same physical-to-logical layout, leading/stride dimensions, base offset, and swizzle mode expected by the MMA operand.

PTX ISA 9.0 permits ordinary tcgen05 shared-memory descriptor modes with no swizzle and 32B, 64B, or 128B spans, plus a 128B/base-32B mode. In the descriptor's three-bit layout field, ordinary no-swizzle is `0`, 128B/base-32B is `1`, ordinary 128B is `2`, 64B is `4`, and 32B is `6`; values `3`, `5`, and `7` are invalid. Legality also depends on MMA kind, element type, major mode, shape, CTA group, and target.

The CUDA 13.0.97 Driver API exposes TMA modes including:

| Tensor-map mode | Documented chunk mapping |
|---|---|
| `CU_TENSOR_MAP_SWIZZLE_NONE` | No bank swizzle |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 16B chunks within a 32B span |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 16B chunks within a 64B span |
| `CU_TENSOR_MAP_SWIZZLE_128B` | 16B chunks within a 128B span |
| `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B` | 32B chunks within a 128B span |
| `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B` | 64B chunks within a 128B span; support is operation/type-specific |

These names define mappings, not universal datatype or tile-size recommendations. The tensor-map encoder imposes mode-, interleave-, datatype-, alignment-, and box-size constraints; for example, ordinary 128B modes require the inner bounding-box byte span not to exceed 128 bytes. Check the exact archived API for the selected type and operation.

## CuTe address-bit form

CUTLASS 4.5.0 defines [`Swizzle<BBits, MBase, SShift>`](https://github.com/NVIDIA/cutlass/blob/e406c186d2cae5782a846f7280af282ca4fecec2/include/cute/swizzle.hpp):

- `BBits` is the number of mask bits;
- `MBase` is the number of least-significant address bits kept invariant; and
- `SShift` is the distance between the two bit fields.

For `Swizzle<3,4,3>`, three address bits beginning above the four invariant low bits are XORed with the three-bit field shifted by three positions. Equivalently, address bits 7:9 affect bits 4:6. This is an address transformation, not a general `row & 7` rule: equivalence to row-based indexing depends on stride, byte units, base alignment, and layout composition.

CUTLASS also distinguishes a position-independent composed swizzle layout from a position-dependent swizzle pointer, because hardware swizzling depends on the shared-memory pointer address. Reuse a pinned complete layout/descriptor construction or prove the composition and required base alignment; a `Swizzle` type alone is not a complete TMA or MMA contract.

## Tensor-map construction checklist

For `cuTensorMapEncodeTiled`, record and validate all inputs rather than copying only the swizzle enumerator:

1. Use a correctly aligned `CUtensorMap` object and global base pointer.
2. Supply rank-sized global dimensions, rank-minus-one byte strides (the fastest dimension is implicit), rank-sized box dimensions, and rank-sized element strides.
3. Satisfy the global alignment, stride, box, interleave, datatype, and selected-swizzle constraints.
4. Check the returned `CUresult`; do not launch with an output descriptor after encoding failed.
5. Use a shared-memory base and tcgen05 descriptor that represent the same mapping as the tensor map.

An invalid encoder combination can fail explicitly. A successfully encoded tensor map paired with the wrong consumer layout may instead read the wrong logical elements, so a successful API return is not a correctness oracle.

## Source-reported tutorial result

Gau Nernst's pinned B200 tutorial compares M=N=K=4096 kernels with PyTorch 2.9.1 and CUDA 13. Its 3D TMA version changes from a 16-byte inner tile with no swizzle to a 128-byte inner tile with 128B swizzling and matching tcgen05 descriptors:

| Tutorial version | Author-reported TFLOP/s |
|---|---:|
| v1b: 3D 16B TMA | 252.81 |
| v2b: 3D 128B TMA plus 128B swizzle | 695.43 |

The combined change is approximately 2.75× and the v2b endpoint is about 46% of the tutorial's 1506.74 TFLOP/s cuBLAS result. It is not a bank-conflict or swizzle-only ablation. The author notes that the earlier contiguous `8×16B` tile might already span all 32 banks, lacked Nsight Compute access, and offers wider TMA transfers as an alternative explanation.

## Verification procedure

For each candidate mapping:

1. Keep global tensor, box, consumer instruction, tile shape, synchronization, and output oracle fixed. Change only the producer/consumer layout pair when an isolated comparison is possible.
2. Check every tensor-map encoder result and run correctness tests that distinguish rows, columns, tiles, boundaries, and out-of-bounds fill. Do not rely on random uniform values that can hide permutations.
3. Use the installed Nsight Compute's `--query-metrics` or Memory Workload Analysis rather than assuming one metric name exists on every chip/tool version. Current documentation includes `l1tex__data_bank_conflicts_pipe_lsu.sum` and request/wavefront/conflict analysis.
4. Compare executed shared-memory requests, ideal versus excessive wavefronts, bank conflicts, TMA behavior, total time, and occupancy. Zero observed CUDA-core bank conflicts neither proves the MMA descriptor matches nor explains a performance change by itself.
5. Record GPU, clocks, toolkit, profiler, compiler, exact addresses/alignment, layout types, descriptor fields, warmup, repetitions, and trial statistic.

Select the mapping that is legal and correct for both producer and every consumer, then retain it only where the controlled target workload improves. Do not infer the choice from architecture name, element width, or “MMA versus non-MMA” alone.
