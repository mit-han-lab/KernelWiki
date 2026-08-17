---
id: kernel-grouped-gemm
title: "Grouped GEMM for MoE"
type: kernel
architectures: [sm100, sm100a, sm90]
tags: [grouped-gemm, moe, gemm, fp8, nvfp4, tcgen05, persistent-kernel, tile-scheduling]
confidence: source-reported
reproducibility: snippet
kernel_types: [grouped-gemm, gemm, moe]
languages: [cuda-cpp, cute-dsl]
related: [kernel-fused-moe, kernel-deepgemm, hw-tcgen05-mma, hw-clc, technique-persistent-kernels, technique-tile-scheduling]
sources: [contest-gpumode-p4, blog-deepgemm, blog-gpu-mode-reward-hack, doc-cutlass-blackwell]
performance_claims: []
blackwell_relevance: "SM100 grouped kernels combine pointer/shape metadata with persistent software scheduling; CLC can transfer unlaunched cluster IDs but is not itself a grouped-tile queue."
---

# Grouped GEMM for MoE

## Operation

Grouped GEMM evaluates a set of independent products:

```python
def grouped_reference(groups):
    return [matmul(group.a, group.b) for group in groups]
```

MoE inference commonly has expert-specific weights and variable token counts, but grouped-GEMM APIs differ. CUTLASS pointer-array interfaces can describe per-group pointers/shapes. DeepGEMM's current M-grouped interfaces specialize the common case where M varies while N/K are fixed; its K-grouped weight-gradient API is a separate contract.

## DeepGEMM layouts

- Contiguous M-grouped input packs expert token rows and uses group-layout metadata; current upstream documentation requires each expert segment to satisfy a configurable M-block alignment.
- Masked M-grouped input supports graph-friendly fixed storage and supplies valid M counts/masks so the kernel avoids invalid work according to its implementation.
- K-grouped APIs target weight-gradient-style products and have fixed-dimension constraints; they should not be represented as a generic concatenated-K struct without checking the upstream signature.

## Scheduling

A scheduler maps global logical tile IDs to `(group, tile_m, tile_n)`. Options include precomputed prefix/range tables, pointer-array metadata, persistent atomic counters, or CUTLASS schedulers. On SM100, CLC can let a running cluster take the launch ID of an unlaunched cluster; software still maps that ID to a grouped tile.

```python
def grouped_worker(scheduler):
    while (work := scheduler.next()) is not None:
        group, tile = work
        descriptors = build_or_load_group_descriptors(group, tile)
        compute_and_store_bounded_tile(group, tile, descriptors)
```

Correctness requires exactly-once coverage, group-specific leading dimensions/scales, zero-sized groups, and bounded tail stores. TMA alignment is descriptor- and datatype-specific, not a universal 128-byte base requirement for every group.

## Reward-hack boundary

GPU Mode's separate post-mortem marks the displayed 11.191-microsecond score invalid because the submission exploited cross-invocation state in the evaluation/timing mechanism. It is not a kernel performance result and is therefore absent from this page's `performance_claims`. The current public API no longer lists it. In the ended API's B200-group snapshot on 2026-08-16, `gau.nernst` led at 13.092434 us; the API also carries a separate NVIDIA group, so every score requires a date and group and is not independently promoted to an award claim.

## Performance considerations

- Small M can waste tile work; smaller/specialized tiles can add dispatch overhead.
- Group transitions require metadata and potentially new TMA descriptors.
- Packing can improve utilization but must not mix scale/layout ownership across experts.
- Static order may improve locality; dynamic order may improve balance. Benchmark both with realistic routing distributions.
