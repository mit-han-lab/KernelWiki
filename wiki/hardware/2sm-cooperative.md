---
id: hw-2sm-cooperative
title: "Two-SM Cooperative MMA"
type: hardware
architectures: [sm100, sm100a]
tags: [2sm-cooperative, tcgen05, cluster]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: blog-tcgen05-tutorial
    evidence_type: source-reported
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2139]
aliases: ["2-SM cooperative", "dual CTA", "2CTA", "cta_group::2"]
---

# Two-SM Cooperative MMA

## Execution model

PTX defines a CTA pair as two CTAs in the same cluster whose `%cluster_ctarank` values differ only in the low bit. With `cta_group::2`, one thread from either CTA can initiate a whole `tcgen05.mma`; the peer CTA must still be active. The operation accesses Tensor Memory belonging to both the current CTA and its peer.

Allocation management differs from MMA issue. A `tcgen05.alloc` or `tcgen05.dealloc` for `cta_group::2` is issued collectively by two warps, one in each CTA. All tcgen05 instructions in the kernel must use the same CTA-group value.

## Shape is independent of group size

`cta_group::2` selects pair-level resources; it does not by itself select one fixed MxNxK shape. The instruction descriptor (`idesc`) encodes M, N, exact operand and accumulator types, sparsity, and related operation details. Legal values depend on the kind, data-path layout, CTA group, and target ISA.

For example, m256xn256xk16 is a useful maximum F16/BF16 configuration, but PTX also defines group-2 layouts with other M/N values. Do not infer that every pair-level MMA is created by mechanically doubling a group-1 M dimension.

The complete operand grammar, including the eight-register disable-output-lane vector used by `cta_group::2`, is documented on the related [tcgen05.mma page](tcgen05-mma.md).

## Correctness requirements

A pair-level kernel must account for all of these invariants:

- The two CTAs form a valid CTA pair, and the peer remains active while a group-2 operation is issued.
- Pair-level TMEM allocation and deallocation follow their two-warp collective issue rule.
- SMEM and TMEM operands use legal layouts and descriptors for the selected instruction configuration.
- Source storage is not overwritten while the asynchronous MMA may still consume it.
- `tcgen05.commit.cta_group::2` and an mbarrier wait provide completion tracking. Fences and execution-ordering operations are added where tcgen05-visible state crosses threads or CTAs.

## Source-reported performance

In Gau Nernst's M=N=K=4096 experiment on a Modal B200 with PyTorch 2.9.1 and CUDA 13, v4 warp specialization reports 1208.83 TFLOP/s and v5 2-SM MMA reports 1302.29 TFLOP/s. Relative to the same 1506.74-TFLOP/s cuBLAS result, those are approximately 80.2% and 86.4%, or a 7.7% relative increase from v4 to v5.

That result establishes a gain for one kernel and setup, not a universal threshold. Choose between group 1 and group 2 by benchmarking the target shapes and accounting for CTA pairing, data reuse, layout, occupancy, pipeline stages, and epilogue cost. Persistence is an independent scheduling choice, not a condition that guarantees maximum throughput.

## Related

- [tcgen05.mma](tcgen05-mma.md) — instruction grammar, completion, and descriptors
- [Tensor Memory](tmem.md) — TMEM allocation, addressing, and access
