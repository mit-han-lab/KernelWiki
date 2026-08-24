---
id: technique-epilogue-fusion
title: Epilogue fusion
type: technique
architectures: [sm100, sm90]
tags: [epilogue-fusion, tmem, warp-specialization]
confidence: source-reported
reproducibility: snippet
prerequisites: [technique-warp-specialization]
related: [technique-warp-specialization, hw-tmem, technique-double-buffering]
sources: [doc-cutlass-blackwell, pr-vllm-16032]
blackwell_relevance: CUTLASS SM100 collectives expose typed epilogue builders; the chosen schedule and synchronization determine whether epilogue work overlaps the mainloop.
artifact_dir: artifacts/kernels/epilogue-fusion
---

# Epilogue fusion

A GEMM epilogue converts accumulator values into output values and may combine
scaling, a source tensor, bias, activation, or quantization before the store.
Fusion can avoid a separate launch and intermediate global-memory round trip.
Overlap with a later mainloop is a separate scheduling property and requires an
explicit buffer-ownership protocol.

vLLM PR 16032 supplies a retained SM100 CUTLASS configuration. This contiguous
excerpt from its pinned upstream file uses the collective builder rather than a
hand-written TMEM-load pseudo-API:

```cpp
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
        LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
```

Use the complete builder and generated kernel for its supported operations.
Blackwell does not prescribe 14 epilogue warps, a 256/256 TMEM split, or one
universal visitor type. The former page and derived artifact asserted all three
and were removed.

Correctness tests must cover alpha/beta or source-tensor semantics, activation
edge cases, output conversion, tails, and buffer reuse. Performance attribution
must distinguish fusion savings from mainloop or tile-shape changes.
