---
id: lang-cute-dsl
title: "CuTe DSL for Blackwell"
type: language
tags: [cute-dsl, tcgen05, tmem, tma]
related: [hw-tcgen05-mma, hw-tmem, kernel-flash-attention-4, doc-cutlass-blackwell, doc-cutlass-cute-dsl]
sources: [doc-cutlass-blackwell, doc-cutlass-cute-dsl, blog-colfax-cutlass, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## Overview

NVIDIA introduced CuTe DSL in CUTLASS 4.0 as a Python programming model that
retains CuTe's layouts, tensors, hardware atoms, and explicit hardware
hierarchy. The maintained documentation covers SM100 tcgen05 programming as
well as TMA, TMEM, synchronization, control flow, compilation, and debugging.

CuTe DSL is an abstraction, not a promise that allocation, synchronization,
or layout constraints disappear. In particular, do not translate a schematic
name such as `alloc_tmem()` or `mma()` into code unless it exists in the pinned
API. The earlier version of this page did so and has been removed.

## Exact SM100 pipeline excerpt

CUTLASS PR 3106 contains a complete SM100 tutorial series. In its step-2
warp-specialized kernel, the MMA role waits for a filled TMA stage, changes the
tcgen05 accumulate field after the first K tile, executes `cute.gemm`, and
releases the shared-memory stage:

```python
handle = ab_consumer.wait_and_advance()

tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
tile_crd = (None, None, None, handle.index)
cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

handle.release()
```

This is a contiguous excerpt from the retained upstream file, not a standalone
kernel. The full file is the source for its pipeline setup, TMEM allocation,
barrier participants, TMA partitioning, and epilogue.

## Full Examples (verbatim upstream code shipped locally)

The following CuTe DSL files ship **verbatim** in this repository under `artifacts/prs/cutlass/` (pinned at each PR's merge SHA). Open them with `python3 scripts/get_page.py <pr-id> --include-code` or read them directly.

| File | Purpose | Size |
|---|---|---|
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) | Step 0 — FP16 GEMM baseline | 447 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py) | Step 1 — 2CTA MMA + TMA multicast | 535 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py) | Step 2 — Warp specialization (TMA / MMA / epilogue warps) | 679 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py) | Step 3 — Static persistent tile scheduler | 769 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py) | Step 4 — Preferred + dynamic clusters | 1065 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py) | Step 5 — TMA prefetch | 919 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py) | Step 6 — Programmatic Dependent Launch (PDL) | 1002 lines |
| [`artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py`](../../artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py) | Persistent GEMM with prefetch | full |
| [`artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py`](../../artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py) | CLC (Cluster Launch Control) Python binding | full |

The `fp16_gemm_{0..6}.py` series from `examples/python/CuTeDSL/blackwell/tutorial_gemm/` in NVIDIA/cutlass PR-3106 is the authoritative CuTe DSL learning path: it walks from a naive FP16 GEMM baseline through 2CTA MMA with TMA multicast, warp specialization, static persistent scheduling, preferred / dynamic clusters, TMA prefetch, and ends with Programmatic Dependent Launch (PDL). Reading them in order is the recommended on-ramp.

## Related
- [tcgen05-mma](../hardware/tcgen05-mma.md) — Underlying MMA instruction
- [flash-attention-4](../kernels/flash-attention-4.md) — CuTe-DSL implementation
- [CUTLASS Blackwell docs](../../sources/docs/nvidia-cutlass-blackwell.md) — Official reference
- [Full CuTe DSL documentation](../../sources/docs/cutlass-cute-dsl.md) — Compiled official Python DSL documentation
