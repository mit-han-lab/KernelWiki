---
id: lang-cute-dsl
title: "CuTe DSL for Blackwell"
type: language
tags: [cute-dsl, tcgen05, tmem, tma]
related: [hw-tcgen05-mma, hw-tmem, kernel-flash-attention-4, doc-cutlass-blackwell, doc-cutlass-cute-dsl]
sources: [doc-cutlass-blackwell, doc-cutlass-cute-dsl, blog-colfax-cutlass, blog-flash-attention-4, pr-cutlass-3106]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - {source_id: doc-cutlass-cute-dsl, evidence_type: official-doc}
  - {source_id: pr-cutlass-3106, evidence_type: upstream-code}
---

# CuTe DSL for Blackwell

## Overview

CuTe DSL is CUTLASS's Python-hosted DSL for constructing GPU kernels with CuTe layouts, copy atoms, MMA atoms, pipelines, and compiled device functions. Its SM100 APIs expose TMA, TMEM allocation/load/store, `tcgen05` MMA variants, cluster operations, and layout utilities while preserving low-level control.

Names and signatures evolve with CUTLASS releases. Use the documentation and examples pinned to the installed version; abbreviated names such as `SM100_MMA_F16BF16_SS` or decorators such as `@cute_kernel` should not be assumed to be public APIs unless they appear in that release.

## Layout-driven construction

```python
def build_blackwell_kernel(problem, dtype, cutlass_version):
    tiled_mma = select_documented_mma(problem, dtype, cutlass_version)
    smem_layouts = derive_operand_layouts(tiled_mma, problem)
    tmem_layout = derive_accumulator_layout(tiled_mma, problem)
    tma_atoms = construct_tma_atoms(problem, smem_layouts)
    return compose_pipeline(tma_atoms, tiled_mma, tmem_layout)
```

This is conceptual code. The important property is that operand partitions, TMEM column count, register fragments, and scale layouts derive from the selected tiled MMA rather than from hand-written row-major assumptions.

## Warp specialization

CuTe DSL allows separate control-flow branches for scheduler, TMA producer, MMA issuer, and epilogue agents. The role IDs and counts are kernel choices. In the current CUTLASS C++ SM100 TMA warp-specialized GEMM, the fixed first four warps are MMA, scheduler, mainloop load, and epilogue load, followed by a configuration-dependent epilogue count; other CuTe DSL examples use different mappings.

Completion remains explicit: TMA stages use `mbarrier` transactions, MMA completion uses `tcgen05.commit` plus an `mbarrier` wait, and TMEM load/store forms use their documented waits where asynchronous. A layout abstraction does not remove these dependencies.

## Compilation claim

The FlashAttention-4 paper reports approximately 2.5/1.4 seconds for its CuTe-DSL forward/backward compilation versus 55/45 seconds for the compared FlashAttention-3 C++ implementation, summarized as 20–30x faster for that project. This does not establish a universal CuTe-DSL/C++ compilation ratio. The paper's 1,613 TFLOP/s peak is likewise a result for the full FA4 kernel, not the language in isolation.

## Verbatim upstream learning path

The captured NVIDIA/cutlass PR-3106 examples are a version-pinned progression:

| File | Step |
|---|---|
| [`fp16_gemm_0.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) | baseline |
| [`fp16_gemm_1.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py) | 2-CTA MMA and multicast |
| [`fp16_gemm_2.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py) | warp specialization |
| [`fp16_gemm_3.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py) | static persistent scheduling |
| [`fp16_gemm_4.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py) | preferred/dynamic clusters |
| [`fp16_gemm_5.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py) | TMA prefetch |
| [`fp16_gemm_6.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py) | PDL |

Read the imports and calls in these files rather than copying simplified pseudocode from this wiki.
