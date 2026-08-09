---
id: lang-cute-dsl
title: "CuTe DSL for Blackwell"
type: language
tags: [cute-dsl, tcgen05, tmem, tma]
related: [hw-tcgen05-mma, hw-tmem, kernel-flash-attention-4, doc-cutlass-blackwell]
sources: [doc-cutlass-blackwell, blog-colfax-cutlass, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-cutlass-blackwell
    evidence_type: official-doc
---

## Scope

CUTLASS v4.5.0 includes CuTe DSL, a Python-native interface for authoring GPU kernels, alongside CUTLASS's C++ template interfaces. This page uses the exact `v4.5.0` tag (`e406c186f510a15091cce01f782020ceb7ba8eb5`); rolling `latest` documentation can contain later names.

FlashAttention-4 is implemented entirely in CuTe DSL. Its author reports roughly 20--30x shorter compile times than C++ templates and a peak of 1605 TFLOP/s on B200 BF16. These are FA4-specific, source-reported results. The 1605-TFLOP/s result was not presented as a matched comparison with a handwritten-C++ FA4 kernel.

## F16/BF16 MMA construction

CUTLASS 4.5.0 uses a configured `tcgen05.MmaF16BF16Op`, not an `SM100_MMA_F16BF16_SS` Python symbol. This excerpt shows the one-CTA operation used by the tagged first tutorial; it is construction code, not a complete kernel.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

op = tcgen05.MmaF16BF16Op(
    cutlass.Float16,
    cutlass.Float32,
    (128, 256, 16),
    tcgen05.CtaGroup.ONE,
    tcgen05.OperandSource.SMEM,
    cute.nvgpu.OperandMajorMode.K,
    cute.nvgpu.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op)
```

The two-CTA tutorial instead uses instruction shape `(256, 256, 16)` and `tcgen05.CtaGroup.TWO`. The accumulator fragment is rebound to an allocated TMEM pointer before `cute.gemm` issues the operation.

## TMEM lifecycle and epilogue copy

The tagged examples use `cutlass.utils.TmemAllocator` around a shared holding buffer. A complete path allocates columns, waits before pointer retrieval, rebinds the accumulator tensor, partitions a typed `tcgen05` TMEM-to-register copy across participating threads, synchronizes readers, and frees the same allocation.

```python
# Excerpt: storage/barrier/tensor definitions and pipeline edges are required.
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf.ptr,
    barrier_for_retrieve=tmem_alloc_barrier,
)
tmem.allocate(num_tmem_cols)
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

tmem_atom = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32
)
tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
# get_slice(), partition_S/D(), and cute.copy() form the per-thread epilogue.

pipeline.sync(barrier_id=1)
tmem.free(tmem_ptr)
```

The excerpt deliberately does not imply that allocation or deallocation is lane-local. Use the complete tutorial for collective participation, two-CTA handling, and pointer lifetime.

## TMA construction

The one-CTA tutorial constructs a typed global-to-shared TMA operation and then derives operand-specific tiled atoms. Kernel code subsequently uses `tma_partition` and `cute.copy` with pipeline barriers.

```python
from cutlass.cute.nvgpu import cpasync, tcgen05

tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    tma_op,
    a,
    a_smem_layout_one_stage,
    mma_tiler_mnk,
    tiled_mma,
)
```

For the two-CTA tutorial, the corresponding operation is `CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` and the cluster layout/multicast participants must agree with the launch.

## Warp specialization and layouts

`fp16_gemm_2.py` specializes TMA, MMA, and epilogue warps and uses `PipelineTmaUmma` plus `PipelineUmmaAsync` to represent full/empty ownership. It also retains explicit TMEM allocation, epilogue TMEM-copy partitioning, TMA-store completion, pipeline tails, and collective teardown. Those details are required; an `if warp_id` sketch alone is not a safe synchronization recipe.

CuTe supplies typed layout algebra and Blackwell helpers such as `make_smem_layout_a` and `make_smem_layout_b`. The author still supplies the MMA tiler, datatypes, operand major modes, alignment, cluster shape, and pipeline policy; layout and swizzle choices are helper-assisted rather than universally automatic.

## Verbatim upstream artifacts

The following files are untruncated, verbatim captures. Their bytes match the SHA-256 records in each PR's `PROVENANCE.yaml`, and the seven numeric sizes below are physical line counts. To inspect through the repository helper, run `conda run -n base python scripts/get_page.py <pr-id> --include-code`.

| File | Upstream purpose | Lines |
|---|---|---:|
| [`fp16_gemm_0.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) | Software-pipelined FP16 tutorial baseline | 447 |
| [`fp16_gemm_1.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py) | Add two-CTA MMA and TMA multicast | 535 |
| [`fp16_gemm_2.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py) | Add TMA/MMA/epilogue warp specialization | 679 |
| [`fp16_gemm_3.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py) | Add static persistent scheduling | 769 |
| [`fp16_gemm_4.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py) | Add preferred and fallback/dynamic clusters | 1065 |
| [`fp16_gemm_5.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py) | Add TMA data prefetch | 919 |
| [`fp16_gemm_6.py`](../../artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py) | Add Programmatic Dependent Launch | 1002 |
| [`dense_gemm_persistent_prefetch.py`](../../artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py) | Persistent GEMM with TMA prefetch | full capture |
| [`clc.py`](../../artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py) | CLC Python binding | full capture |

PR 3106's series is an official progressive tutorial: software-pipelined FP16, two-CTA multicast, warp specialization, static persistence, preferred/dynamic clusters, prefetch, and PDL. In the v4.5.0 tag, the same series is under `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/`.

## Primary references

- [CUTLASS v4.5.0 tag](https://github.com/NVIDIA/cutlass/tree/e406c186f510a15091cce01f782020ceb7ba8eb5)
- [Pinned v4.5.0 tutorial series](https://github.com/NVIDIA/cutlass/tree/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm)
- [FlashAttention-4 first-party post](https://tridao.me/blog/2026/flash4/)

## Related

- [tcgen05-mma](../hardware/tcgen05-mma.md) — underlying instruction semantics
- [flash-attention-4](../kernels/flash-attention-4.md) — evidence-scoped CuTe DSL case study
- [CUTLASS Blackwell source card](../../sources/docs/nvidia-cutlass-blackwell.md) — version-pinned source routes
