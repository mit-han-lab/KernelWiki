---
id: technique-epilogue-fusion
title: Epilogue Fusion
type: technique
architectures:
- sm100
- sm90
tags:
- epilogue-fusion
- tmem
- warp-specialization
confidence: verified
evidence_basis:
  - source_id: doc-cutlass-blackwell
    evidence_type: official-doc
  - source_id: pr-vllm-16032
    evidence_type: upstream-code
reproducibility: pseudocode
prerequisites:
- hw-tmem
- technique-warp-specialization
related:
- technique-warp-specialization
- hw-tmem
- technique-double-buffering
sources:
- doc-cutlass-blackwell
- blog-colfax-cutlass
- pr-vllm-16032
blackwell_relevance: SM100 tcgen05 accumulators are read from TMEM into registers before supported output transforms and stores.
artifact_dir: artifacts/kernels/epilogue-fusion
---

# Epilogue Fusion

## Definition and boundary

Epilogue fusion computes output transformations—such as `alpha*acc + beta*C`, bias, activation, output conversion, or auxiliary values—inside the producer kernel before final output materialization. It can remove an intermediate tensor or launch only when the unfused comparison would otherwise materialize or launch that work.

Fusion does not by itself imply overlap with the next MMA tile. A schedule may drain one completed accumulator after the mainloop, or it may use disjoint storage and specialized roles to overlap a completed region's epilogue with independent MMA work. Participant counts and role IDs come from the concrete kernel; there is no architectural default of fourteen epilogue warps.

## SM100 TMEM-to-output path

For tcgen05, D resides in TMEM. An ordinary arithmetic epilogue first uses a legal collective `tcgen05.ld` mapping, or a library wrapper around it, to transfer a partition into per-thread registers. The load is asynchronous with respect to the issuing thread, so the matching `tcgen05.wait::ld` completion must occur before those registers are consumed.

CUTLASS 4.5.0's `fp16_gemm_2.py` demonstrates the typed route:

```python
copy_atom_t2r = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
)
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi)
thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
tTR_rAcc = cute.make_rmem_tensor(..., cutlass.Float32)
cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)
```

This is an API-routing fragment, not standalone code: the official file supplies the exact tensor layouts, selected load shape, participant group, edge predicates, accumulator-completion handoff, register-to-SMEM conversion, TMA-store pipeline, tail, and TMEM cleanup.

## Correctness contract

| Boundary | Proof required before crossing it |
|---|---|
| tcgen05 MMA → TMEM reader | the relevant asynchronous MMA is committed and its completion observed |
| TMEM → result registers | the legal collective load completes before dependent arithmetic |
| registers → output | element/layout mapping and output-edge predicates cover exactly the valid coordinates |
| TMEM reader → region reuse | every reader's load has completed and the matching reusable barrier phase is released |
| kernel tail | all output stores complete as required and every TMEM allocation is collectively freed |

For a multi-region overlap schedule, simultaneously live TMEM regions must also be disjoint. Equal 256-column halves are one possible policy for a 512-column allocation, not an epilogue-fusion requirement. Reusable mbarriers need correct expected-arrival counts and phase/parity tracking.

## CUTLASS 4.5.0 fusion interface

`cutlass::epilogue::fusion::LinCombEltAct` has this parameter order:

```cpp
template <
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
struct LinCombEltAct;
```

For SM100 construction, `cutlass::epilogue::collective::CollectiveBuilder` receives architecture, operator class, tile/cluster shapes, `EpilogueTileAuto` or an explicit epilogue tile, accumulator/compute/C/D types and layouts, alignment, `EpilogueScheduleAuto` or an explicit supported schedule, and finally a supported fusion operation or callbacks type.

There is no CUTLASS 4.5.0 type named `Sm100EpilogueTmaWarpSpecialized`. Support is constrained by the exact architecture, operator class, schedule, tile, layout, alignment, datatype, and fusion callback combination. Use `Gemm::can_implement(arguments)` and a tagged example rather than reconstructing a builder signature from prose.

The vLLM PR 16032 NVFP4 wrapper is one pinned C++ construction example: it uses `CollectiveBuilder<... EpilogueTileAuto, ... EpilogueScheduleAuto>` and derives mainloop shared-memory stages with `StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>`.

## Operation shapes

| Operation | Data dependency |
|---|---|
| Source linear combination | `D = alpha*acc + beta*C` |
| Per-row/per-column bias | add a separately laid-out broadcast bias input |
| Activation | apply a supported functor such as ReLU, GELU, or SiLU to the output fragment |
| Output conversion/quantization | convert the accumulator fragment and, when required, generate/store scale metadata |
| Gated product | combine two values, for example `SiLU(gate) * up`; requires both inputs and a supported/custom callback |
| Online-softmax rescale | rescale a prior partial output after a new row maximum; requires the attention reduction state |
| Residual | combine a separately supplied residual/source tensor at the defined point in the expression tree |

These are operation categories, not a claim that every composition is supported by one built-in SM100 visitor. In particular, reductions, multiple outputs, auxiliary tensors, and broadcasts add synchronization and layout requirements beyond a pointwise activation.

## Overlap and performance evaluation

When an implementation overlaps the epilogue with independent MMA work, verify both directions of the handoff: compute-complete before read, and load/read-complete before overwrite. Include the prologue (no prior result), steady-state wraparound, final drain, and deallocation path in tests.

Evaluate a defined fused/unfused pair with identical inputs, output semantics, launch policy, warmup, and timing statistics. Record registers, spills, SMEM, TMEM columns, barriers, threads/CTA, cluster shape, and occupancy. Sweep supported epilogue group sizes, load shapes, output tiles, and store stages. The PTX ISA defines no equal-share bandwidth model for a fixed number of epilogue warps; diagnose TMEM-load, conversion, barrier, and store bottlenecks with generated code and profiler data.

Fusing output conversion can avoid a global FP32 intermediate when the baseline would write and reread that intermediate. It does not guarantee a speedup: added registers, shared-memory staging, synchronization, edge handling, or reduced occupancy can outweigh saved traffic.

Useful negative tests delay one reader, skip a barrier phase, reuse a TMEM region early, omit the final store tail, and exercise partial M/N tiles. Each fault should be detected by output comparison or a bounded watchdog.

## Artifact provenance

The artifact bundle contains mixed provenance modes:

- `full/nvfp4_scaled_mm_kernels.cu` is verbatim from vLLM merge `ed7a29d9f8b48978e3bbf43599d21b4de65387e0` and byte-matches SHA-256 `e8aed5ccb3dd9de26c3aeff159a242a46dcb7c7d8d0351b6c44ff1f8d2f7effa`.
- `full/tmem-load-into-registers-for-epilogue.cu` is an extracted historical snippet from a local source card; it is not an upstream-verbatim kernel and must not be treated as standalone safe code.
- `variants/01-double-buffered-tmem-epilogue-skeleton.cu` is explicitly labeled derived/not-upstream and is incomplete pseudocode.

Retrieve the page and bundle with:

```bash
conda run -n base python scripts/get_page.py technique-epilogue-fusion --include-code
```

## Primary references

- [PTX ISA 9.0 tcgen05 load and wait](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-ld)
- [PTX ISA 9.0 tcgen05 completion](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-commit)
- [CUTLASS 4.5.0 fusion operations](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/include/cutlass/epilogue/fusion/operations.hpp)
- [CUTLASS 4.5.0 collective builder](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/include/cutlass/epilogue/collective/collective_builder.hpp)
- [CUTLASS 4.5.0 CuTe DSL epilogue example](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_2.py)
- [vLLM PR 16032 NVFP4 wrapper](https://github.com/vllm-project/vllm/blob/ed7a29d9f8b48978e3bbf43599d21b4de65387e0/csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu)

## Related

- [Tensor Memory](../hardware/tmem.md) — load, wait, and lifetime rules
- [warp specialization](warp-specialization.md) — participant groups and role invariants
- [double buffering](double-buffering.md) — optional multi-region overlap protocol
