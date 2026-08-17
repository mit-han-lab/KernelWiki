---
id: migration-wgmma-to-tcgen05
title: "Migrating from wgmma to tcgen05"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tcgen05, wgmma, tmem]
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-cutlass-blackwell, pr-cutlass-2139]
blackwell_relevance: "SM100 tcgen05 changes the issue model, accumulator location, descriptors, instruction shapes, and completion protocol relative to SM90 wgmma."
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
reproducibility: pseudocode
---

# Migrating from `wgmma` to `tcgen05`

## The architectural change

SM90 `wgmma.mma_async` is a warpgroup operation with distributed register accumulators. SM100 `tcgen05.mma` is issued by one thread and writes a CTA-owned TMEM allocation. This is not a mnemonic substitution: operand descriptors, supported shapes and kinds, ownership, completion, and epilogue mapping all change.

## Migration checklist

- Select an exact documented `tcgen05.mma` form, including CTA group, kind, shapes, datatypes, and instruction descriptor.
- Allocate the required TMEM columns and distribute the returned address according to the PTX ownership rules.
- Build A and B operands in a legal layout. B is a shared-memory descriptor; A can be a shared-memory descriptor or TMEM for supported forms.
- Preserve the requested accumulation semantics: `enable_input_d = 0` computes `D = A*B`; `1` computes `D = A*B+D`.
- Establish MMA completion with `tcgen05.commit` to an `mbarrier`, then wait on that barrier before dependent TMEM loads or reuse.
- Use `tcgen05.fence::before_thread_sync` and `::after_thread_sync` where the PTX memory model requires ordering around cross-thread synchronization. A fence is not a completion wait.
- Load accumulator fragments from TMEM for the epilogue, then deallocate the same allocation consistently across the CTA or CTA pair.
- Retune the whole schedule. Warp count, pipeline depth, tile shape, and swizzle are configuration choices.

## Dependency-oriented sketch

```python
def sm100_mainloop(k_tiles, tmem_stage, completion_barrier):
    initialize_tmem_allocation(tmem_stage)
    for index, tile in enumerate(k_tiles):
        wait_until_smem_operands_ready(tile)
        enable_input_d = index != 0
        issue_tcgen05_mma(tmem_stage, tile, enable_input_d)

    commit_tcgen05_to_mbarrier(completion_barrier)
    wait_on_mbarrier(completion_barrier)
    result = load_tmem_for_epilogue(tmem_stage)
    deallocate_tmem(tmem_stage)
    return result
```

This sketch deliberately omits inline PTX: instruction signatures vary by kind, and plausible-looking shortened signatures are often invalid.

## Operand layouts and swizzle

Do not mechanically change a Hopper 64-byte swizzle to 128 bytes. The SM100 matrix-descriptor tables permit different swizzle modes for different layouts. Select a legal descriptor from the PTX table and make the TMA/shared-memory producer match it exactly.

## Warp specialization

Only one thread issues `tcgen05.mma`, but a production CTA still uses warps for scheduling, loads, epilogues, and barriers. Current CUTLASS SM100 TMA warp-specialized GEMM code uses one warp each for MMA, scheduler, mainloop load, and epilogue load, followed by a configuration-dependent number of epilogue warps. This is one library schedule, not an architectural requirement or a promise that five warps are optimal.

## Common failure modes

- Treating a fence plus `__syncthreads()` as MMA completion.
- Passing an SMEM descriptor where a TMEM address is required, or vice versa.
- Using an undocumented kind such as `kind::mxf8` instead of `kind::mxf8f6f4`.
- Assuming `enable_input_d = 0` zeroes an existing accumulator after the operation; it selects non-accumulating MMA semantics for that issue.
- Reusing or deallocating TMEM while an MMA or epilogue still accesses it.
- Copying a swizzle, tile shape, or warp map without checking the selected instruction form and resource limits.

## CUTLASS path

When practical, select a CUTLASS SM100 kernel and epilogue schedule matching the datatype and 1-SM/2-SM mode. CUTLASS encodes the instruction traits, TMEM storage, pipeline barriers, and legal epilogue policies, but application-level alignment, shape, workspace, and numerical checks remain the caller's responsibility.
