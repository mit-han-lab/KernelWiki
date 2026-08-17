---
id: migration-register-to-tmem
title: "Register Accumulators to TMEM"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tmem, tcgen05]
related: [hw-tmem, hw-tcgen05-mma, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-cutlass-2139, pr-vllm-22738]
blackwell_relevance: "SM90 wgmma exposes distributed register accumulators; SM100 tcgen05 writes its destination to explicitly allocated TMEM."
confidence: verified
evidence_basis:
  - {source_id: doc-ptx-isa-sm100, evidence_type: official-doc}
  - {source_id: pr-cutlass-2139, evidence_type: upstream-code}
reproducibility: pseudocode
---

# Register Accumulators to TMEM

## What changes

SM90 `wgmma` returns accumulator fragments in registers distributed across the warpgroup. SM100 `tcgen05.mma` writes a layout-defined destination in TMEM. Migrating therefore replaces an implicit register lifetime with an explicit allocation, asynchronous completion, fragment-load, and deallocation lifecycle.

```python
def migrate_accumulator(k_tiles, tmem_columns):
    base = allocate_tmem_columns(tmem_columns)
    for index, operands in enumerate(k_tiles):
        issue_tcgen05_mma(base, operands, enable_input_d=(index != 0))
    commit_mma_to_mbarrier()
    wait_for_mma_completion()
    for fragment in assigned_tmem_fragments(base):
        registers = load_tmem_fragment(fragment)
        run_epilogue(registers)
    deallocate_tmem_columns(base, tmem_columns)
```

## Resource implications

Moving the destination out of registers can substantially lower **accumulator** register demand. It does not set total registers to a small constant or guarantee occupancy: descriptors, pipeline state, address arithmetic, epilogue fragments, and fused operations still use registers, while TMEM, shared memory, threads, and clusters add other residency limits.

Compute register and TMEM use from the compiled kernel and the selected instruction layout. Hand-written budgets such as “29 registers per thread” or “a 128-by-256 tile always uses 256 columns” are not architecture facts.

## Completion and ordering

Do not replace a WGMMA wait with `tcgen05.fence` plus `__syncthreads()`. Commit MMA completion to an `mbarrier` and wait before the epilogue reads TMEM. Use the paired tcgen fence protocol where work issued by one thread is ordered around cross-thread synchronization. Async TMEM load/store variants have their own `tcgen05.wait::ld`/`::st` requirements.

## Epilogue and overlap

An epilogue must load a documented TMEM fragment into registers before applying ordinary arithmetic. Multiple TMEM accumulator stages can permit overlap with another work unit, but only if their column regions are disjoint and stage-release barriers cover every user. Register-based SM90 schedules can also overlap work through multiple warpgroup/accumulator strategies, so overlap is not literally impossible there; its resource tradeoff differs.

## Avoid invented latency tables

The cited official sources do not specify universal 4-cycle register, 30-cycle shared-memory, 200-cycle spill, or 420-cycle TMEM latencies. Such values depend on dependency structure, cache state, instruction form, conflicts, and target. Use a pinned microbenchmark if latency is material.

## Migration checks

- Derive TMEM columns and lane mapping from the exact MMA/load traits.
- Test both overwrite (`enable_input_d = 0`) and accumulate paths.
- Wait for all MMA/TMEM operations before reuse or deallocation.
- Retune tile and stage counts; larger is not automatically faster.
- Compare output error and epilogue rounding with the SM90 reference.
