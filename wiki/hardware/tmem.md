---
id: hw-tmem
title: "Tensor Memory (TMEM)"
type: hardware
architectures: [sm100, sm100a]
tags: [tmem, tcgen05]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tcgen05-mma, hw-2sm-cooperative, technique-double-buffering, pattern-register-pressure]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139]
aliases: [TMEM, "tensor memory", "Tensor Memory"]
---

# Tensor Memory (TMEM)

## What TMEM is

Tensor Memory is an addressable, on-chip memory introduced for fifth-generation Tensor Core operations on Blackwell. Each SM has 256 KiB, organized as 128 lanes by 512 columns of 32-bit cells. `tcgen05.mma` writes its D accumulator to TMEM; depending on the instruction form, A can also reside there.

Moving D out of general-purpose registers changes the resource balance relative to Hopper WGMMA. For example, CUTLASS 4.5.0's SM90 m64n256 FP32 WGMMA wrapper exposes 128 accumulator registers **per participating thread**. TMEM avoids that particular register-resident D fragment, but it does not remove registers needed by operands, control flow, or the epilogue.

## Addressing and warp access

A TMEM address (`taddr`) is a 32-bit value:

| Bits | Meaning |
|---|---|
| 31:16 | TMEM lane |
| 15:0 | TMEM column |

The lane is a TMEM data-path lane, not storage independently owned by a CUDA thread. `tcgen05.ld` and `tcgen05.st` are warp-collective. Under their four-warp access model, the warps cover these lane chunks:

| Participating warp | TMEM lanes |
|---|---|
| 0 | 0-31 |
| 1 | 32-63 |
| 2 | 64-95 |
| 3 | 96-127 |

The chosen load/store shape determines how values from that lane-column region map to each thread's register vector. Use the PTX shape-specific layout tables rather than treating a logical MxN accumulator as a universal row-major array owned one row per thread.

## Allocation lifecycle

TMEM has an explicit software-managed lifetime:

1. Reserve shared memory for the 32-bit allocation result.
2. Execute `tcgen05.alloc` collectively from one warp for `cta_group::1`, or from two warps—one per paired CTA—for `cta_group::2`.
3. Synchronize consumers as required, then read the returned base `taddr` from shared memory.
4. Use the allocation for MMA, copy, load, or store operations with the same CTA-group mode.
5. Execute `tcgen05.dealloc` with the corresponding collective issue pattern.
6. Deallocate every TMEM allocation before kernel exit.

The allocation operand counts columns. Legal allocation sizes are powers of two from 32 through 512 columns. Allocation can block until the requested TMEM is available; the ISA does not define folklore outcomes such as silent corruption for an invalid size.

### Column budgeting

All allocations share the SM's 512-column capacity. Some useful whole-allocation budgets are:

| Allocation per stage | Maximum stages if TMEM has no other use | Columns left |
|---|---:|---:|
| 128 columns | 4 | 0 |
| 256 columns | 2 | 0 |
| 512 columns | 1 | 0 |

A logical need of 192 columns cannot be requested directly: reserve 256 columns, or suballocate that logical region within another legal power-of-two reservation. Two 256-column accumulator stages consume the entire capacity, so scale-factor tensors or other scratch state must fit inside those reservations or the pipeline must use fewer columns.

## Completion and ordering

The relevant mechanisms are distinct:

- `tcgen05.commit` attaches completion of prior asynchronous MMA operations to an mbarrier. Wait on that barrier before consuming their results.
- `tcgen05.wait::ld` and `tcgen05.wait::st` are the completion mechanisms for the corresponding asynchronous TMEM load/store operations.
- `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` order tcgen05 operations around a documented execution-ordering handoff. A fence is not an MMA completion wait.
- Producer buffers must remain live until the asynchronous operation that reads them has completed according to its instruction contract.

CTA synchronization alone does not replace these completion operations.

## Data movement

`tcgen05.ld` transfers TMEM into registers, and `tcgen05.st` transfers registers into TMEM. Their documented shapes include `.16x64b`, `.16x128b`, `.16x256b`, `.32x32b`, and `.16x32bx2`, with supported repetition qualifiers. There is no scalar `.32x1b` form.

`tcgen05.cp` performs shaped shared-memory-to-TMEM copies. For example, PTX ISA 9.0 defines this exact form:

```ptx
tcgen05.cp.cta_group::1.128x256b [taddr], sdesc;
```

Here `taddr` is the TMEM destination and `sdesc` is the 64-bit shared-memory matrix descriptor. Follow the instruction's asynchronous ordering and completion rules before reusing the source or consuming the destination.

## Staging accumulators

Multiple accumulator stages can overlap MMA production with draining a different, completed TMEM stage. The design is valid only when:

- the combined allocations and suballocations fit the 512-column budget;
- producer and consumer roles use explicit pipeline barriers;
- MMA completion is observed before a consumer drains a stage;
- TMEM load completion is observed before register results are used; and
- no stage is recycled while an asynchronous operation can still access it.

This is a pipeline design, not an automatic consequence of alternating two addresses. CUTLASS's version-pinned SM100 tutorials show complete producer/consumer implementations.

## CUTLASS 4.5.0 Python DSL

CUTLASS 4.5.0 exposes TMEM through `cutlass.utils.TmemAllocator` and layout-aware CuTe tensors. The official tutorial sequence is:

1. create a `TmemAllocator` over shared storage for the allocation result;
2. call `allocate(num_columns)` from the configured allocator warp;
3. use `wait_for_alloc()` before other warps retrieve the address;
4. obtain a typed pointer with `retrieve_ptr(dtype)`;
5. bind that pointer to the MMA accumulator layout with `cute.make_tensor`;
6. drain it with `tcgen05` copy atoms and the pipeline's completion protocol; and
7. call `free(tmem_ptr)` before exit.

See the pinned [CUTLASS 4.5.0 FP16 GEMM tutorial](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py) and [TMEM allocator implementation](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/python/CuTeDSL/cutlass/utils/tmem_allocator.py). APIs such as `Layout(memory_space=MemorySpace.TMEM)` and standalone `tcgen05_mma()` are not CUTLASS 4.5.0 interfaces.

## Primary references

- [PTX ISA 9.0: Tensor Memory](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensor-memory)
- [PTX ISA 9.0: `tcgen05.alloc`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit)
- [PTX ISA 9.0: `tcgen05.ld`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld)
- [PTX ISA 9.0: `tcgen05.commit`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tcgen-async-sync-operations-commit)
- [CUTLASS 4.5.0 SM100 Python tutorial](https://github.com/NVIDIA/cutlass/tree/e406c186f510a15091cce01f782020ceb7ba8eb5/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm)
