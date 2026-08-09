---
id: migration-register-to-tmem
title: "Register Accumulators to TMEM"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tmem, tcgen05]
related: [hw-tmem, hw-tcgen05-mma, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-vllm-22738]
blackwell_relevance: "Hopper WGMMA D fragments reside in per-thread registers; Blackwell tcgen05 D resides in dynamically allocated TMEM, changing accumulator lifetime and resource tradeoffs."
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
  - source_id: pr-vllm-22738
    evidence_type: upstream-code
reproducibility: pseudocode
---

# Register Accumulators to TMEM

## What actually changes

For Hopper `wgmma.mma_async.m64nNk16` with an FP32 accumulator, each warpgroup thread holds `N/2` FP32 D registers. At `N=256`, that is 128 registers per thread for D. WGMMA updates that register vector asynchronously; the program uses the WGMMA fence, commit-group, and wait-group mechanisms before dependent use.

For `sm_100a`, tcgen05 D is addressed in Tensor Memory rather than supplied as a per-thread register vector. The CTA-visible TMEM structure has 128 lanes by 512 columns of 32-bit cells, or 256 KiB when fully allocated. Allocation is dynamic in columns.

This removes the resident D vector from the ordinary register file. It does not remove all register pressure: descriptors, addresses, loop state, pipeline state, and each TMEM-to-register epilogue batch still need registers. Both Hopper and compute-capability-10.0 Blackwell have 64K 32-bit registers per SM and at most 255 registers per thread. Occupancy, spilling, and practical tile size therefore remain properties of the compiled kernel and launch configuration.

## Storage and lifecycle contrast

| Concern | Hopper WGMMA | Blackwell tcgen05 |
|---|---|---|
| Resident D | Per-thread register fragment | TMEM region addressed by `taddr` |
| Example FP32 D cost | `N/2` registers/thread; 128 at `N=256` | No resident per-thread D vector |
| First accumulation | Set WGMMA `scale-d` false to compute `D=A*B`, or initialize/use D when accumulation is intended | Set `enable-input-d` false to compute `D=A*B`, or explicitly initialize D when the algorithm needs another value |
| Compute completion | WGMMA commit/wait group | `tcgen05.commit` to an mbarrier, then wait before a consumer reads/reuses D |
| Epilogue access | Use the thread's D registers | Collective `tcgen05.ld`, then `tcgen05.wait::ld` before consuming result registers |
| Cleanup | Register lifetime ends with the thread | Matching collective `tcgen05.dealloc` before kernel exit |
| Double buffering | Two simultaneously live D fragments consume two register fragments | Two live outputs consume disjoint TMEM columns inside the allocation |

The table describes storage contracts, not an occupancy prediction. Instruction shape is also not necessarily the same as a composed CTA tile.

## Migration lifecycle

1. Choose one CTA-group mode for the kernel. For allocation, `nCols` is a power of two in `[32, 512]`; allocations are column-granular and cover all 128 lanes.
2. Have every lane of one designated warp execute the same `.cta_group::1` allocation. Synchronize before other threads read the 32-bit `taddr` written to shared memory. Two-CTA mode requires one warp in each live peer CTA.
3. Initialize and publish the barriers used by TMA and tcgen05. If the first MMA should compute only `A*B`, supply a false `enable-input-d` predicate instead of assuming a mandatory TMEM zero-store pass.
4. Issue MMA from the permitted elected thread with valid A/B descriptors, instruction descriptor, predicate, and lifetimes.
5. Attach completion of the relevant tcgen05 work to an mbarrier with `tcgen05.commit`. A fence controls execution ordering but is not a completion wait.
6. After the completion handoff, participating epilogue lanes execute a legal `tcgen05.ld` shape and `tcgen05.wait::ld` before using the loaded registers.
7. After all readers finish, every lane of the designated warp executes the matching deallocation on every kernel exit path. Allocation/deallocation address, column count, and CTA-group mode must agree.

Representative PTX ISA 9.0 forms are shown below. They omit declarations, descriptor construction, collective control flow, barriers, and inline-assembly constraints, so they are not a standalone kernel.

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [saddr], nCols;
tcgen05.mma.cta_group::1.kind::f16 [taddr], a_desc, b_desc, idesc, p;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [mbar];
tcgen05.ld.sync.aligned.32x32b.x1.b32 {r0}, [taddr];
tcgen05.wait::ld.sync.aligned;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
```

## Overlap without invented guarantees

TMEM makes it practical to keep multiple output regions independently addressable while different warp roles issue MMA and drain a completed region. Safe overlap needs three distinct proofs:

- compute completion occurs before an epilogue loads a region;
- all epilogue loads complete before that region is overwritten or deallocated;
- producer/consumer state and phase cannot alias a different pipeline stage.

TMEM is not the only way to overlap tensor-core and non-matmul work. FlashAttention-3 already uses warp specialization and matmul/softmax interleaving on Hopper with register accumulators. The migration benefit is the different storage and lifetime tradeoff, not a new theorem that overlap was previously impossible.

## FlashAttention-4 case study

FlashAttention-4's first-party paper places score/output accumulators in TMEM and assigns MMA, softmax, and correction work to specialized warp roles. The accompanying author post describes software-selected polynomial `exp2` work on CUDA cores and reports up to 1605 TFLOP/s on B200 BF16, labeled 71%. That is a source-reported maximum, not evidence for a universal register count, occupancy, or migration speedup.

## Evidence-driven retuning

Do not translate an instruction shape directly into a required CTA tile or assume that moving D guarantees no spills. For each concrete kernel:

1. Record the compiler's registers/thread, spill loads/stores, static/dynamic shared memory, barriers, threads/CTA, and cluster shape.
2. Compute occupancy with the CUDA occupancy APIs for the actual launch; do not equate one warpgroup with one CTA.
3. Sweep legal instruction descriptors, CTA tiles, pipeline depth, TMEM column partitioning, and epilogue batch width.
4. Benchmark identical shapes, datatypes, outputs, synchronization, warmup, and trial statistics. A fixed latency such as “420 cycles per TMEM load” is not an ISA guarantee.
5. Inspect generated PTX/SASS to confirm that the intended tcgen05 shapes, waits, and no unexpected spills are present.

## Common migration errors

- TMEM is not an ordinary CUDA pointer space. Load through a legal collective tcgen05 copy operation before scalar arithmetic.
- Do not issue collective allocation, load/store, or deallocation from lane 0 alone.
- Do not substitute `tcgen05.fence` or `__syncthreads()` for asynchronous MMA completion.
- Do not zero TMEM unconditionally when a false `enable-input-d` predicate provides the intended first-write semantics.
- Do not treat TMEM-resident D as the entire kernel's register allocation; the epilogue can still spill.
- Do not describe missing deallocation as a defined recoverable “leak.” The normative rule is explicit matching deallocation before exit.

## Primary references

- [PTX ISA 9.0 WGMMA register fragments](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16)
- [PTX ISA 9.0 Tensor Memory](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensor-memory)
- [PTX ISA 9.0 tcgen05 allocation](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc)
- [PTX ISA 9.0 tcgen05 MMA](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma)
- [CUDA 13.0.2 Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html#occupancy)
- [CUDA 13.0.2 Blackwell Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-tuning-guide/index.html#occupancy)
- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4 paper v1](https://arxiv.org/abs/2603.05451v1)

## Related

- [Tensor Memory](../hardware/tmem.md) — allocation, addressing, and access constraints
- [tcgen05 MMA](../hardware/tcgen05-mma.md) — exact operation and completion semantics
- [register pressure](../patterns/register-pressure.md) — diagnosis and compiler-resource evidence
