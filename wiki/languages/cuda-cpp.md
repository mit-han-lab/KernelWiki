---
id: lang-cuda-cpp
title: "CUDA C++ for Blackwell Kernels"
type: language
tags: [cuda-cpp, ptx, tcgen05, tmem]
related: [lang-ptx, hw-tcgen05-mma, hw-tmem, blog-tcgen05-tutorial]
sources: [blog-tcgen05-tutorial, doc-nvidia-tuning-guide, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
---

## Scope

CUDA C++ can host PTX instructions that do not yet have a convenient CUDA intrinsic. Gau Nernst's pinned `tcgen05` tutorial uses that approach for a B200 GEMM. For M=N=K=4096 in its disclosed PyTorch 2.9.1/CUDA 13 environment, v6 reports 1475.93 TFLOP/s versus 1506.74 TFLOP/s for cuBLAS, or 97.96%. This is an author-reported result for that configuration, not a portable performance guarantee.

## Inline-PTX boundary

The CUDA front end does not parse the instruction text inside an `asm()` statement. Operand constraints and address-space conversion therefore remain the wrapper author's responsibility. Use `"r"` for a 32-bit integer register, `"l"` for a 64-bit integer register, and convert a generic pointer with `__cvta_generic_to_shared` before supplying a shared-memory address. Add a `"memory"` clobber when the assembly has memory effects that are hidden from the compiler.

The allocation instruction is collective. For `.cta_group::1`, every lane of one designated warp must execute the same instruction. Synchronize the CTA before another warp reads the address written to shared memory.

```cuda
// All 32 lanes of alloc_warp execute this branch.
if (warp_id == alloc_warp) {
    uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_tmem_addr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
        "[%0], %1;"
        :: "r"(smem_addr), "r"(num_cols) : "memory");
}
__syncthreads();
uint32_t taddr = *smem_tmem_addr;
```

An unscaled `kind::f16` MMA takes one instruction descriptor and an `enable-input-d` predicate. A single elected thread may issue this MMA form; converting an ordinary CUDA integer to the required PTX predicate inside the assembly keeps the C++ interface well typed.

```cuda
__device__ inline void tcgen05_mma_f16(
    uint32_t taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(idesc), "r"(enable_input_d));
}
```

`tcgen05.ld` is likewise a warp-level instruction. All participating lanes execute it, then the warp executes `tcgen05.wait::ld.sync.aligned` before consuming the loaded registers. After all readers finish, synchronize at the required CTA or cluster scope and have all lanes of one warp execute the matching collective `tcgen05.dealloc` before kernel exit.

## Barrier lifecycle

An `mbarrier.arrive.expect_tx` plus a parity wait is not a complete pipeline by itself. A staged TMA-to-MMA pipeline has these invariants:

1. An elected thread initializes each barrier with the intended arrival count, then publishes initialization to the async proxy with the appropriate `fence.mbarrier_init`.
2. The producer reserves a reusable stage, issues TMA against that stage's full barrier, and accounts for the expected transaction bytes.
3. The consumer waits with acquire semantics on the matching phase before reading the shared-memory stage.
4. After issuing tcgen05 MMA, it commits completion to a separate barrier; the epilogue waits before loading TMEM.
5. The last stage user signals a separate empty/reuse barrier. A stage cannot be overwritten until its owner observes that handoff.
6. Each barrier's parity flips only when its circular stage is revisited. Inline assembly that reads or writes memory invisibly to C++ uses a `"memory"` clobber.

Arrival counts, transaction bytes, scope, and which CTA owns each barrier depend on whether the kernel uses one-CTA or two-CTA MMA. Copy the complete protocol from the pinned implementation rather than treating isolated wait/arrive wrappers as a standalone recipe.

## One verified role split

The tutorial's v6 kernel launches six warps per CTA: warp 0 elects one lane for TMA, warp 1 collectively allocates TMEM and elects one lane for MMA, and warps 2--5 run the epilogue. Before exit, all threads finish their TMEM reads and warp 0 collectively deallocates TMEM. This is the role split of that implementation, not a requirement of CUDA C++ or tcgen05.

## Primary references

- [CUDA 13.0.2 Inline PTX Assembly guide](https://docs.nvidia.com/cuda/archive/13.0.2/inline-ptx-assembly/index.html)
- [PTX ISA 9.0 tcgen05 instructions](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05)
- [Pinned tutorial implementation](https://github.com/gau-nernst/learn-cuda/tree/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100)

## Related

- [ptx-sm100](ptx-sm100.md) — version-pinned PTX instruction forms
- [tcgen05 tutorial](../../sources/blogs/tcgen05-tutorial.md) — benchmark provenance and progression
