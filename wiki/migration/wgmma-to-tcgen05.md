---
id: migration-wgmma-to-tcgen05
title: "Migrating from wgmma to tcgen05"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tcgen05, wgmma, tmem]
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, blog-tcgen05-tutorial]
blackwell_relevance: "WGMMA uses warpgroup issue and register accumulators; tcgen05 uses single-thread MMA issue and TMEM accumulators."
confidence: verified
reproducibility: snippet
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
---

# Migrating from wgmma to tcgen05

## What actually changes

Port the programming model, not just the opcode:

| Concern | SM90 WGMMA | SM100 tcgen05.mma |
|---|---|---|
| MMA issue | Warpgroup collective | One thread for group-1 or group-2 MMA |
| D accumulator | Per-thread registers | TMEM |
| A operand | SMEM-descriptor or register forms, depending on instruction | SMEM-descriptor or TMEM-address forms |
| B operand | SMEM descriptor | SMEM descriptor |
| MMA completion | Commit/wait groups | `tcgen05.commit` plus mbarrier wait |
| Cross-thread ordering | WGMMA-specific rules | tcgen05 fences composed with an execution-ordering operation |
| Narrow floating formats | FP8 WGMMA forms | FP8/FP6/FP4 plus block-scaled MX/NVFP4 kinds |

The exact old WGMMA form matters. Do not assume every Hopper kernel loads A through `ldmatrix`: descriptor-sourced WGMMA forms already read A from SMEM. Likewise, tcgen05 can source A from SMEM or TMEM.

## Dependency-ordered migration checklist

1. Identify the exact WGMMA operand form, accumulator type, shape, and group-completion points in the SM90 kernel.
2. Choose a legal tcgen05 kind, A source, CTA group, instruction descriptor, and data-path layout for the SM100 target.
3. Replace register-resident D with a TMEM allocation. Group-1 allocation/deallocation is warp-collective; group 2 requires one warp in each CTA of the pair.
4. Keep A/B backing storage unchanged until all asynchronous MMA consumers have completed.
5. Replace WGMMA group completion with `tcgen05.commit` and an mbarrier wait. Add tcgen05 fences only where an execution-ordering handoff must order tcgen05-visible state across threads or CTAs.
6. Transfer completed accumulator values from TMEM to registers with `tcgen05.ld`, observe its completion/access rules, then run the epilogue.
7. Deallocate every dynamic TMEM allocation before kernel exit. Use the same `cta_group` value for all tcgen05 instructions in the kernel.
8. Retune CTA/cluster shapes, pipeline stages, thread roles, descriptors, and epilogue scheduling on the target workload.

## TMEM lifecycle

`tcgen05.alloc` writes a 32-bit TMEM address to shared memory. Allocation size is expressed in columns, in power-of-two multiples permitted by PTX. It is a synchronous warp instruction: a lane-0-only call is invalid. `tcgen05.dealloc` has the same issue granularity, and PTX requires all allocated TMEM to be deallocated before kernel exit—not only in persistent kernels.

The MMA issuer and the allocation participants are different concepts. One thread initiates a group-1 MMA, but one full warp collectively allocates or deallocates its TMEM. A group-2 allocation uses one warp from each peer CTA.

## Completion, ordering, and epilogue

`tcgen05.mma` is asynchronous. `tcgen05.commit.cta_group::N.mbarrier::arrive::one.b64` makes an mbarrier track prior MMA work issued by the current thread; waiting on that barrier observes completion. A `tcgen05.fence` is an ordering and code-motion primitive, not a completion wait, so fence plus `__syncthreads()` is not a substitute for commit/mbarrier.

After completion and any necessary thread handoff, epilogue warps use `tcgen05.ld` to bring their accessible TMEM lanes into registers. Ordinary bias, activation, conversion, and global-store code then operates on those register values. Preserve the `tcgen05.ld` warp-access mapping and use `tcgen05.wait::ld` where its asynchronous completion must be observed.

## Layout and shape selection

Do not mechanically convert every 64-byte swizzle to 128-byte swizzling. The tcgen05 shared-memory descriptor defines valid no-swizzle, 128B, 64B, and 32B modes, subject to mode-specific alignment and layout constraints. The 128B choice can be much faster for a particular tile, but it is not a universal correctness condition.

Similarly, there is no universal Hopper-to-Blackwell rule that doubles M. WGMMA and tcgen05 each expose multiple shapes; tcgen05 M/N are encoded in `idesc` and constrained by kind, layout, CTA group, and target ISA. Treat m128xn256xk16 and m256xn256xk16 as useful F16/BF16 maximum examples, not a complete migration table.

## Warp specialization

Single-thread MMA issue can free instruction-issue capacity for TMA, descriptor preparation, epilogue, reductions, or scheduling. It does not determine the kernel's total thread count: TMEM allocation, TMEM loads, TMA, epilogue, and barriers remain warp- or CTA-level work. Start from a pinned SM100 implementation and measure a role partition rather than copying fixed warp counts from an SM90 kernel.

Moving D out of the GPR file also changes the register budget. Revisit tile size, pipeline depth, and launch bounds, but do not assume that lower accumulator pressure automatically increases occupancy; SMEM, TMEM, barriers, threads, and registers used by other roles can become limiting resources.

## CUTLASS migration

CUTLASS does not reduce this port to replacing an architecture tag and one schedule token. In pinned SM100 examples, the collective builders are re-instantiated with SM100 operator classes, tile and cluster shapes, stage policy, mainloop schedule, and epilogue schedule; the resulting collectives feed `cutlass::gemm::kernel::GemmUniversal` and a device adapter. Schedule tags are configuration-specific—for example, the pinned tree defines `KernelTmaWarpSpecialized1SmSm100` and `KernelTmaWarpSpecialized2SmSm100`, not `KernelScheduleSm100CpAsyncWarpSpecialized`.

Use a complete pinned example such as CUTLASS `examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu`, then revalidate alignment, workspace, hardware support, dispatch, and numerical results for the migrated configuration.
