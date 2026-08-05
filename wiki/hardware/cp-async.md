---
id: hw-cp-async
title: "cp.async — Asynchronous Global→Shared Copy (Ampere)"
type: hardware
architectures: [sm80, sm86]
tags: [cp-async, mbarrier, pipeline-stages, double-buffering]
confidence: source-reported
reproducibility: snippet
related: [hw-tma, hw-mbarrier, technique-pipeline-stages, technique-double-buffering, migration-hopper-to-ampere, hw-ampere-memory-model]
sources: [doc-ptx-isa-ampere, doc-ampere-tuning-guide, doc-cutlass-ampere]
aliases: [cp.async, LDGSTS, "async copy", "cuda::memcpy_async"]
blackwell_relevance: "cp.async remains available on sm_90/sm_100 for small non-bulk operands, but on Ampere it is the ONLY asynchronous load path — every TMA-based Hopper/Blackwell design backported to sm_8x lands on cp.async."
---

# cp.async — Asynchronous Global→Shared Copy (Ampere)

## Overview

`cp.async` (SASS: LDGSTS) is Ampere's asynchronous copy instruction: data moves from global memory directly into shared memory **without staging through the register file**, and the issuing warp continues executing. It is the mechanism behind every multi-stage software pipeline on sm_80/sm_86 — the CUTLASS `MmaMultistage` mainloop, FlashAttention-2's K/V staging, and Triton's `num_stages > 2` pipelining all lower to it.

It is also the semantic ancestor of Hopper's TMA. When backporting TMA-based kernels, `cp.async` is what TMA becomes — with the address generation and tile bookkeeping moving back into the kernel code.

## Key Properties

| Property | Detail |
|---|---|
| Direction | GMEM → SMEM only (no store direction, unlike TMA) |
| Granularity | 4 / 8 / 16 bytes **per thread** (`.ca`); 16 bytes only (`.cg`) |
| Addressing | Each thread computes its own src/dst addresses (vs TMA descriptor) |
| Cache behavior | `.ca` allocates L1+L2; `.cg` bypasses L1 (use for operand tiles) |
| OOB handling | Optional `src-size` operand zero-fills the tail (no branches needed) |
| Completion | Commit-group counting (`commit_group`/`wait_group N`) or mbarrier arrive |
| Issue cost | One instruction per 16 B per thread — a 128x64 bf16 tile = 128 issues/warp vs 1 TMA descriptor op |
| Register cost | No data registers consumed; address registers still needed |

## Programming Surfaces (pick one level)

**PTX (full control):**

```cuda
__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* gmem_src) {
    unsigned dst = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit()      { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group()  { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
```

**CUDA intrinsics (`<cuda_pipeline.h>`)** — `__pipeline_memcpy_async(dst, src, 16)`, `__pipeline_commit()`, `__pipeline_wait_prior(N)` map 1:1 to the PTX above.

**libcu++** — `cuda::memcpy_async(group, dst, src, size, pipe)` with `cuda::pipeline`; adds mbarrier-backed completion and cleaner multi-stage structure at slight codegen-trust cost.

**CuTe atoms** — `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>` / `SM80_CP_ASYNC_CACHEALWAYS<T>` + `cp_async_fence()` / `cp_async_wait<N>()`.

## Canonical Multi-Stage Pipeline (compilable, sm_80/sm_86)

The universal Ampere mainloop shape — `STAGES` SMEM buffers, prologue fills `STAGES-1`, steady state overlaps one load with one compute:

```cuda
// nvcc -arch=sm_86 -c cp_async_pipeline.cu
#include <cuda_pipeline.h>
#include <cstdint>

constexpr int STAGES   = 3;
constexpr int TILE     = 4096;              // bytes per stage per block
constexpr int THREADS  = 128;
constexpr int PER_THR  = TILE / THREADS / 16;  // 16B copies per thread per stage

extern __shared__ uint8_t smem[];           // STAGES * TILE bytes

// Placeholder compute — a real kernel runs its ldmatrix/mma.sync schedule here
__device__ void consume_tile(const uint8_t* stage_base, float* acc) {
    *acc += (float)stage_base[threadIdx.x];
}

__global__ void pipelined_kernel(const uint8_t* __restrict__ g_in, int num_tiles, float* out) {
    float acc = 0.f;
    auto stage_ptr = [&](int s) { return smem + s * TILE; };
    auto issue_stage = [&](int s, int tile_idx) {
        const uint8_t* src = g_in + (size_t)tile_idx * TILE + threadIdx.x * 16;
        uint8_t*       dst = stage_ptr(s) + threadIdx.x * 16;
        #pragma unroll
        for (int i = 0; i < PER_THR; i++)
            __pipeline_memcpy_async(dst + i * THREADS * 16, src + i * THREADS * 16, 16);
        __pipeline_commit();
    };

    // Prologue: fill STAGES-1 buffers
    for (int s = 0; s < STAGES - 1 && s < num_tiles; s++)
        issue_stage(s, s);

    for (int t = 0; t < num_tiles; t++) {
        // Wait until the oldest in-flight group (tile t) has landed:
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();                        // make stage visible to all warps
        consume_tile(stage_ptr(t % STAGES), &acc);
        __syncthreads();                        // stage fully consumed, safe to overwrite
        int next = t + STAGES - 1;
        if (next < num_tiles)
            issue_stage(next % STAGES, next);   // refill the buffer just freed
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}
```

Notes on the pattern:

- `__pipeline_wait_prior(STAGES - 2)` = "at most STAGES-2 groups still in flight" — the oldest stage is complete, the rest keep loading.
- Both `__syncthreads()` are load-bearing: cp.async completion is only observed by the *issuing* thread's wait; other warps need the barrier (or an mbarrier scheme) before reading the stage.
- 16-byte alignment of both addresses is mandatory for the 16 B variant; misalignment is an illegal-instruction trap, not a slowdown.

## Zero-Fill OOB Idiom

For boundary tiles, instead of predicating loads, issue the full 16 B copy with a clamped `src-size` — the hardware zero-fills the remainder:

```cuda
__device__ __forceinline__ void cp_async_cg_zfill(void* smem_dst, const void* gmem_src, int valid_bytes) {
    unsigned dst = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(gmem_src), "r"(valid_bytes));
}
```

This replaces the `zfill` role of TMA's built-in OOB clamping in backports (TMA clamps per tensor descriptor; here you clamp per 16 B transaction).

## Performance Rules

1. **Use `.cg` (16 B) for operand tiles.** `.ca` pollutes L1 that the compute path wants for other data. CUTLASS uses `.cg` for A/B tiles.
2. **3–5 stages** is the practical range. More stages hide more latency but eat SMEM (99 KB/block ceiling on sm_86 — see hw-ampere-memory-model) and raise `__syncthreads()` pressure.
3. **Issue density matters.** Unlike TMA (one thread issues a whole tile), cp.async spends warp issue slots — interleave issues with compute rather than bursting all copies back-to-back when the mainloop is issue-bound.
4. **Swizzle SMEM destinations** (XOR patterns) so subsequent `ldmatrix` reads are bank-conflict-free; cp.async has no hardware swizzle-on-store (TMA does).
5. **Don't mix commit-group and mbarrier tracking** for the same transfers; pick one completion scheme per pipeline.

## vs TMA (backport view)

| Concern | TMA (sm_90+) | cp.async (sm_8x) |
|---|---|---|
| Who computes addresses | Hardware, from descriptor | Every thread, in registers |
| Issue cost | 1 op / tile | 1 op / 16 B / thread |
| OOB | Descriptor clamp | Per-copy `src-size` zfill |
| Swizzle | On the fly (32/64/128 B) | Do it yourself in the dst index math |
| Multicast to CTAs | Yes (cluster) | No — each CTA loads its own copy (L2 absorbs the overlap) |
| SMEM→GMEM direction | Yes (`cp.async.bulk` store) | No — plain `st.global` epilogue |
| Completion | mbarrier expect_tx (bytes) | Commit-group count or mbarrier arrivals |

The deltas in this table are exactly the work items of a Hopper→Ampere port; see migration-hopper-to-ampere.
