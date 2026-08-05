---
id: hw-mma-sync-ampere
title: "mma.sync + ldmatrix — Ampere Tensor Core Programming"
type: hardware
architectures: [sm80, sm86]
tags: [mma-sync, ldmatrix, register-budgeting, swizzling]
confidence: source-reported
reproducibility: snippet
related: [hw-cp-async, technique-register-budgeting, technique-swizzling, migration-hopper-to-ampere, hw-tcgen05-mma, hw-ampere-memory-model]
sources: [doc-ptx-isa-ampere, doc-ga102-whitepaper, doc-cutlass-ampere]
aliases: [mma.sync, HMMA, m16n8k16, "warp MMA", ldmatrix]
blackwell_relevance: "mma.sync is what wgmma (sm_90) and tcgen05.mma (sm_100) replaced. Backporting any Hopper/Blackwell tensor-core kernel to Ampere means re-expressing its MMAs in this warp-scope synchronous form with register-resident accumulators."
---

# mma.sync + ldmatrix — Ampere Tensor Core Programming

## Overview

On sm_80/sm_86 the tensor core is driven by **`mma.sync`** — a *synchronous, warp-scope* instruction: 32 threads collectively hold the operand fragments in registers, issue the op, and stall until the result is back in their registers. There is no asynchronous tensor-core path on Ampere. Everything Hopper/Blackwell later added — warpgroup-scope async (`wgmma`), SMEM-sourced operands, TMEM accumulators, single-thread issue — exists precisely to remove the costs this page describes. Backports must add those costs back in.

Operand fragments are loaded from shared memory with **`ldmatrix`**, which materializes the exact per-lane register layout `mma.sync` expects.

## The workhorse shapes

| Inputs | Acc | Shape (M×N×K) | Per-warp per-instr FLOPs |
|---|---|---|---|
| f16/bf16 | f32 | 16×8×16 | 4096 |
| f16 | f16 | 16×8×16 | 4096 |
| tf32 | f32 | 16×8×8 | 2048 |
| s8 | s32 | 16×8×32 | 8192 (OPs) |

A warp-level GEMM tile (e.g., 64×64) is built by tiling these primitives: 64×64 from m16n8k16 = 4×8 = 32 `mma.sync` issues per K-step of 16.

## Register economics (the defining constraint)

All three operand classes live in registers, fragmented across 32 lanes:

- **Accumulator**: an M×N f32 warp tile costs `M*N/32` registers per thread. A 64×64 tile = **128 registers/thread** — half the 255 budget before A/B fragments, addresses, and indices.
- **A fragment** (m16n8k16, f16): 4× b32 regs/thread per 16×16 tile; **B**: 2× b32 per 16×8.
- Consequences: Ampere warp tiles top out around 64×64; register spills show up as `LDL`/`STL` in SASS and destroy throughput; `-maxrregcount`/`__launch_bounds__` tuning is a first-class knob (see technique-register-budgeting).

This is the direct contrast with sm_90 (`wgmma` reads A/B from SMEM, accumulator still registers but per-warpgroup) and sm_100 (`tcgen05` moves accumulators to TMEM entirely).

## GA10x quirk: FP32 accumulation runs at half rate

On sm_86 GeForce/workstation silicon (RTX 3090, RTX A6000, A40), FP16-input MMA with **FP32 accumulate runs at half the rate** of FP16 accumulate (71 vs 142 dense TFLOPS on RTX 3090). BF16 always requires an f32 accumulator, so BF16 is always on the slow path. A100 (sm_80) does not have this segmentation — both run at full rate.

| Consequence | Practical guidance |
|---|---|
| BF16 GEMM ceiling on 3090 = 71 TFLOPS | Don't chase A100-style BF16 efficiency numbers on GA10x |
| f16×f16→f16 acc is the only full-rate FP path | Usable for attention P·V and short-K accumulations with scaling discipline; risky for deep-K GEMM |
| INT8→s32 runs at 284 TOPS (4× the f32-acc path) | Weight-only INT4/INT8 (Marlin-style) and W8A8 kernels are disproportionately rewarded on GA10x |
| TF32 = 35.6 TFLOPS | Still ~2× FP32 CUDA cores; 3xTF32 emulation (CUTLASS ex. 27) buys near-FP32 accuracy |

## ldmatrix (and the missing stmatrix)

```asm
ldmatrix.sync.aligned.m8n8.x4.shared.b16       {d0,d1,d2,d3}, [addr];   // 4 tiles of 8x8xb16
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {d0,d1},       [addr];   // transposed (col-major B)
```

- Each lane passes a row address; the instruction shuffles data into the canonical fragment layout. Pair with an XOR-swizzled SMEM layout to make all 32 row reads bank-conflict-free (see technique-swizzling).
- `.trans` handles the B-operand transpose at load time — no separate transpose pass.
- **`stmatrix` does not exist on Ampere** (sm_90+). Epilogues write accumulators out with plain `st.shared`/`st.global.v4`; getting coalesced stores from the mma fragment layout requires a register→SMEM→register shuffle roundtrip (CUTLASS `Epilogue` does exactly this) — budget SMEM and cycles for it in backports.

## Minimal compilable warp-tile fragment (sm_80/sm_86)

One 16×16 A tile × 16×8 B tile accumulating a 16×8 f32 result — the primitive every Ampere GEMM mainloop repeats:

```cuda
// nvcc -arch=sm_86 -c mma_fragment.cu
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void* smem) {
    unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16_f32(float (&d)[4], const uint32_t (&a)[4],
                                                 const uint32_t (&b)[2], const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Per-lane addresses for the canonical fragment layouts:
__global__ void warp_tile_mma(const half* __restrict__ /*unused*/, float* out) {
    __shared__ __align__(16) uint8_t sA[16 * 16 * 2];   // 16x16 f16, row-major
    __shared__ __align__(16) uint8_t sB[16 * 8  * 2];   // 16x8  f16, col-major view
    int lane = threadIdx.x & 31;

    uint32_t a[4], b[2];
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // A: lanes 0..15 address rows 0..15 of the left 16x8 half, lanes 16..31 the right half
    ldmatrix_x4(a, sA + ((lane & 15) * 16 + (lane >> 4) * 8) * 2);
    // B: lanes 0..15 address the 16 rows of the 16x8 tile (transposed load)
    ldmatrix_x2_trans(b, sB + (lane & 15) * 8 * 2);
    mma_m16n8k16_f32(acc, a, b, acc);

    // D fragment layout (PTX ISA): lane = 4*row_group + col_pair; acc[0..1] cover
    // rows 0..7, acc[2..3] rows 8..15 of the 16x8 tile. Real epilogues remap via SMEM.
    out[lane * 4 + 0] = acc[0]; out[lane * 4 + 1] = acc[1];
    out[lane * 4 + 2] = acc[2]; out[lane * 4 + 3] = acc[3];
}
```

(Address math for `ldmatrix` source rows follows the PTX ISA fragment tables; real kernels wrap it in a swizzle function.)

## vs wgmma / tcgen05 (what a backport gives up)

| Property | tcgen05 (sm_100) | wgmma (sm_90) | mma.sync (sm_8x) |
|---|---|---|---|
| Scope | 1 thread issues | warpgroup (128 thr) | warp (32 thr) |
| Async | Yes (mbarrier) | Yes (commit/wait) | **No — warp stalls** |
| A/B source | SMEM/TMEM | SMEM (desc) | **Registers (via ldmatrix)** |
| Accumulator | TMEM | Registers | **Registers** |
| Max shape | 128×512×k | 64×256×k | 16×8×16 |
| Latency hiding | Hardware | Warpgroup switch | **Occupancy + ILP only** |

Because the warp blocks on every `mma.sync`, Ampere kernels hide tensor-core latency with *other warps* (occupancy) and instruction-level interleaving of ldmatrix/cp.async/mma — not with intra-warp async. This is why warp specialization pays off far less on Ampere than on Hopper (see migration-hopper-to-ampere).
