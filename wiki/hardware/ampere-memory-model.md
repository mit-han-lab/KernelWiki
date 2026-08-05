---
id: hw-ampere-memory-model
title: "Ampere Memory Model: SMEM/L2/Occupancy on sm_80 vs sm_86"
type: hardware
architectures: [sm80, sm86]
tags: [cp-async, l2-persistence, cache-policy, shared-memory-optimization, mbarrier]
confidence: source-reported
reproducibility: snippet
related: [hw-cp-async, hw-mma-sync-ampere, technique-tile-scheduling, technique-cache-policy, technique-swizzling, migration-hopper-to-ampere]
sources: [doc-ampere-tuning-guide, doc-ga102-whitepaper, doc-ptx-isa-ampere]
aliases: ["L2 persistence", cudaAccessPolicyWindow, "shared memory carveout", "GA102 memory"]
blackwell_relevance: "Capacity planning is where Hopper/Blackwell kernel configs break first on Ampere: 99 KB SMEM/block (vs 227 KB) and single-digit-MB L2 (vs 50–126 MB) invalidate tile shapes and scheduling assumptions before any instruction is translated."
---

# Ampere Memory Model: SMEM/L2/Occupancy on sm_80 vs sm_86

## Overview

Two different "Amperes" exist. GA100 (A100, sm_80) is the datacenter chip most papers mean; GA10x (RTX 3090/3080, RTX A6000, A40 — sm_86) is what consumer rigs run. They differ exactly where kernel configuration lives: SMEM capacity, occupancy ceilings, L2 size, and DRAM technology. This page is the capacity card for both, plus the L2-persistence and async-barrier features shared across Ampere.

## Capacity card

| Resource | A100 (sm_80) | RTX 3090 (sm_86) | H100 (sm_90, for contrast) |
|---|---|---|---|
| SMs | 108 | 82 | 132 |
| Max threads / SM | 2048 (64 warps) | **1536 (48 warps)** | 2048 |
| Max blocks / SM | 32 | **16** | 32 |
| Registers | 64K × 32-bit / SM, 255/thread | same | same |
| Unified L1/SMEM | 192 KB | **128 KB** | 256 KB |
| Max SMEM carveout / SM | 164 KB | **100 KB** | 228 KB |
| Max SMEM / block (opt-in) | 163 KB | **99 KB** | 227 KB |
| L2 | 40 MB | **6 MB** | 50 MB |
| DRAM | HBM2e, ~1.6–2.0 TB/s | GDDR6X, 936 GB/s | HBM3, 3.35 TB/s |
| BF16 tensor peak (f32 acc) | 312 TFLOPS | **71 TFLOPS** | 989 TFLOPS |

Deriving the two headline constraints for sm_86 backports:

1. **SMEM budget: 99 KB per block.** A Hopper kernel using 200+ KB (FA3-style Q/K/V staging, 4–6 pipeline stages) must shrink stage count, tile size, or both. Practical Ampere configs: 2–4 stages, ≤48 KB static + dynamic opt-in via `cudaFuncAttributeMaxDynamicSharedMemorySize`.
2. **Occupancy: 1536 threads/SM.** Three 512-thread blocks or six 256-thread blocks fill an SM. A 2×1024 config (fine on A100) strands 25% of the sm_86 thread budget.

## Roofline position: why 3090 optimization priorities differ

FLOP/byte knee = tensor peak ÷ DRAM bandwidth:

- RTX 3090 (bf16, f32 acc): 71e12 / 936e9 ≈ **76 FLOP/B**
- A100 (bf16): 312e12 / 2.0e12 ≈ 156 FLOP/B
- H100 (bf16): 989e12 / 3.35e12 ≈ 295 FLOP/B

The 3090 is *bandwidth-rich relative to its tensor throughput*: kernels cross into compute-bound territory at ~4× lower arithmetic intensity than on H100. Consequences:

- GEMM-class kernels saturate tensor cores earlier — operand-delivery tricks (deep pipelining, giant tiles) pay off less than on Hopper; hitting the 71/142 TFLOPS ceiling is the game.
- Memory-bound kernels (decode GEMV, elementwise, KV-cache scans) sit closer to parity with datacenter cards than the TFLOPS gap suggests — this is why weight-only-quantized decode on GA10x is competitive.
- The small 6 MB L2 makes **tile scheduling for L2 reuse** (technique-tile-scheduling) *more* important than on A100/H100, not less: a 128×128 bf16 output tile's operand footprint already competes for the whole cache.

## L2 persistence window (sm_80+)

Pin a hot region (router weights, decode KV heads, quant scales) into a set-aside L2 slice:

```cpp
// host side — compilable, CUDA 11+
#include <cuda_runtime.h>
void pin_l2(cudaStream_t stream, void* ptr, size_t bytes) {
    cudaDeviceProp prop;                     cudaGetDeviceProperties(&prop, 0);
    size_t setaside = prop.persistingL2CacheMaxSize;              // e.g. up to 75% of L2
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setaside);
    cudaStreamAttrValue v{};
    v.accessPolicyWindow.base_ptr  = ptr;
    v.accessPolicyWindow.num_bytes = bytes;                       // ≤ accessPolicyMaxWindowSize
    v.accessPolicyWindow.hitRatio  = 1.0f;
    v.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &v);
}
```

On a 6 MB-L2 part this is a scarce resource — reserve it for operands re-read by *many* CTAs, and mark streaming tensors `evict_first` (PTX cache hints, technique-cache-policy) so they don't fight the window.

## Async barriers (sm_80+)

`cuda::barrier<cuda::thread_scope_block>` (PTX `mbarrier`) enables arrive/wait split-phase sync — producer warps arrive after issuing `cp.async`, consumer warps wait without a full `__syncthreads()`. Ampere supports init/arrive/test_wait and `cp.async.mbarrier.arrive`; it does NOT support sm_90's `expect_tx` byte-count tracking (count arrivals instead — see hw-cp-async and migration-hopper-to-ampere).

## SMEM banking (unchanged since Volta, still decisive)

32 banks × 4 B; `ldmatrix` issues 32 row-reads per instruction, so unswizzled row-major tiles with power-of-two leading dimensions conflict catastrophically. Standard fix: XOR-swizzle the store index of `cp.async` destinations (technique-swizzling). GA10x and GA100 behave identically here — swizzle math ports from Hopper kernels unchanged, minus the TMA hardware swizzle modes (do it in index arithmetic).

## GDDR6X behavior notes (sm_86)

- 936 GB/s is the pin-rate ceiling; sustained achievable in well-coalesced kernels is ≈85–90% of it, and random-access patterns degrade harder than on HBM (narrower effective row buffer locality).
- Latency is higher than HBM — deeper cp.async pipelines (or higher occupancy) are needed to cover it, in tension with the smaller SMEM. This is THE central tuning trade-off on GA10x: stages × tile SMEM ≤ 99 KB while still covering GDDR6X latency.
