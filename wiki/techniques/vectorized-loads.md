---
id: technique-vectorized-loads
title: "Vectorized Loads and Cache Hints"
type: technique
architectures: [sm100, sm90]
tags: [vectorized-loads, cache-policy, register-budgeting]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
reproducibility: concept
prerequisites: []
related: [technique-register-budgeting, kernel-nvfp4-gemv, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, doc-cuda-register-controls, blog-yue-nvfp4, blog-amandeep-nvfp4, contest-gpumode-p1]
blackwell_relevance: "PTX ISA 9.0 adds SM100 support for 256-bit global vector loads through v8.b32 or v4.b64 forms; their alignment, cache-hint, register, and workload tradeoffs remain explicit verification obligations."
---

# Vectorized Loads and Cache Hints

## What a wider load establishes

A PTX vector load moves several typed elements into several registers with one instruction. This can reduce the number of load instructions issued by a thread, but PTX only says that vector loads *may* improve memory performance. Source-level width alone does not prove fewer physical memory transactions, higher achieved bandwidth, or lower kernel latency.

PTX ISA 9.0 permits the following global-memory forms on `sm_100`. The first accesses 16 bytes and the second accesses 32 bytes:

```ptx
.reg .b64 addr;
.reg .u32 x<4>;
.reg .u64 y<4>;

ld.global.v4.u32 {x0, x1, x2, x3}, [addr];
ld.global.v4.u64 {y0, y1, y2, y3}, [addr];
```

The address must be naturally aligned to the total access size: 16 bytes for `v4.u32` and 32 bytes for `v4.u64`. A misaligned PTX address has undefined behavior; the ISA says it may have low address bits masked or fault. It does not specify a fallback to narrower loads. Guard tails or dispatch them to an access whose complete range and alignment are valid.

The 256-bit `v4.b64` family is an SM100-or-newer feature. A 32-byte load contains 64 values when the payload is densely packed E2M1 at two 4-bit values per byte. That arithmetic says nothing about whether the width is profitable: unpack cost, live registers, instruction selection, per-thread address patterns, and tail handling can outweigh a lower source-level load count.

## Cache controls are hints

PTX distinguishes cache operators, eviction priorities, prefetch-size hints, and non-coherent read-only loads. Representative legal forms are:

```ptx
ld.global.L1::no_allocate.v4.u32 {x0, x1, x2, x3}, [addr];
ld.global.L1::evict_last.v4.u32  {x0, x1, x2, x3}, [addr];
ld.global.L2::256B.b32 x0, [addr];
ld.global.nc.b32 x0, [addr];
```

| Form | Documented meaning | Limit on inference |
|---|---|---|
| `L1::no_allocate` | L1 eviction-priority selection that may be applied | Does not guarantee an L1 bypass or a speedup for a streaming operand |
| `L1::evict_last` | Requests the corresponding L1 eviction priority | Does not guarantee that a line remains resident |
| `L2::256B` | Hints that additional data of the stated size be prefetched into L2 | Is not a 256-byte load or a promotion guarantee |
| `ld.global.nc` | Loads through a non-coherent read-only cache | Has architecture- and parallelism-dependent latency/throughput; it is not a general coherence optimization |

Cache-policy operands and prefetch-size qualifiers are performance hints and do not change the program's memory-consistency behavior. A reuse argument can motivate `evict_last`, and a one-pass stream can motivate `no_allocate`, but only a matched comparison determines whether either helps the concrete working set and launch.

## Reproducible selection procedure

For each candidate width and cache policy:

1. Prove base and per-thread address alignment for every executed access. Use a separate scalar or narrower-vector path for tails; validate minimum, maximum, and awkward sizes.
2. Hold the algorithm, mapping, decode, unrolling, launch shape, compiler, and inputs fixed while changing one load or cache choice. Compare identical correctness oracles before timing.
3. Inspect generated PTX and SASS. Record actual load instructions, registers per thread, spill loads/stores, stack bytes, shared memory, and launch parameters. A source cast or inline-PTX block does not bypass compiler register allocation.
4. Profile the matched variants. Check achieved bandwidth, requested versus transferred bytes, cache hit behavior, instruction issue/stalls, and active warps with metrics available on the installed Nsight Compute and GPU versions.
5. Use warmups, synchronization, repeated trials, and the same statistic for every production shape. Keep the change only where the declared metric improves without a correctness or resource regression.

Register caps and launch bounds are a separate variable. A nominal cap may be inert when natural allocation is lower, or it may trade registers for spills and extra instructions. Follow the resource and occupancy workflow in [Register Budgeting](register-budgeting.md) rather than inferring residency from the cap value.

## NVFP4 GEMV case study

Yue Zhang reports the following CUDA progression for GPU Mode Problem 1. These are author-reported endpoints without raw repeated-trial data or released complete submission code:

| Stage | Combined change | Reported latency |
|---|---|---:|
| Initial CUDA | Naive hand-written path | about 2000 µs |
| CUDA optimization 1 | Coalescing, shared B, thread collaboration, warp reduction | about 443 µs |
| CUDA optimization 2 | Remove shared B, per-thread tiles, `float4` loads, hardware intrinsics | about 39 µs |
| CUDA optimization 3 | Vectorized PTX FP4 and scale decode | about 27 µs |
| Parameter tuning | Threads per row and rows per block | about 26 µs |
| ILP | Two tiles per loop iteration | about 22.9 µs |
| Aggressive PTX fusion | Decode, scales, multiply, and accumulation in a larger PTX block | about 22.3 µs |

The submitted public-leaderboard score was 22.392 microseconds, the geometric mean over three benchmark shapes. The pinned task's theoretical model gives 8.622, 17.275, and 4.317 microseconds for those separate shapes; 8.622 microseconds is therefore not an aggregate “speed of light.” The 39-to-27 endpoints change the PTX load/decode path together and do not isolate cache policy, load width, or conversion choice.

Amandeep Singh's attempts supply useful negative controls for the same task. Replacing two `uchar4` loads with one `uint2` load was reported 16–25% slower because extraction added instructions. Lowering `-maxrregcount` from 80 to 64 had no effect because natural allocation was already below the cap. The author later observed wider PTX forms and differentiated L1 policies in other solutions, but published no contestant code or controlled ablation for those observations.

Together, the reports justify testing vector width, decode organization, cache hints, and register limits. They do not establish that the widest load, a particular cache hint, or the lowest register cap is essential for GEMV, sub-byte arithmetic, or decode workloads.
