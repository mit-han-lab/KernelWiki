---
id: technique-vectorized-loads
title: "Wide Vectorized Loads and Cache Policies"
type: technique
architectures: [sm100, sm90]
tags: [vectorized-loads, cache-policy, register-budgeting]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-nvfp4-gemv, pattern-memory-bound]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, contest-gpumode-p1, doc-ptx-isa-sm100]
blackwell_relevance: "Aligned vector accesses and PTX cache hints can reduce instruction pressure or cache pollution on SM100, subject to workload-specific reuse and resource tradeoffs."
---

# Wide Vectorized Loads and Cache Policies

## What vectorization changes

A vector load names multiple adjacent scalar words in one instruction. It can reduce load-instruction and address-generation pressure and expose useful instruction-level parallelism. It does not widen an already coalesced warp transaction beyond what the memory subsystem supports, guarantee higher bandwidth, or make a 32-bit load inherently wasteful.

```cuda
// Compiler-visible 16-byte access. The pointer and index must satisfy alignment.
uint4 value = reinterpret_cast<const uint4*>(ptr)[index];
float sum = float(value.x) + float(value.y);
out[index] = sum + float(value.z) + float(value.w);
```

Use a type or alignment contract the compiler can prove. For tails or arbitrary offsets, select a scalar/masked fallback. Inspect generated code: the compiler may split an access when alignment, register pressure, or target rules require it.

## Width tradeoffs

Wider per-thread accesses increase live registers and may over-fetch unused elements. A 32-byte load of packed FP4 contains 64 four-bit values, but decoding and accumulating all of them can become the limiting dependency. Sweep load width with identical work and report both memory transactions and register/spill changes.

## Cache hints

PTX supports cache operators and eviction-priority hints such as `L1::no_allocate` and `L1::evict_last` for applicable global loads. These are performance hints, not semantic promises:

```ptx
ld.global.L1::no_allocate.v4.u32 {r0,r1,r2,r3}, [streaming_ptr];
ld.global.L1::evict_last.v4.u32  {s0,s1,s2,s3}, [reused_ptr];
```

The “streaming matrix, reused vector” policy can help a GEMV, but reuse distance, cross-CTA sharing, L2 behavior, and other traffic determine the result. `.nc` selects a distinct read-only/non-coherent path with its own semantic restrictions; it is not a generic faster load.

## Register budget interaction

Vectorization and unrolling often raise registers per thread. Limiting registers may increase residency, but can also cause spills or inhibit instruction scheduling. Use compiler reports, occupancy/resource calculations, and end-to-end measurements together.

## Evidence boundary

The GPU Mode NVFP4 competition sources report large end-to-end improvements across a sequence that changed layout, coalescing, decoding, PTX, cache hints, vector width, instruction-level parallelism, and register tuning. Those numbers are valuable case studies, but they are not a causal “89x from vectorized loads” result and do not generalize to every GEMV or decode shape.

## Checklist

- Prove alignment and bounds for every vector path.
- Compare scalar, 8-byte, 16-byte, and (where legal) 32-byte variants.
- Check transaction efficiency, load-instruction pressure, registers, and spills.
- Validate FP4 nibble order, block-scale indexing, accumulation precision, and tails.
- Keep cache hints only when repeated measurements show a benefit on representative workloads.
