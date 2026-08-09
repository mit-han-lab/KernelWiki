---
id: technique-cache-policy
title: "PTX Cache Eviction and Prefetch Hints"
type: technique
architectures: [sm100, sm90]
tags: [cache-policy, vectorized-loads]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
reproducibility: concept
prerequisites: []
related: [technique-vectorized-loads, kernel-nvfp4-gemv, pattern-memory-bound]
sources: [doc-ptx-isa-sm100, blog-yue-nvfp4, blog-amandeep-nvfp4, contest-gpumode-p1]
blackwell_relevance: "PTX 9.0 exposes L1/L2 eviction priorities and L2 prefetch-size hints on SM100; these remain performance hints rather than cache-residency guarantees."
---

# PTX Cache Eviction and Prefetch Hints

## Contract

PTX global loads and stores can carry cache operators and eviction-priority qualifiers. Global loads can also carry L2 prefetch-size hints. These controls let a kernel express a preference for a particular access; they do not guarantee cache admission, residence, eviction time, hit rate, or performance.

| Form | PTX meaning | Not guaranteed |
|---|---|---|
| `L1::no_allocate` | Selects an L1 eviction priority that may be applied | That the access bypasses L1 |
| `L1::evict_first` | Requests first-eviction priority | Immediate eviction |
| `L1::evict_last` | Requests last-eviction priority | Persistent residence |
| `L2::64B`, `128B`, `256B` | Hints that additional data of that size be fetched into L2 | A wider memory instruction or a completed prefetch |
| `L2::cache_hint` with a policy operand | Supplies a created L2 eviction policy | That the hint is respected or changes memory consistency |

The following are legal PTX 9.0 instruction fragments when their operands and addresses are declared with matching types. Each vector access shown is 16 bytes and therefore requires 16-byte natural alignment:

```ptx
.reg .b64 addr_a, addr_b, addr_c;
.reg .u32 a<4>, b<4>, c<4>;

ld.global.L1::no_allocate.v4.u32 {a0, a1, a2, a3}, [addr_a];
ld.global.L1::evict_last.v4.u32  {b0, b1, b2, b3}, [addr_b];
st.global.L1::evict_first.v4.u32 [addr_c], {c0, c1, c2, c3};
```

The common “stream A, retain B” explanation is a hypothesis about reuse and interference, not the semantics of the code. Even with correct alignment, the hardware may apply the priorities differently than a literal bypass/keep model suggests.

## Selecting a policy

Start from the concrete access trace rather than a kernel label or tensor name:

1. Identify which addresses each warp touches, the reuse distance for each line, the live working set, and competing traffic at the same launch shape.
2. Establish a correct default-policy baseline. Hold mapping, vector width, decode, unrolling, register controls, compiler, inputs, and launch parameters constant.
3. Change one qualifier at a time. Re-run the same correctness oracle, including sizes that exercise alignment and tail paths.
4. Record generated instructions and resources, then profile cache hit behavior, requested/transferred bytes, stalls, achieved bandwidth, and active warps with metrics available on the installed tool and target.
5. Benchmark every production shape with the same warmup, synchronization, repetitions, and statistic. Retain only scoped improvements; a policy may help one shape and regress another.

An input being larger than L2, a kernel being described as memory-bound, or a tensor being called streamed/reused does not by itself predict a useful qualifier. Concurrent blocks can reuse data, a nominally reused vector may exceed the effective working set, and a hint can alter other traffic. Those are measurements, not properties inferred from names.

## Evidence boundary for the NVFP4 reports

Amandeep Singh reports that three inspected B200 NVFP4 GEMV solutions used `L1::no_allocate` for A and `L1::evict_last` for B alongside raw PTX decode, wide loads, exact-K specialization, tighter register caps, and—in one solution—sharing B reads across M rows. The report does not release those contestant implementations or a cache-policy-only ablation, and it does not support the former page's rank-1 per-K policy attribution.

Yue Zhang reports approximately 39 microseconds for a stage combining removal of B shared-memory staging, per-thread tiles, `float4` loads, and hardware conversion, followed by approximately 27 microseconds for a vectorized PTX FP4/scale-decode stage. Those endpoints change the load and decode path together. They do not measure a cache-hint-only speedup or establish cache policy as the dominant lever for memory-bound kernels.
