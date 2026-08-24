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
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, contest-gpumode-p1, pr-cccl-3517]
blackwell_relevance: Wide loads and per-stream cache hints are candidates for measured, bandwidth-limited kernels; neither is universally optimal.
---

# Wide Vectorized Loads and Cache Policies

Memory-bound kernels can reduce load-instruction overhead and improve transaction formation by moving aligned contiguous words per thread. The legal and useful width depends on pointer alignment, per-lane address mapping, register pressure, and how the compiler lowers the operation.

## NVFP4 GEMV evidence

Yue Zhang reports the following end-to-end optimization sequence for one GPU Mode problem-1 submission:

| Stage | Reported latency |
|---|---:|
| Naive CUDA attempt | 2000 microseconds |
| Coalesced rewrite | 443 microseconds |
| Hardware-intrinsic rewrite | 39 microseconds |
| PTX rewrite | 27 microseconds |
| Final combined submission | 22.392 microseconds |

The stages combine layout, decoding, PTX, cache-policy, and instruction-level-parallelism changes. They do not establish a vector-load-only speedup. Amandeep Singh separately reports 26.7 microseconds for the CUDA baseline and an 8.6-microsecond speed-of-light estimate on the largest configuration in that article; this is a different benchmark context.

## Evaluation procedure

1. Confirm from profiling and a byte-count roofline that the kernel is bandwidth-limited.
2. Prove alignment and contiguous per-thread access for every supported shape, including tails.
3. Compare scalar and candidate vector widths while holding work mapping and cache policy fixed.
4. Inspect generated instructions and measure memory sectors, transactions, replay, and registers.
5. Only then test cache operators separately for streaming and reused inputs.

Wider source syntax does not guarantee a single wider hardware transaction, and an excessive width can reduce occupancy or complicate tails. Treat register limits and cache operators as independent tuning dimensions rather than part of a universal recipe.

CCCL PR 3517 shows the alignment guard required before its CUB block-load path
reinterprets a scalar pointer as a vector pointer. This is a contiguous excerpt
from the captured upstream patch:

```cpp
using vector_t = typename CubVector<device_word_t, vector_size>::Type;
if (reinterpret_cast<uintptr_t>(block_src_ptr) % alignof(vector_t) == 0)
{
  vector_t vec_items[vectors_per_thread];
  const vector_t* vec_ptr = reinterpret_cast<const vector_t*>(block_src_ptr)
                            + linear_tid * vectors_per_thread;
}
```

The complete upstream branch includes the vector-load loop and fallback path;
the excerpt alone is not a replacement for `BlockLoad`.
