---
id: technique-external-source-map-research
title: External Source-Map Research For Kernel Edits
type: technique
architectures:
- sm90a
tags:
- cuda-cpp
- cute-dsl
- tma
- wgmma
- swizzling
- vectorized-loads
- persistent-kernel
confidence: source-reported
reproducibility: snippet
prerequisites:
- technique-vectorized-loads
- technique-swizzling
- technique-persistent-kernels
related:
- technique-pipeline-stages
- technique-tile-scheduling
- lang-cute-dsl
sources:
- blog-nvidia-code-samples
- blog-colfax-article-source-kernels
- blog-colfax-cutlass-kernels
- blog-simveit-effective-transpose
- blog-simveit-load-and-store
blackwell_relevance: The pinned repositories provide Hopper SM90a examples. They can suggest candidate mechanisms for Blackwell, but an SM100 transfer requires separate version-matched source evidence and measurement.
---

## Use

Use external source-map research after a profile or benchmark identifies an edit
family but the local PR pages do not expose a small enough implementation
example. Treat the profiler result as a search key, not a root-cause proof:
inspect a pinned upstream revision and exact implementation files before
adapting an idea.

```bash
git init external/colfax-cfx
git -C external/colfax-cfx remote add origin \
  https://github.com/ColfaxResearch/cfx-article-src
git -C external/colfax-cfx fetch --depth=1 origin \
  fbecfed88de2e4246f104a023188ba722937c5fc
git -C external/colfax-cfx checkout --detach FETCH_HEAD
test "$(git -C external/colfax-cfx rev-parse HEAD)" = \
  fbecfed88de2e4246f104a023188ba722937c5fc

rg -n "SM90_TMA|mbarrier" \
  external/colfax-cfx/tma external/colfax-cfx/pipeline-gemm
rg -n "PersistentTileScheduler|StreamK" external/colfax-cfx/streamk
```

At that revision, `tma/tma_copy.h` contains SM90 TMA load/store and transaction
barrier handling; `pipeline-gemm/` contains multistage and warp-specialized
Hopper GEMMs; and `streamk/tile_scheduler.hpp` contains non-persistent,
data-parallel persistent, and Stream-K scheduler classes. The other pinned
source maps are narrower:

- `simveit/effective_transpose` at
  `994b2b5acaa67f80e411df3e8274b6ae13fd1949` contains SM90a TMA transpose and
  128-byte tensor-map swizzle examples.
- `simveit/load_and_store` at
  `05d828cf910dd43f0053ddbbe4744218a06e9d7f` contains only `ldmatrix` and
  `stmatrix` x1/x2/x4 instruction examples.
- `ColfaxResearch/cutlass-kernels` at
  `84f0802e2b4a1bf068ac70359f20ffdb368c8f6a` contains Hopper SM90a
  GEMM/FMHA TMA, WGMMA, pipelining, and optional warp-specialization examples;
  it is not a persistent-scheduler or SM100 source.
- `NVIDIA-developer-blog/code-samples` at
  `3350d216083a902ccbf5b31665e3b82096a75b55` is a cross-generation collection.
  Exact relevant paths include `series/cuda-cpp/coalescing-global/coalescing.cu`
  and `series/cuda-cpp/transpose/transpose.cu`.

## When It Helps

- For long-scoreboard or poor sector utilization, inspect the exact coalescing,
  transpose, and matrix load/store examples as candidate mechanisms. The metric
  alone does not establish vector width or layout as the cause.
- For barrier or TMA-wait stalls, inspect the SM90 TMA and pipelined GEMM paths
  before proposing stage-count or producer/consumer changes. The examples do
  not guarantee that either edit will help another kernel.
- For tail waves, compare the pinned non-persistent, data-parallel persistent,
  and Stream-K scheduler implementations before adding shape dispatch. Profile
  the target shapes rather than treating persistence as a universal remedy.

## Provenance Rule

Do not cite this page as implementation evidence by itself. Cite one of its
source pages only as a discovery lead. Implementation evidence must name the
upstream repository, full immutable revision, exact file, and a symbol or line
range. For Blackwell transfer, add an SM100-versioned primary source or a
controlled SM100 experiment; Hopper source compatibility must not be assumed.
