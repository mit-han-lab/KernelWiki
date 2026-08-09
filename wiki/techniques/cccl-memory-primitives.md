---
id: technique-cccl-memory-primitives
title: CCCL CUB Memory Primitives For Selection And Scan
type: technique
architectures: [sm100, sm90]
tags: [cuda-cpp, top-k-selection, parallel-scan, vectorized-loads, cache-policy]
confidence: source-reported
reproducibility: concept
prerequisites: [technique-vectorized-loads, technique-cache-policy]
related: [pattern-memory-bound, technique-tile-scheduling]
sources: [pr-cccl-3559, pr-cccl-6152]
blackwell_relevance: "PR 3559 adds type- and offset-specific SM100 scan policies; PR 6152 is only a generic TopK debug-log correction and provides no Blackwell tuning evidence."
---

## Exact Source Scope

The two linked PRs have different evidentiary value:

| PR | Semantic scope | Transfer value |
|---|---|---|
| CCCL 3559, captured merge `25523da2` | Adds B200/SM100 exclusive-sum scan tuning, expands policy classification to input/output/accumulator/offset types, and updates scan dispatch/tests. | A concrete example of architecture- and type-specific CUB policy selection. |
| CCCL 6152, captured merge `3fb05826` | Changes only `CUB_DEBUG_LOG` formatting and stale variable names in `DispatchTopK`. | Evidence for the corrected debug output, not TopK algorithm, performance, determinism, or SM100 tuning. |

Do not use PR 6152 as evidence for DSA TopK design or performance. Its captured key file exposes surrounding TopK implementation for inspection, but the PR's semantic delta is the small debug-only patch.

## What PR 3559 Actually Tunes

The new `sm100_tuning` specializations select a tuple rather than a single “vectorized” bit:

| Policy dimension | Examples in the captured source |
|---|---|
| Classification | input value size, accumulator type/size, offset size, and recognized `plus` operator |
| Work partition | `threads` and `items` per thread |
| Memory policy | `BLOCK_LOAD_*`, `BLOCK_STORE_*`, and `LOAD_DEFAULT` or `LOAD_CA` |
| Look-back behavior | an exponential backoff/delay constructor and its parameters |
| Fallback | `Policy1000` selects a matching SM100 specialization; otherwise it falls back to `Policy900`; the double specialization explicitly inherits an SM90 tuning |

`items >= 4` does not establish vectorized access. Vector width also depends on iterator contiguity, element type, alignment, load/store algorithm, compiler lowering, and the executed policy. Treat items-per-thread, block size, load/store algorithm, cache modifier, and delay policy as separate variables.

## Transfer Workflow

1. Identify the exact primitive and semantic contract: inclusive/exclusive scan, operator, input/output/accumulator types, offset width, aliasing, and empty/large-size behavior.
2. Follow runtime architecture dispatch to the active policy. Confirm whether SM100 has a matching specialization or inherits the SM90/default route.
3. Reproduce the upstream baseline and changed policy over the size/type/operator matrix relevant to the application. A policy comment containing benchmark ratios is source context, not a portable speedup guarantee.
4. Change one policy dimension at a time where practical and record time, bandwidth, occupancy, register/spill data, and correctness.
5. For selection, separately verify membership, output count, ordering, ties, NaNs, signed zero, key/value association, and repeatability. Those properties are not established by PR 6152.

The linked evidence does not support claims about fill, histogram, reduce, block-load/store vectorization, or application-level DSA score computation. Add a directly relevant CCCL source before transferring policy conclusions to those primitives.
