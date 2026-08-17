---
id: doc-triton-3.6-blackwell
title: "Triton 3.6.0 Release Notes — Blackwell (SM100) Lowering"
url: https://github.com/triton-lang/triton/releases/tag/v3.6.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, 2sm-cooperative, block-scale, nvfp4, warp-specialization]
retrieved_at: 2026-08-16
---

# Triton 3.6.0 Release Notes — Blackwell Lowering Expansion

## Overview

Triton 3.6.0 (released `2026-01-21`, release commit `7c56a5e`) is the first stable release tracked in this registry as a broad Blackwell milestone, but it did not originate all `tcgen05` and tensor-memory operations. The release/3.3.x source already defines `tc_gen5_mma`, `tc_gen5_mma_scaled`, `tmem_alloc`, `tmem_load`, `tmem_store`, and `tmem_copy`. Version 3.6 materially generalized and hardened those paths and exposed broader warp-specialization, block-scaled, and Gluon/2CTA work. This record therefore does not repeat either the former blanket WGMMA-fallback claim or a false “introduced in 3.6” claim.

This doc page summarizes only the SM100-relevant items from the 3.6.0 release notes; per-pathway breakdown with verified-vs-needs-verification classification lives in `data/triton-3.6-evidence.md`.

## Blackwell-Relevant Items in the 3.6.0 Release Notes

### Tensor Memory (TMEM) infrastructure

The release expands and hardens TMEM allocation, copy, and layout primitives that the Blackwell backend lowers through `ttng.tmem_alloc`, `ttng.tmem_copy`, `ttng.tmem_load`, and `ttng.tmem_store`. Source PRs: `#8136`, `#8148`, `#8202`. These operation names already existed in the 3.3 branch; the 3.6 work is a maturity and coverage milestone. Accumulators on SM100 may live in TMEM rather than registers, so the older blanket "accumulators stay in registers" claim is not correct.

### `tcgen05` lowering

Generic `tcgen05` load/store/copy lowering and `tcgen05.mma` generalization land via `#8225`, `#8421`, `#8495`, `#8102`, `#8338`, `#8386`. The dialect's `ttng.tc_gen5_mma` and `ttng.tc_gen5_mma_scaled` operations predate this release; 3.6 broadens and refines their lowering and TMEM-token semantics.

### Warp specialization end-to-end

End-to-end aref-style warp specialization plumbing on the Blackwell path: `#8262`, `#7826`, `#8009`, `#8123`, `#8534`, `#8451`, `#8651`. The strongest user-visible surface is `tl.range(..., warp_specialize=True)` on top of descriptor / TMA matmul kernels, as documented in the Triton persistent matmul tutorial.

### Gluon front-end and 2-CTA support

Initial 2-CTA cluster support in the Gluon front-end (`#8644`, `#8653`), `num_ctas` plumbing (`#8645`), and Gluon-side `tcgen05 mma scaled` support (`#8393`). The Gluon path is the most explicit Blackwell-native surface; the release notes describe it as initial support.

### Block-scaled matmul (NVFP4 / MXFP)

Hardware-accelerated block-scaled matmul on Blackwell tensor cores via `tl.dot_scaled`. Backend exposes `ttng.tc_gen5_mma_scaled` (`#8393`); frontend fixes `#8564`, `#8658`. Format coverage centers on NVFP4 / MXFP per the official block-scaled matmul tutorial.

## Predecessor Release for Context

Triton 3.5.1 (released `2025-11-12`) is the last 3.5.x patch before the 3.6 Blackwell story. Pages with `version_sensitive` claims valid for `>=3.5,<3.6` should pin to 3.5.1.

## Successor release

Triton 3.7.0 was released on `2026-05-07`, before this repository's `2026-05-20` refresh cutoff. It is now the release of record for the `>=3.6` registry claim; see [`doc-triton-3.7-blackwell`](triton-3.7-blackwell.md). The 3.6 page remains the source for the registry's broad Blackwell-maturity milestone, not for the origin of every underlying operation.

## When To Cite This Page

Pages making claims about Triton's SM100 capabilities should use the `vs-triton-3.6-blackwell-tcgen05` registry entry. It pins the latest checked stable release, currently 3.7.0, while listing this page and `doc-triton-3.7-blackwell` as official sources. Downstream-code anchors are cataloged separately in `data/triton-3.6-evidence.md`.

## Caveats

- The 3.6 release broadens and hardens native Blackwell lowering paths but does not by itself prove that every plain `tl.dot` matmul on SM100 lowers through TMEM-backed `tcgen05`. The strongest checked path is descriptor/TMA + `tl.range(warp_specialize=True)` + `tl.dot`, plus the Gluon multi-CTA / 2CTA path.
- Release-note support for a lowering surface does not prove performance parity for every workload. Backend comparisons and routing decisions belong to their downstream PR records rather than this official-release summary.
