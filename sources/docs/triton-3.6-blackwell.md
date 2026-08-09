---
id: doc-triton-3.6-blackwell
title: "Triton v3.6.0 — Incremental Blackwell Changes"
url: https://github.com/triton-lang/triton/releases/tag/v3.6.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, 2sm-cooperative, block-scale, warp-specialization]
retrieved_at: 2026-08-08
---

# Triton v3.6.0 — Incremental Blackwell Changes

Triton v3.6.0 was released on 2026-01-21 at commit `7c56a5e40f7fd928dfd5c72902d5def0097db73a`. It is not the first release with a native Blackwell backend: the pinned v3.2.0-to-v3.3.0 comparison in [`doc-triton-3.3-blackwell`](triton-3.3-blackwell.md) establishes the earlier TCGen5/TMEM boundary.

## Blackwell-relevant changes in v3.6.0

The release notes describe incremental work including:

- generic `tcgen05` copy support and broader TMEM bit-width and layout handling;
- more general `tcgen05` load/store layouts and MMA lowering;
- additional aref-style warp-specialization plumbing;
- initial Gluon multi-CTA and 2-CTA support, including `num_ctas`-related cluster work; and
- Gluon scaled-MMA and `tl.dot_scaled` fixes.

These changes extend a backend already present in v3.3.0. The word “initial” matters for the multi-CTA path: the v3.7.0 release subsequently adds more end-to-end 2-CTA, multicast, and TMA work. v3.7.1, published 2026-06-18 at commit `f797708`, fixes two regressions and advertises no new API or feature.

## Evidence boundary

The release notes and pinned compiler tests prove that Triton contains native TCGen5/TMEM paths. They do not prove that every plain `tl.dot` shape selects one, nor that a downstream kernel containing `tl.dot` emitted a particular PTX instruction. Such a claim requires target- and configuration-specific IR or PTX evidence.
