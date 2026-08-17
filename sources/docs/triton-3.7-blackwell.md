---
id: doc-triton-3.7-blackwell
title: "Triton 3.7.0 Release Notes — Blackwell Follow-on Support"
url: https://github.com/triton-lang/triton/releases/tag/v3.7.0
source_category: official-doc
architectures: [sm100, sm100a]
tags: [triton, tcgen05, tmem, 2sm-cooperative, tma, warp-specialization]
retrieved_at: 2026-08-16
---

# Triton 3.7.0 Release Notes — Blackwell Follow-on Support

Triton 3.7.0 was released on `2026-05-07` at release commit `5f3f125`, before the repository refresh cutoff of `2026-05-20`. It supersedes 3.6.0 as the release of record for the `>=3.6` Blackwell capability claim.

Blackwell-relevant release-note items include:

- end-to-end Gluon multi-CTA / 2-CTA work, including M=64 mode, synchronization cleanup, and TMEM deallocation timing;
- TMA multicast backend support and `tcgen05.mma` multicast support;
- a verifier that rejects overly large `tcgen05.mma` N dimensions instead of permitting a miscompile, plus an MMAv5 illegal-instruction fix;
- nested-loop warp specialization, partition-scheduling improvements, mixed TMA/non-TMA fixes, and other warp-specialization lowering hardening; and
- continued TMEM/shared-layout generalization and Blackwell scale swizzling.

These are incremental additions and fixes. Triton 3.6.0 remains the first stable release in this registry with native Blackwell `tcgen05`/TMEM infrastructure; 3.7.0 is the latest stable release within the cutoff and contains material follow-on support.

Source: [official Triton 3.7.0 release notes](https://github.com/triton-lang/triton/releases/tag/v3.7.0).
