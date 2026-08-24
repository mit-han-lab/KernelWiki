# Schema Reference

Condensed reference for the wiki's controlled vocabulary and page schemas. Full definitions live in `data/schemas.yaml`.

## Page Types and IDs

Every page has a unique `id` with a type-specific prefix:

| Type | ID Prefix | Purpose |
|------|-----------|---------|
| source-pr | `pr-<repo>-<N>` | A PR from a tracked repo with an evidence-backed status (942 of 944 are `merged`; two are `closed` without merge) |
| source-doc | `doc-*` | Official NVIDIA docs, papers |
| source-blog | `blog-*` | Community blog posts, tutorials |
| source-contest | `contest-*` | Competition problems / tracks |
| wiki-hardware | `hw-*` | Blackwell hardware feature pages |
| wiki-technique | `technique-*` | Optimization techniques |
| wiki-kernel | `kernel-*` | Kernel case studies with perf claims |
| wiki-pattern | `pattern-*` | Problem → solution diagnosis |
| wiki-language | `lang-*` | DSL / language guides |
| wiki-migration | `migration-*` | Hopper → Blackwell migration |

## Required Frontmatter by Type

### source-pr

This example is an exact copy of the current `pr-cutlass-2472` frontmatter;
a regression keeps every concrete field synchronized with the source page.

```yaml
id: pr-cutlass-2472
repo: NVIDIA/cutlass
pr: 2472
title: "Add Blackwell MLA forward (shape: d=192, dv=128) implementation"
author: dianzhangchen
date: '2025-07-16'
url: https://github.com/NVIDIA/cutlass/pull/2472
source_category: upstream-code
architectures: [sm100]
architecture_disposition: exact
architecture_evidence:
  - {architecture: sm100, basis: exact-sm-token, locator: "architecture-guard:examples/77_blackwell_fmha/77_blackwell_mla_fwd.cu", evidence: SM100}
  - {architecture: sm100, basis: exact-sm-token, locator: "changed-path:examples/77_blackwell_fmha/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp", evidence: sm100}
  - {architecture: sm100, basis: exact-sm-token, locator: "changed-path:examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp", evidence: sm100}
tags: [attention, flash-attention, fp8, gemm, mla, tma]
techniques: [persistent-kernel, tile-scheduling]
hardware_features: [fp8, tma]
kernel_types: [attention, flash-attention, gemm, mla]
languages: [cuda-cpp]
captured_at: '2026-08-18'
status: merged
merge_sha: 9baa06dd57804ce8fb5efe9e471b3451341522c6
inclusion_reason: "retain: CUDA/CuTe/PTX device implementation path(s): examples/77_blackwell_fmha/77_blackwell_fmha.cu, examples/77_blackwell_fmha/77_blackwell_mla_fwd.cu"
scope_disposition: retained
scope_evidence:
  rule: cuda-cute-ptx-device-source
  paths:
    - examples/77_blackwell_fmha/77_blackwell_fmha.cu
    - examples/77_blackwell_fmha/77_blackwell_mla_fwd.cu
changed_files_count: 13
changed_files_enumerated_count: 13
changed_files_listing_complete: true
changed_paths:
  - examples/77_blackwell_fmha/77_blackwell_fmha.cu
  - examples/77_blackwell_fmha/77_blackwell_mla_fwd.cu
  - examples/77_blackwell_fmha/CMakeLists.txt
  - examples/77_blackwell_fmha/collective/fmha_fusion.hpp
  - examples/77_blackwell_fmha/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp
  - examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
  - examples/77_blackwell_fmha/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp
  - examples/77_blackwell_fmha/collective/sm100_fmha_mla_load_tma_warpspecialized.hpp
  - examples/77_blackwell_fmha/common/pipeline_mla.hpp
  - examples/77_blackwell_fmha/kernel/fmha_causal_tile_scheduler.hpp
  - examples/77_blackwell_fmha/kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp
  - examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
  - examples/77_blackwell_fmha/reference/fmha_fwd_reference.hpp
changed_paths_complete: true
body_contract: upstream-pr-v1
upstream_body_locator: https://github.com/NVIDIA/cutlass/pull/2472 (PR description)
upstream_body_sha256: 19b902f4f2fa76401afbfb93f0a8f1254b3fdfb3aaf165032aef04932624ad0d
upstream_excerpt_sha256: 371af3e97496ca98f8ef7d0dd9cbd462858c53e92b7e18e791caa36007975c70
upstream_files_sha256: 50d53bdd395636f65807a1c7d88222c7d24bc170761cec0f442184ccd05f59d3
upstream_patches_complete: null
artifact_dir: artifacts/prs/cutlass/PR-2472
```

`changed_files_count` is the PR object's authoritative total;
`changed_files_enumerated_count` is the number returned by the REST files
endpoint. `changed_files_listing_complete: false` exposes that endpoint's
3,000-file cap. A capped page whose complete pull diff was independently
evaluated also records `changed_files_evidence_count`,
`changed_files_evidence_complete`, `changed_files_evidence_method:
github-pull-diff`, and `changed_files_evidence_receipt`; its
`upstream_files_sha256` covers those full-diff evidence records.
For an inconclusive `.cu` hunk, that digest also binds the immutable
PR-head complete-file SHA-256, the device-signal verdict, and the exact policy
pattern digest used to derive the verdict. The digest includes that receipt
only while the current policy requires it, so historical enrichment cannot
make a clean live re-derivation hash differently. A filename extension,
tile/warp shape variables, host launch configuration (`gridDim`/`blockDim`),
host-visible template names such as `GemmType`, or host-side `<<<...>>>`
launch syntax alone is not positive kernel-implementation evidence.

For a non-explicit Python path whose changed hunk contains only weak tuning
names such as `block_m` or `num_warps`, retention requires an immutable
complete-file receipt proving a recognized Triton, CuTe DSL, or TileLang
device-kernel construct. Python comments and strings are removed before this
decision. Exact `cpu/` path components are outside the GPU implementation
scope; a mixed PR can still qualify through a separate GPU implementation
path.
`changed_paths` is the displayed/evidence subset, independently qualified by
`changed_paths_complete`.

`merge_sha` is a full 40-hex commit ID required exactly when `status: merged`;
it is omitted for a PR that closed without merge.

### wiki-kernel (must have `performance_claims`)
```yaml
id: kernel-flash-attention-4
title: "FlashAttention-4"
type: kernel
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative]
confidence: source-reported
reproducibility: snippet
kernel_types: [attention, flash-attention]
languages: [cute-dsl]
related: [technique-warp-specialization, technique-software-exp, hw-tcgen05-mma]
sources: [doc-flash-attention-4, blog-flash-attention-4, pr-...]
performance_claims:
  - gpu: B200
    dtype: bf16
    shape: "paper-reported B200 BF16 sweep; exact maximizing shape not stated"
    metric: TFLOPS
    value: 1613
    utilization: "71%"
    source_id: doc-flash-attention-4
    source_locator: "arXiv 2603.05451 abstract and §5"
```

### wiki-pattern (diagnostic flow)
```yaml
id: pattern-memory-bound
title: "Memory Bandwidth Bound"
type: pattern
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound]
sources: [...]
```

## Confidence Levels

- **`verified`**: Requires ≥1 `official-doc` + ≥1 `upstream-code` in sources. Enforced by validator.
- **`source-reported`**: Cited by ≥1 authoritative source (paper, major blog, major repo).
- **`inferred`**: Synthesized from multiple sources, no single authoritative one.
- **`experimental`**: Undocumented, PTX tricks, version-sensitive. Include CUDA version.

## Reproducibility Levels

For `wiki-technique`, `wiki-kernel`, `wiki-language`, must be ≥ `snippet`.

| Level | Meaning |
|-------|---------|
| `concept` | Text only |
| `pseudocode` | Language-agnostic algorithm |
| `snippet` | Compilable code fragment (verified by validator) |
| `runnable` | Self-contained buildable example |
| `benchmarked` | Runnable + perf numbers with env metadata |

## Controlled Vocabulary

`data/tags.yaml` is the single source of truth for every controlled value in
`architectures`, `hardware_features`, `techniques`, `kernel_types`, `languages`,
`source_categories`, `confidence`, and `reproducibility`. The validator rejects
values absent from that file. This reference deliberately does not duplicate
the lists, so vocabulary additions and removals cannot leave a stale schema
summary behind.

## Canonical Aliases (from data/aliases.yaml)

When asking about:
- UMMA → canonical tag is `tcgen05`
- Tensor Memory / TMEM → `tmem`
- Cluster Launch Control / CLC → `clc`
- Blackwell → architecture family `blackwell`
- B200 / GB200 → architecture `sm100`
- B300 / GB300 → architecture `sm103`
- Hopper / H100 → architecture `sm90`
- MoE / Mixture of Experts → `moe`
- MLA / Multi-head Latent Attention → `mla`
- GDN / Gated Delta Net → `gated-delta-net`
- NSA / Native Sparse Attention → `sparse-attention`
- WGMMA / wgmma.mma_async → `wgmma`

## Cross-Reference Fields

- `sources`: list of source IDs whose content backs this wiki page
- `related`: list of wiki page IDs that are topically related
- `prerequisites`: list of wiki page IDs the reader should read first
- `candidate_techniques` (pattern only): list of technique/hw/migration IDs that address the symptoms

## Blackwell-First Scope

Wiki pages with a Hopper-family value and no Blackwell-family value in `architectures` MUST include a `blackwell_relevance:` field explaining why the Hopper content is kept. The validator derives both families from the canonical exact-target policy, including controlled `a`/`f` variants and the family tokens themselves. Source pages preserve upstream evidence and are exempt. Enforced for `wiki-*` page types.
