# Schema Reference

Condensed reference for the wiki's controlled vocabulary and page schemas. Full definitions live in `data/schemas.yaml`.

## Page Types and IDs

Every page has a unique `id` with a type-specific prefix:

| Type | ID Prefix | Purpose |
|------|-----------|---------|
| source-pr | `pr-<repo>-<N>` | A tracked PR record; `status` distinguishes merged, open, and closed |
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
```yaml
id: pr-cutlass-2472
repo: NVIDIA/cutlass
pr: 2472
title: "Add Blackwell MLA forward"
author: username
date: 2025-07-16
url: https://github.com/NVIDIA/cutlass/pull/2472
source_category: upstream-code
architectures: [sm100]
tags: [mla, attention, prefill]
techniques: [warp-specialization, pipeline-stages]
hardware_features: [tcgen05, tmem, tma]
kernel_types: [mla, attention, prefill]
languages: [cute-dsl]
captured_at: 2026-04-17
status: merged
merge_sha: abc12345
inclusion_reason: "kernel file changes"
changed_paths: [...]
changed_paths_total: 12
changed_paths_truncated: true
upstream_body_text_sha256: <sha256-of-full-GitHub-GraphQL-bodyText>
upstream_files_sha256: <sha256-of-ordered-full-path-list>
```

PR bodies contain only a bounded prefix of GitHub GraphQL `bodyText`, with
line-end whitespace normalized locally. GitHub constructs `bodyText` as a
plain-text rendering, so Markdown markers, HTML comments, and link syntax may
differ from the raw PR body. The full unnormalized `bodyText` is identified by
SHA-256, and the audit verifier
re-fetches GitHub and checks both hashes, the stored path prefix/count, all
metadata/classifications, and the deterministic rendering; local schema
validation alone is not an authoritative upstream comparison.

For generated PR records, `architectures`, `tags`, `hardware_features`,
`kernel_types`, and `languages` are deterministic relevance surfaces derived
from the upstream title and complete changed-path list. They describe what the
record touches or mentions; they do not by themselves claim that the PR's
primary intent was to add that feature or that every changed source implements
it. The title, bounded upstream prefix, URL, full-list count/hash, and
`inclusion_reason` provide the surrounding scope.

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
    shape: "paper benchmark sweep; peak configuration not identified in prose"
    metric: TFLOPS
    value: 1613
    utilization: "71%"
    source_id: doc-flash-attention-4
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
| `snippet` | Fenced code or logical-reference fragment; the page labels pseudocode/non-executable boundaries (the validator does not compile it) |
| `runnable` | Self-contained buildable example |
| `benchmarked` | Runnable + perf numbers with env metadata |

## Controlled Vocabulary

All values in these fields must appear in `data/tags.yaml`:

- **architectures**: blackwell, sm80, sm90, sm90a, sm100, sm100a, sm103, sm103a,
  sm110, sm110a, sm120, sm120a, sm121, sm121a
- **hardware_features**: tcgen05, tmem, tma, clc, 2sm-cooperative, pdl, gdc, nvfp4, fp8, fp6, fp4, block-scale, wgmma, cluster, mbarrier, ldmatrix, stmatrix
- **techniques**: warp-specialization, persistent-kernel, swizzling, pipeline-stages, double-buffering, register-reuse, shared-memory-optimization, tma-multicast, epilogue-fusion, tile-scheduling, communication-overlap, software-exp, ping-pong-scheduling, conditional-rescaling, loop-unrolling, vectorized-loads, cache-policy, register-budgeting, per-k-specialization, data-reuse, kernel-fusion, chunk-parallelism, fine-grained-quantization, cuda-core-promotion, jit-compilation, top-k-selection, parallel-scan, stream-k
- **kernel_types**: gemm, attention, moe, sparse-attention, gemv, grouped-gemm, gated-delta-net, fused-kernel, decode, prefill, quantization, flash-attention, mla, linear-attention, gated-dual-gemm, batched-gemv, topk, scan, reduction, sort
- **languages**: mojo, cuda-cpp, cute-dsl, triton, tilelang, cutile, ptx, python, jax-pallas
- **source_category**: official-doc, upstream-code, paper, benchmark-blog, contest-report, community-note

## Canonical Aliases (from data/aliases.yaml)

When asking about:
- UMMA → canonical tag is `tcgen05`
- Tensor Memory / TMEM → `tmem`
- Cluster Launch Control / CLC → `clc`
- Blackwell / B200 → architecture `sm100`
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

Pages including `sm90` in `architectures` WITHOUT any `sm100*` variant MUST include a `blackwell_relevance:` field explaining why the Hopper content is kept. Enforced by validator.
