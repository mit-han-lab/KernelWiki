---
name: KernelWiki
description: Use when the user asks about optimizing NVIDIA Blackwell (SM100, B200) or Hopper (SM90, H100) GPU kernels — tcgen05/TMEM/CLC/NVFP4/2-SM cooperative, warp specialization, FlashAttention-4, DeepGEMM, FlashMLA, MoE, grouped GEMM, CuTe-DSL/PTX/Triton on Blackwell, or wants concrete PR references from CUTLASS/SGLang/vLLM/FlashInfer/PyTorch. Do NOT use for generic CUDA Q&A that is not Blackwell/Hopper-specific, host-side framework integration, or distributed systems (DeepEP/EPLB/DualPipe).
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# KernelWiki — Blackwell & Hopper Kernel Optimization Wiki

Query a structured, cross-referenced knowledge base of GPU kernel optimization for NVIDIA Blackwell (SM100) and Hopper (SM90). The repository update date is recorded in `README.md`; run `python3 scripts/repo_status.py` for current corpus counts.

## When To Use This Skill

Trigger this skill when the user asks about:

- **Blackwell/SM100 kernel programming** — tcgen05.mma, TMEM, CLC, 2-SM cooperative, NVFP4, FP8/FP4 block scaling, PDL/GDC
- **Kernel implementations** — FlashAttention-4, DeepGEMM, FlashMLA, NSA, GatedDeltaNet, NVFP4 GEMM/GEMV, fused MoE, gated dual GEMM
- **Performance patterns** — low SM utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls
- **DSLs for Blackwell** — CuTe DSL, CUDA C++ with PTX inline, Triton on Blackwell
- **Hopper → Blackwell migration** — wgmma → tcgen05, register → TMEM accumulators
- **PR references** — "how did vLLM/SGLang/FlashInfer/CUTLASS/PyTorch implement X for SM100?"
- **Competition solutions** — GPU Mode NVFP4 hackathon, FlashInfer MLSys 2026 submissions

Do NOT use this skill for:

- Generic CUDA questions unrelated to Blackwell/Hopper tensor cores
- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe are out of scope

## How To Query

All commands below run from the skill directory (the clone root — the directory this `SKILL.md` lives in). The scripts auto-resolve the wiki root; **no environment variable required**.

### Path 1: Unified search (preferred for natural language)

```bash
python3 scripts/query.py "how to fuse gate-up dual GEMM on Blackwell"
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom tail-effect --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag`
and `--architecture` accept aliases — `--tag UMMA` matches `tcgen05`,
`--architecture B200` matches `sm100`, etc.

### Path 2: Fetch a specific page by id or path

```bash
python3 scripts/get_page.py kernel-flash-attention-4
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py kernel-flash-attention-4 --body-only
```

### Path 3: Regex text search across wiki bodies and PR pages

```bash
python3 scripts/grep_wiki.py "tcgen05\\.fence"
python3 scripts/grep_wiki.py "2-CTA backward" --only wiki
python3 scripts/grep_wiki.py "nvfp4" "block_scale" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — 17 technique pages with architectures, confidence, reproducibility, and source counts
- `queries/by-hardware-feature.md` — tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
- `queries/by-kernel-type.md` — gemm/attention/moe/mla/gated-delta-net → pages
- `queries/by-language.md` — cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
- `queries/by-repo.md` — PR pages grouped by source repository

### Path 5: Primer, schema, examples

Companion docs under `references/`:

- `references/primer.md` — topic map: hardware features, techniques, symptoms, canonical page IDs. Read this first when the question is broad.
- `references/schema.md` — condensed frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- `references/examples.md` — 10 worked query patterns mapping user questions → command sequences → synthesis.

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/docs.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`. Call out when a claim is `experimental` or `inferred`.
4. **Include code snippets** from wiki pages when they exist. Reproducibility
   metadata distinguishes logical or schematic snippets from extracted,
   compilable, and benchmarked assets; do not describe pseudocode as compiled.
5. **Report performance claims with all six fields** — `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id`.

## Knowledge Base Contents

- Source PR pages, synthesized wiki pages, blog/doc/contest summaries, candidate ledgers, query indices, and artifact bundles.
- **Verbatim asset bundles** in `artifacts/` (PR diffs and kernel files) — pinned to upstream SHAs via `PROVENANCE.yaml`; the current audit has no extracted or derived bundles
- **Auto-generated query indices** in `queries/`
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`
- **Hybrid version-claim registry** — per-page `version_sensitive: <id>` pointers + `data/version-claims.yaml` central registry, validated for bidirectional consistency
- **Status script** `scripts/repo_status.py` — current corpus counts
- **Validator** `scripts/validate.py` — schema, link, artifact, and ledger checks
- **Blackwell-first** — SM90 pages only appear when they carry explicit `blackwell_relevance`

To refresh the corpus: run `scripts/refresh_candidate_ledger.py`, regenerate PR pages and query indices, then validate.

## Quality Guarantees

- Every `verified` page has official-doc + upstream-code evidence
- Every technique/kernel/language page carries validator-checked reproducibility
  metadata; the level can range from logical snippet to benchmarked artifact.
- Every PR page has an `inclusion_reason` and records the GitHub state captured for
  that PR; closed proposals are not represented as merged implementation evidence.
- Hopper-inclusive synthesized pages state their Blackwell relevance where the
  schema requires it.
