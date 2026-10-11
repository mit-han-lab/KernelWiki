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
- **Competition context** — GPU Mode NVFP4 hackathon and FlashInfer MLSys 2026 task definitions, speed-of-light data, and dated results snapshots; participant write-ups are separate blog sources

Do NOT use this skill for:

- Generic CUDA questions unrelated to Blackwell/Hopper tensor cores
- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe are out of scope

## How To Query

All commands below run from the skill directory (the clone root — the directory this `SKILL.md` lives in). The scripts auto-resolve the wiki root; **no environment variable required**.

### Runtime dependencies

The query and maintenance scripts are self-contained. They use the host
PyYAML package when present and fall back to the bundled pure-Python loader and
dumper when it is missing. Invoke them directly with `python3`; do not assume
that `pip`, a virtualenv, network access, or a pre-installed PyYAML package is
available. `requirements.txt` is optional and only installs the faster host
implementation.

### Path 1: Unified search with a task-specific strategy

Use `query.py` as the research entry point. Read
[the strategy interface](references/search-strategies.md), copy
`scripts/strategy_template.py` outside the wiki as `strategy.py`, and adapt its
`retrieve(index, context)` function to the current question and profiling evidence.
Keep generated strategies, task context and run logs outside the shared corpus.

The fixed index supplies corpus loading, OR keyword/alias matching, metadata
filters and page access. Your strategy chooses queries, combinations, selection
and candidate order. `index.search()` returns every match without a score or
count cutoff. Master's keyword score is available only as an optional helper.
Use one or several queries as appropriate; no fixed route count or quota applies.

```bash
python3 scripts/query.py --strategy <strategy.py> --context <context.json> --run-dir <run-dir> --compact
```

`query.py` preserves strategy order, deduplicates pages and emits a prefix within
`--max-chars` (default 16,000 characters, not tokens). In strategy mode there is
no default candidate-count cap; optional `--limit` caps the number emitted.
The summary and receipt disclose omissions. There is no fixed relationship
between candidate count and how many pages you ultimately choose to read.

Assess the returned candidates using the current operator, measured bottleneck,
hardware/dtype/layout compatibility and actionable evidence. For single-GPU work,
consider compute, memory/layout, pipeline and tail efficiency. These judgments
belong to the calling LLM; the Python script makes no model calls. Read promising
pages, revise judgments when their full text changes applicability, and adjust
the strategy if a mechanism is missing. Return to experiments once the evidence
supports a concrete change. A first-match excerpt is not an article abstract.

For auditable experiments, pass the same `--run-dir` and the returned `--query-id`
to `get_page.py`. It records the content returned, including source previews and
code; it cannot establish what the model attended to. Direct `cat`/`rg` reads are
not automatically traced. Use `get_page.py` for the actual evidence reads you
want recorded, including artifact paths. Explain selections in the experiment
conversation; do not treat candidate order as a completed relevance ranking.

The ordinary CLI remains available with master's scoring and `--limit` behavior
(default 10 candidates), subject to the character budget:

```bash
python3 scripts/query.py "gate-up GEMM" --type kernel
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom tail-effect --compact
```

Existing filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--has-code`. CLI filters constrain every strategy
route. Keywords match case-insensitive substrings with OR; quoted arguments still
split on whitespace. Use canonical terms for multiword aliases, and `rg -F` for
literal phrases. The script does not infer task intent or single-GPU scope.

### Path 2: Fetch a specific page by id or path

Use this to read candidates selected by the LLM, follow their cited sources, or
open a page the user explicitly identified. Add `--run-dir <run-dir> --query-id
<query-id>` to associate each read with its retrieval; direct artifact paths also
work.

```bash
python3 scripts/get_page.py kernel-flash-attention-4
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py kernel-flash-attention-4 --body-only
```

`--follow-sources` previews only the first 500 characters of each source body.
Fetch the cited page itself and read applicable code sections for more evidence.

### Path 3: Supplementary text and code search

After unified search, use these tools to locate implementation details or
investigate evidence missing from the candidate pages.

```bash
python3 scripts/grep_wiki.py "tcgen05" --only wiki
python3 scripts/grep_wiki.py "two-CTA" --only wiki
python3 scripts/grep_wiki.py "nvfp4" "block_scale" --any
python3 scripts/grep_wiki.py 'mbarrier.*wait' --only artifacts --limit 5
rg -l -F 'tcgen05.mma' artifacts/prs/cutlass
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-architecture.md` — exact SM, family-only, and validated-unknown architecture evidence
- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — 17 techniques with architectures, confidence, reproducibility, source count
- `queries/by-hardware-feature.md` — tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
- `queries/by-kernel-type.md` — gemm/attention/moe/mla/gated-delta-net → pages
- `queries/by-language.md` — cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
- `queries/by-repo.md` — PR pages grouped by source repository

### Path 5: Primer, schema, examples

Companion docs under `references/`:

- `references/primer.md` — topic map: hardware features, techniques, symptoms, canonical page IDs. Use it to refine queries when the question is broad.
- `references/schema.md` — condensed frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- `references/examples.md` — 10 worked query patterns mapping user questions → command sequences → synthesis.

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/docs.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`. Call out when a claim is `experimental` or `inferred`.
4. **Include code snippets** from wiki pages when they exist — technique/kernel/language pages are guaranteed `snippet`-reproducibility (validator-enforced).
5. **Report performance claims with all six fields** — `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id`.

## Knowledge Base Contents

- Source PR pages, synthesized wiki pages, blog/doc/contest summaries, candidate ledgers, query indices, and artifact bundles.
- **Verbatim upstream asset bundles** in `artifacts/` (PR patches and complete kernel files or excerpts) — pinned to upstream SHAs via `PROVENANCE.yaml`
- **Auto-generated query indices** in `queries/`
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`
- **Hybrid version-claim registry** — per-page `version_sensitive: <id>` pointers + `data/version-claims.yaml` central registry, validated for bidirectional consistency
- **Status script** `scripts/repo_status.py` — current corpus counts
- **Validator** `scripts/validate.py` — schema, link, artifact, and ledger checks
- **Blackwell-first** — Hopper-only wiki pages carry explicit `blackwell_relevance`; source pages preserve upstream evidence and are exempt

To refresh the corpus: run `scripts/refresh_candidate_ledger.py`, regenerate PR pages and query indices, then validate.

## Quality Guarantees

- Every `verified` page has official-doc + upstream-code evidence
- Every technique/kernel/language page has a compilable snippet
- Every PR page has `inclusion_reason` and an evidence-backed status; current distribution: 942 merged, 2 closed without merge
- All Hopper-only wiki pages have explicit `blackwell_relevance`; source pages are exempt
