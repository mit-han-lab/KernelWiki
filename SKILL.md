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

The scripts auto-resolve the wiki root; **no environment variable required**.
You may invoke their absolute paths from the optimization workspace. Relative
strategy, context and run paths resolve from the calling directory. In examples
using `scripts/...`, run from the skill directory and set `search_run` to an
absolute path under the optimization workspace's writable `tmp/` directory.

### Runtime dependencies

The query and maintenance scripts are self-contained. They use the host
PyYAML package when present and fall back to the bundled pure-Python loader and
dumper when it is missing. Invoke them directly with `python3`; do not assume
that `pip`, a virtualenv, network access, or a pre-installed PyYAML package is
available. `requirements.txt` is optional and only installs the faster host
implementation.

### Path 1: Unified search with an explicitly chosen strategy

Use `query.py` as the research entry point. Choose the query shape and
`--strategy` according to the evidence needed:

| Situation | Useful approach |
| --- | --- |
| One precise mechanism or implementation question, such as mbarrier semantics or tcgen05 use in CUTLASS | A focused `master` query |
| Several plausible directions need different filters, such as pipeline-stall patterns and barrier hardware semantics | `interleave` with separate routes, or independent focused queries |
| The next question depends on what a page says | Query, read, then formulate the next query; separate calls are appropriate |
| Candidate priority depends on task-specific compatibility, measurements or previous reads | Adapt `scripts/strategy_template.py` for custom selection and ordering |

The LLM decides which evidence is needed, then chooses the questions, filters,
ranking rules and pages to read. The table illustrates options; any task may
start with any suitable strategy and reuse, combine or switch strategies as
evidence changes. Several independent queries can cover distinct directions;
combining them is useful when a shared, deduplicated candidate pool helps.
There is no required strategy sequence, mix or route count.

Keep generated strategies, task context and logs in the optimization workspace's
`tmp/kernelwiki-search/` (or another explicitly writable experiment directory),
outside the shared wiki corpus. Do not assume system `/tmp` is writable. The
caller supplies `--run-dir`; the scripts do not allocate system temporary files.

The two examples below run from the skill directory and share this setup.
Replace the placeholder with an absolute path in the optimization workspace:

```bash
search_run="<optimization-directory>/tmp/kernelwiki-search"
```

**Focused question:** inspect mbarrier hardware evidence with one query.

```bash
python3 scripts/query.py --strategy master --run-dir "$search_run" "mbarrier" --type hardware --compact
```

**Several directions for one problem:** for producer/consumer waits in a GEMM,
compare symptom patterns with synchronization primitives. Each route gets its
own filter instead of putting all keywords into one pooled score:

```bash
python3 scripts/query.py --strategy interleave --route "pipeline --type pattern --rule master" --route "mbarrier --type hardware" --run-dir "$search_run" --compact
```

Both examples create the run directory automatically. Repeat `--route` for each
query direction; filters and an optional `--rule` go inside its quoted value.
No JSON file is needed. For richer task context, `--context` remains available;
supply routes through either `--route` or `context.queries`, not both.

Interleave takes turns across routes and removes duplicates. It does not
understand semantics or guarantee better ranking. `master` scores within a route;
`keyword` enumerates its matches in path order. Choose either per route. The LLM
still judges applicability and can change routes, their order, or the strategy.

For custom selection, read [the strategy API](references/search-strategies.md)
and adapt the template in the writable run directory. The index supplies corpus
loading, aliases, filters and page access; `index.search()` and `index.master()`
expose every match before candidate selection. Invoke the generated file with
`--strategy <strategy.py>`; no need to write Python for the examples above.

`query.py` preserves strategy order, deduplicates pages and emits a prefix within
`--max-chars` (default 16,000 characters, not tokens). No strategy, including
master, has a default candidate-count cap. Choose optional `--limit` and the
character budget based on the task, or follow the fixed experiment budget.
The summary and receipt disclose omissions. There is no fixed relationship
between candidate count and how many pages you ultimately choose to read.

Assess the returned candidates using the current operator, measured bottleneck,
hardware/dtype/layout compatibility and actionable evidence. For single-GPU work,
consider compute, memory/layout, pipeline and tail efficiency. These judgments
belong to the calling LLM; the Python script makes no model calls. Read promising
pages and revise the questions or strategy if results repeat known evidence,
miss a mechanism or conflict with new measurements. For example, if profiling
shifts the hypothesis from pipeline waits to tile tails, change the query topic
instead of just increasing the same query's limit. Record the choice and useful
evidence in context or the conversation. Return to experiments once the evidence
supports a concrete change. A first-match excerpt is not an article abstract.

For auditable experiments, pass the same `--run-dir` and the returned `--query-id`
to `get_page.py`. It records the content returned, including source previews and
code; it cannot establish what the model attended to. Direct `cat`/`rg` reads are
not automatically traced. Use `get_page.py` for the actual evidence reads you
want recorded, including artifact paths. Explain selections in the experiment
conversation; do not treat candidate order as a completed relevance ranking.

Existing filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--has-code`. CLI filters constrain every strategy
route; put filters that differ between directions inside each `--route` value
or context route object instead.
Both `--strategy` and `--run-dir` are required. Keywords match case-insensitive
substrings with OR; quoted arguments still split on whitespace. Use canonical
terms for multiword aliases, and `rg -F` for literal phrases. The script does not
infer task intent or single-GPU scope.

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
