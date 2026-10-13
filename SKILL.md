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

### Path 1: Unified search guided by optimization evidence

Use `query.py` as the research entry point. Choose a ready-to-use strategy from
the current evidence; ordinary retrieval needs neither Python code nor JSON.

| Strategy | Choose when | Behavior |
| --- | --- | --- |
| `keyword` | Initial discovery or one focused question | Master's keywords, aliases, filters and fixed score; `master` remains a compatible alias |
| `multi-query` | Several complementary hypotheses need evidence | Score each query separately, alternate unseen candidates, deduplicate |
| `guided` | Measurements change what matters next | Apply agent-selected focus keywords, preferred/deferred IDs, filters and optional extra queries |

These are peer choices; `keyword` is not a required first step. Start directly
with guided when measurements already identify a priority, or multi-query when
complementary hypotheses need evidence. Select by the current question, not a
quota of calls to each preset.

Turn natural-language questions and profiling evidence into short technical
queries. For example, "why is my pipeline waiting?" can become `mbarrier` or
`pipeline` with focus `producer wait` when the trace supports that hypothesis.
Inspect the returned evidence and revise terms or filters if results are too
broad or miss the mechanism. Matching is substring OR with aliases: adding words
can broaden recall, and quoting a sentence does not make it a phrase search.
This query reformulation also works with master; it is an agent decision, not
automatic semantic matching by the script.

Run from the skill directory, with logs under the optimization workspace:

```bash
search_run="<optimization-directory>/tmp/kernelwiki-search"
python3 scripts/query.py "pipeline" --strategy keyword --type pattern --run-dir "$search_run" --compact
python3 scripts/query.py --strategy multi-query --query "GEMM tile" --query "TMA layout" --query "mbarrier wait" --run-dir "$search_run" --compact
python3 scripts/query.py "pipeline" --strategy guided --type pattern --focus "throughput" --run-dir "$search_run" --compact
```

Choose guided controls according to the evidence:

- `--focus <terms>`: new measurements change the direction to investigate.
  Use short technical terms; this reranks recalled pages with the existing
  keyword score, without interpreting profiling prose. Result excerpts prefer
  matching focus terms in the body, using the same aliases.
- `--prefer <id>`: an observed candidate is directly applicable to the current
  question. Put it first before the output budget is applied.
- `--defer <id>`: a recurring candidate has little remaining value for the current
  question because its relevant material is exhausted, the issue is resolved,
  or its applicability conditions do not fit. Move it to the end, without
  exclusion. Having read a page alone is not a reason to defer it.
- `--reason <text>`: record the measurement and conditions supporting the choice.
  It records evidence; only the selection parameters change ranking.

Prefer/defer are repeatable and override focus ordering. There is no required
focus-then-prefer sequence or per-round usage quota. Both need IDs in the current
recall pool: change the query/filters or use `--extra-query <text>` if the needed
page is missing. Neither focus nor prefer/defer broadens recall. See
[context fields and ordering](references/search-strategies.md) for details.

**With the same query and filters and no focus, extra queries or ID decisions,
guided retains master's order before output limits.** Excerpts, budgets and logs
still differ from the master branch. A preset name alone does not imply improved
relevance. The agent chooses the evidence and priorities; Python executes them.

When a new measurement changes the bottleneck hypothesis or an attempt fails,
reconsider the next evidence needed. Update the query, filters, priorities or
strategy when useful; record the reason via `--reason` or `decision.reason` when
using guided. If retaining the decision, explain why it still applies. If existing
evidence is sufficient, continue experimenting. No preset is required every round.

**Two successive queries with the same keyword.** The measurements below are
illustrative; replace them with actual evidence. Reuse `search_run` from above.

```bash
python3 scripts/query.py "pipeline" --strategy guided --type pattern --focus "waits" --reason "Illustrative: trace shows producer waits" --run-dir "$search_run" --compact
python3 scripts/get_page.py pattern-pipeline-stalls --run-dir "$search_run" --query-id <first-query-id>
python3 scripts/get_page.py pattern-compute-bound --run-dir "$search_run" --query-id <first-query-id>
```

The first query returns both pages, with `pattern-pipeline-stalls` first in this
corpus. Read them to assess their mechanisms and applicability. Suppose subsequent
measurements at unchanged shape/dtype/layout show that producer waits are resolved,
more pipeline stages give no gain, and math throughput is now limiting. The agent
judges the compute page directly applicable and the already-investigated wait
material less useful for this question. Keep the keyword and update priorities:

```bash
python3 scripts/query.py "pipeline" --strategy guided --type pattern --focus "throughput" --prefer pattern-compute-bound --defer pattern-pipeline-stalls --reason "Illustrative: waits resolved at unchanged shape/dtype/layout; more stages gave no gain; math throughput is now limiting" --run-dir "$search_run" --compact
python3 scripts/get_page.py pattern-compute-bound --run-dir "$search_run" --query-id <second-query-id>
```

The second query explicitly promotes `pattern-compute-bound` and moves
`pattern-pipeline-stalls` to the end; both remain candidates. Selection parameters
cause this change, not the reason text. CLI choices and candidate orders are saved
as separate snapshots. If waits return or conditions change, reconsider the
deferral and reread relevant sections. This is an evidence-based example, not a
fixed round-to-topic rule; no parameter needs to be used just to increase its
call count. If a known page is all that is needed, read it directly.

For longer experiments, [optional JSON context](references/search-strategies.md#optional-reusable-context)
can retain `state.question`, `evidence`, `attempts` and `reads`.
Record resolved questions with measurements and failed approaches with their
shape/dtype/layout/target conditions. Revisit material when conditions change;
reads are not a blacklist. Updating `state` alone does not change ranking.

History logs are not loaded automatically: reusing `--run-dir` does not carry
priorities forward. On CLI-only calls, repeat or update the desired parameters.
When reloading a `--context` file, its saved decisions still apply unless explicitly
overridden; review them for stale priorities. Omitting a CLI flag does not clear
its JSON value. See [context reuse](references/search-strategies.md#optional-reusable-context)
for replacing or clearing decisions.

Only write a custom `strategy.py` when presets cannot express the desired policy.
The optional `scripts/strategy_template.py` and [strategy API](references/search-strategies.md)
retain full control of selection and ordering; independent queries also work.

`query.py` shows actual searches/filters, applied preset decisions, selection
counts, the output budget and the receipt location. A search rule of `master`
describes base retrieval; guided may subsequently reorder its results. Full
emitted/omitted IDs are in the receipt, so long execution details need not
consume candidate space. Guided excerpts show the first focus body match when
available, otherwise the first query body match or an opening preview. Labels
distinguish these excerpts; they are not article abstracts. The `matches` field
still describes the recall query, not the focus used to select an excerpt.

The LLM can adjust `--max-chars` (default 16,384 characters) each call, subject to
the experiment's total budget; optional `--limit` caps candidate count. If useful
evidence is missing, inspect omissions and refine the query, selection or budget.
For single-GPU work, assess compute, memory/layout, pipeline and tail efficiency.
Return to experiments once the evidence supports a concrete change.

Both `--strategy` and `--run-dir` are required. CLI filters constrain every search.
Pass the returned query ID and same run directory to `get_page.py` for linked
reading records. Direct `cat`/`rg` reads are not automatically traced; receipts
show content returned, not model attention. Keep selection reasons in context.

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
