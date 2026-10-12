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

Use `query.py` as the research entry point. During optimization, maintain a small
context file with the current question, measurements, pages read, and approaches
tried with their results. Use that evidence to decide what to search and which
candidates deserve priority. Update the decision when evidence changes; reuse a
strategy while it still fits. There is no fixed round-to-topic or scoring rule.

Start with [the reusable template and example context](references/search-strategies.md#start-a-run).
Keep these files in the optimization workspace's writable `tmp/kernelwiki-search/`,
outside the shared corpus; system `/tmp` need not be writable. From the skill
root, after copying the template and writing context:

```bash
search_run="<optimization-directory>/tmp/kernelwiki-search"
python3 scripts/query.py --strategy "$search_run/strategy.py" --context "$search_run/context.json" --run-dir "$search_run" --compact
```

The template executes the LLM's decision: filter the base query, optionally add
queries, and prioritize observed page IDs before output limits apply. Leave
priorities empty during initial discovery. Adapt the Python when a different
selection rule is useful; it can search multiple times, inspect metadata/body,
and combine or order candidates. A numeric score is optional. The remaining
unprioritized results use search order, which is not semantic relevance.

Read promising candidates with `get_page.py`, then use experiment results to
revise the question, filters or priorities and record why. For example, evidence
of pipeline waits can prioritize synchronization material; measurements of poor
tile tails can shift attention to scheduling. The same operator query can remain
in use. Already-read pages can become useful again under a new hypothesis.
Merely adding profiling prose to context does not change retrieval: **the calling
LLM must translate it into a decision or strategy change**. Python makes no model
calls and does not infer bottlenecks.

Master's scoring remains an explicit option for a focused lookup or baseline:

```bash
python3 scripts/query.py "mbarrier" --strategy master --run-dir "$search_run" --type hardware --compact
```

It uses fixed keyword scores, regardless of optimization state. Within custom
code, `index.master()` is also available alongside unscored `index.search()`.
Both expose all matches before selection; neither has a default top-10 cutoff.

`query.py` deduplicates, preserves strategy order and emits a prefix within
`--max-chars` (default 16,384 characters, adjustable). Optional `--limit` caps the
candidate count. Check reported omissions: narrow a query, revise selection or
adjust the budget if useful evidence may be missing. Excerpts show matches or
opening text, not article abstracts. For single-GPU optimization, assess compute,
memory/layout, pipeline and tail efficiency against measured evidence.

Every query snapshots the strategy, context, searches and candidate decisions.
Pass the same `--run-dir` and returned `--query-id` to `get_page.py` to record
subsequent page/code output. These records show returned content, not model
attention; direct `cat`/`rg` reads are not automatically traced. Keep the reason
for selections and experiment outcomes in context or the conversation.

Both `--strategy` and `--run-dir` are required. Existing filters and alias/OR
matching are retained; CLI filters constrain every search in a strategy. See
[the API and budget details](references/search-strategies.md#strategy-contract)
when adapting code. Return to experiments once the evidence supports a change.

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
