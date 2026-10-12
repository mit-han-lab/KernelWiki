# KernelWiki — Blackwell & Hopper Kernel Optimization Knowledge Base
> [!IMPORTANT]
> This skill is maintained as a standalone submodule of
> [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
> for easy installation.
>
> For bug reports, feature requests, and discussions, please use the main KDA repository:
> https://github.com/mit-han-lab/kernel-design-agents

> **Evidence boundaries:** the audited upstream PR corpus closes at 2026-05-20;
> documentation, releases, and leaderboard snapshots were verified through
> 2026-08-18 UTC.

A structured knowledge base of NVIDIA Blackwell (SM100, B200) and Hopper (SM90, H100) GPU kernel optimization, packaged as a Claude Code skill. The repository root **is** the skill directory — clone it directly into `~/.claude/skills/` and it works out of the box.

## Install as a Claude Code Skill

```bash
git clone git@github.com:mit-han-lab/KernelWiki.git ~/.claude/skills/KernelWiki
```

That's it. The skill auto-registers (because `SKILL.md` lives at the clone root), and the query scripts auto-resolve the wiki root to their own directory — no environment variable or Python package installation is required. The scripts prefer a host PyYAML installation and transparently use the bundled pure-Python fallback when PyYAML is unavailable (including offline or `python3 -S` environments).

PyYAML remains an optional performance dependency. If a package installation is available, it can be enabled with:

```bash
python3 -m pip install -r ~/.claude/skills/KernelWiki/requirements.txt
```

Smoke test:

```bash
cd ~/.claude/skills/KernelWiki
search_run="<optimization-directory>/tmp/kernelwiki-search"
python3 scripts/query.py --strategy master --run-dir "$search_run" --tag nvfp4 --type kernel --compact
python3 scripts/get_page.py kernel-flash-attention-4 --frontmatter-only
```

Optional override for relocating the scripts:

```bash
export BLACKWELL_WIKI_ROOT=/path/to/KernelWiki
```

## What's Here

- Source PR pages, synthesized wiki pages, blog/doc/contest summaries, candidate ledgers, query indices, and artifact bundles.
- Verbatim upstream asset bundles under `artifacts/` (PR patches and complete kernel files or excerpts) — pinned to upstream SHAs via `PROVENANCE.yaml`.
- Auto-generated cross-reference indices — [by architecture](queries/by-architecture.md) / problem / technique / hardware feature / repo / kernel type / language.
- Reviewed candidate ledgers with include/defer/exclude decisions.
- **Hybrid version-claim registry** ([`data/version-claims.yaml`](data/version-claims.yaml)) — per-page `version_sensitive: <id>` pointers + central registry, validated for bidirectional consistency
- Run `python3 scripts/repo_status.py` for current corpus counts.

## Query Tools

Tools locate the wiki from their script path. Use absolute script paths from the
optimization workspace, or run these relative-script examples from the skill root.
Set `search_run` to an absolute writable path under the optimization workspace's
`tmp/`; system `/tmp` access is unnecessary.

| Tool | Purpose |
|---|---|
| `scripts/query.py` | Unified entry point: explicitly select master, interleave or an LLM-authored strategy; emit candidates within a budget |
| `scripts/get_page.py` | Read pages/code; optional `--run-dir` records full reads, previews and code output |
| `scripts/grep_wiki.py` / `rg -F` | Supplementary regex/literal search; use `--only artifacts` or an artifact path to locate captured code |

For focused queries and a complete multi-query example, see the paired recipes
in [SKILL.md](SKILL.md#path-1-unified-search-with-an-explicitly-chosen-strategy).
Use interleave when separate filters and a shared candidate pool help; independent
queries are equally valid when each read informs the next question.

Page and code lookup:

```bash
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/grep_wiki.py "tcgen05" --only wiki
```

The calling LLM chooses `--strategy master` for the original keyword scoring,
`--strategy interleave` with repeated `--route` arguments (or context JSON), or a Python
file for custom selection and ordering. Interleave routes can individually use
master scoring or unscored keyword enumeration. Copy `scripts/strategy_template.py`
into the optimization workspace's `tmp/` when custom logic is useful. The model
chooses queries and candidates, reads evidence, and revises the strategy when
measurements change. The scripts make no model API calls.

Every query requires an explicit strategy and `--run-dir`. No strategy has a
default candidate-count limit; optional `--limit` and the adjustable character
budget control output. There is no implicit master fallback or fixed candidate
multiplier. The index exposes all matches before strategy selection.

Each logged query saves its strategy source, context, runtime snapshot, searches,
proposed/emitted/omitted candidates and exact output. `get_page.py --run-dir
<run-dir> --query-id <query-id>` records subsequent page and code output. Keep
experiment artifacts outside the corpus. Master remains an explicit scoring
option; use `--limit 10` when that count is wanted for comparison.

See [strategy API, budgets and experiment records](references/search-strategies.md).

## Companion Docs

- [`SKILL.md`](SKILL.md) — Skill entry point: when to engage, 5 navigation paths, output contract.
- [`references/primer.md`](references/primer.md) — Topic map: hardware features, techniques, kernels, symptoms → canonical page IDs.
- [`references/schema.md`](references/schema.md) — Frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- [`references/examples.md`](references/examples.md) — 10 worked query patterns (user question → command sequence → synthesis).
- [`CLAUDE.md`](CLAUDE.md) — Extended schema + navigation reference for Claude Code.
- [`index.md`](index.md) — Human-facing curated top-level index.

## Architecture

Three layers (inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)):

1. **`sources/`** — Raw data. Immutable summaries of PRs, blogs, docs, contests.
2. **`wiki/`** — Synthesized knowledge pages. Cross-referenced by `id`. All have YAML frontmatter.
3. **`queries/`** — Auto-generated cross-reference indices. Do not edit manually; regenerate via `scripts/generate-indices.py`.

Supporting files:
- `data/schemas.yaml` — Required/optional fields per page type
- `data/tags.yaml` — Controlled vocabulary (80+ tags)
- `data/aliases.yaml` — Canonical → synonym mappings
- `data/version-claims.yaml` — Central registry for version-sensitive claims (DEC-1 hybrid)
- `data/tool-versions.yaml` — Snapshot of tracked tool releases (Triton, CUTLASS, CUDA, PTX, …)
- `data/refresh-cutoff.yaml` — Internal refresh-round metadata used by validators
- `candidates/` — Reviewed PR candidate ledgers (per repo)
- `artifacts/` — Verbatim upstream asset bundles, each with `PROVENANCE.yaml`

## Maintenance Tooling

| Script | Purpose |
|---|---|
| `scripts/validate.py` | Validate YAML frontmatter, enforce schema, check link integrity |
| `scripts/generate-indices.py` | Regenerate `queries/*.md` from frontmatter |
| `scripts/generate-pr-pages.py` | Batch-generate source PR pages from candidate ledgers |
| `scripts/repo_status.py` | Print current corpus counts |

```bash
python3 scripts/validate.py
python3 scripts/repo_status.py
python3 scripts/generate-indices.py    # regenerate query indices
```

These commands also work without PyYAML. Installing `requirements.txt` is optional and only selects the host implementation.

## Quality Gates

- `scripts/validate.py` reports 0 validation errors
- `scripts/verify_verbatim.py` verifies upstream-pinned assets
- `scripts/check_dod_fixtures.py` verifies every active Definition-of-Done asset contract and audits retired fixture tombstones
- `scripts/verify_pr_architecture_upstream.py` re-derives a deterministic high-risk PR sample from live paginated GitHub evidence
- `scripts/verify_core_prs.py` verifies generated PR manifests
- `scripts/repo_size_check.py` enforces the repository size budget
- 0 broken links across all internal references
- All `verified` wiki pages have official-doc + upstream-code evidence (enforced by `evidence_basis` field)
- All technique/kernel/language pages have compilable code snippets (`reproducibility >= snippet`)
- All Hopper-only wiki pages explain their `blackwell_relevance`; source pages preserve upstream evidence and are exempt
- Version-sensitive claims (currently Triton 3.6) carry `version_sensitive: <id>` pointers resolving to the central registry

## Scope Rules

- **Blackwell-first** — SM100 content is primary. Hopper-only wiki pages require an explicit `blackwell_relevance` field; source pages are exempt.
- **Kernel-only** — No distributed-system topics (DeepEP, DualPipe, EPLB are out of scope).
- **English canonical** — All content in English.
- **First-class DSLs** — CuTe DSL, CUDA C++, PTX, Triton. TileLang / cuTile / JAX-Pallas mentioned but no dedicated guides.

## Repository Layout

```
KernelWiki/                             (= ~/.claude/skills/KernelWiki/)
├── SKILL.md                           # Skill entry point
├── README.md                          # This file
├── CLAUDE.md                          # Extended navigation + schema reference
├── index.md                           # Curated top-level index
├── requirements.txt                   # Optional host PyYAML dependency
│
├── scripts/                           # Query tools + maintenance tooling
│   ├── _yaml_compat.py                # Host PyYAML / offline fallback selector
│   ├── query.py                       # Unified search
│   ├── get_page.py                    # Page fetcher
│   ├── grep_wiki.py                   # Regex search
│   ├── _wiki_root.py                  # Shared root resolver
│   ├── validate.py                    # Schema validator
│   ├── generate-indices.py            # Query-index generator
│   └── generate-pr-pages.py           # Batch PR page generator
│
├── references/                        # Skill knowledge layer
│   ├── primer.md                      # Topic map
│   ├── schema.md                      # Condensed schema reference
│   └── examples.md                    # 10 worked query patterns
│
├── data/                              # Schema + vocabulary
│   ├── schemas.yaml
│   ├── tags.yaml
│   └── aliases.yaml
│
├── candidates/                        # Reviewed PR ledgers (ingestion source of truth)
│   ├── cutlass.yaml
│   ├── sglang.yaml
│   ├── vllm.yaml
│   ├── flashinfer.yaml
│   ├── pytorch.yaml
│   └── deepgemm.yaml
│
├── sources/                           # Layer 1: raw data
│   ├── prs/{repo}/PR-{N}.md
│   ├── contests/{contest}/
│   ├── docs/
│   └── blogs/
│
├── wiki/                              # Layer 2: synthesized knowledge
│   ├── hardware/
│   ├── techniques/
│   ├── kernels/
│   ├── patterns/
│   ├── languages/
│   └── migration/
│
└── queries/                           # Layer 3: auto-generated indices
    ├── by-problem.md
    ├── by-architecture.md
    ├── by-technique.md
    ├── by-hardware-feature.md
    ├── by-repo.md
    ├── by-kernel-type.md
    └── by-language.md
```

## License

Summaries and wiki syntheses in this repository are derivative works citing upstream PRs, blogs, and docs. The tooling (`scripts/`, `references/`, `data/`) is MIT-style; see individual files for any exceptions.
