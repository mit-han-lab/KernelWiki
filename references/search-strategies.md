# Retrieval guided by optimization evidence

`query.py` is the unified entry point. The calling LLM interprets the current
question, measurements and experiment history, then writes the retrieval decision
or Python strategy. The runner executes it, budgets candidate output and records
what happened. No model API is called by these scripts.

## Start a run

Keep strategy, context and logs in the optimization workspace's writable `tmp/`,
outside the shared corpus. Run this setup from the wiki root, replacing the
placeholder with an absolute workspace path. System `/tmp` is unnecessary.

```bash
search_run="<optimization-directory>/tmp/kernelwiki-search"
mkdir -p "$search_run"
cp scripts/strategy_template.py "$search_run/strategy.py"
cat > "$search_run/context.json" <<'JSON'
{
  "query": "pipeline",
  "state": {
    "question": "Which producer/consumer dependency should I investigate?",
    "evidence": [],
    "attempts": [],
    "reads": []
  },
  "decision": {
    "filters": {"type": "pattern"},
    "priority_ids": [],
    "reason": "Inspect stall patterns before choosing an experiment."
  }
}
JSON
python3 scripts/query.py --strategy "$search_run/strategy.py" --context "$search_run/context.json" --run-dir "$search_run" --compact
```

This is an exploratory example, not a measurement claim. Use your actual question
and evidence. `state` is a lightweight record for the LLM: preserve relevant
measurements and their provenance, pages read, and attempted changes with results.
Its keys are suggestions; the runner does not require a workflow-state schema.
In `evidence`, record which question is resolved and the measurement supporting
that conclusion. In `attempts`, record the change, outcome and relevant
shape/dtype/layout/target conditions. Use `reads` to identify available evidence,
not a blacklist. Keep unresolved questions in `question`. The LLM must translate
these records into a changed query, filters, priorities or strategy when needed;
state prose is not interpreted by the runner.

The reusable template executes `decision`:

- `filters` narrows the base `query`.
- `extra_queries` optionally adds searches with their own filters; separate calls
  are also suitable and no combination is required.
- `priority_ids` puts observed, matching page IDs first, in the LLM's chosen order.
  Start empty; after inspecting candidates/pages, use real IDs, not invented ones.
- `reason` records why this decision follows from the current evidence.

The template uses `index.master()` to order each search by the existing keyword
score, without a count cutoff. It then promotes `priority_ids`. Remaining results
retain that base order; this heuristic does not assess semantic relevance. An
empty keyword query still uses path order within its filters. Optional extra
searches are appended in caller-chosen order, with each search scored separately;
these are not pooled relevance scores. The runner deduplicates before budgeting.
To use another ordering or an unscored search, edit the strategy. Priorities that
no longer match the searches or filters fail explicitly, so a stale decision
cannot silently select unrelated pages.

For the same query and filters, an unmodified template with `priority_ids` and
`extra_queries` empty or omitted returns master's candidate order before output
limits. Output details, budgets and receipts still differ from the master branch.
This applies to optional decision fields: the template still requires a
`decision` object, and the CLI requires `--strategy` and `--run-dir`.

After reading and measuring, update `state` and translate new evidence into
`decision` (or edit the strategy). Reconsider retrieval when a measurement changes
the bottleneck hypothesis or an attempt fails; record the reason for a changed or
retained decision in `decision.reason`. If existing evidence is sufficient,
continue experimenting. For example, if results now support a pipeline
wait hypothesis, keep `query: "pipeline"` and prioritize the observed
`pattern-pipeline-stalls` ID. If a later trace points elsewhere, change priorities,
filters or queries and record the reason. Previously read pages are not excluded;
a new hypothesis may require rereading a different section. A failed experiment
is evidence to reconsider applicability under its recorded conditions, not a
permanent ban on that technique. `priority_ids` promotes pages; an empty list does
not demote anything. To deprioritize solved material, promote other applicable
pages or explicitly change ordering/selection in the generated Python. Do not
claim a negative weight or exclusion was applied merely because `state` changed.

Updating prose in `state` alone does **not** change the order. The LLM must make
an executable decision. This avoids hardcoding optimization stages or pretending
that Python interprets profiling. Reuse the same files while useful: each query
saves a separate snapshot before they are edited again. Read via `get_page.py`
with the printed query ID to associate evidence with that decision. The
[complete two-query example](../SKILL.md#path-1-unified-search-guided-by-optimization-evidence)
keeps the keyword fixed and records a resolved issue and a conditional failed
attempt before changing the explicit priority.

## Custom selection and multiple searches

The template is a starting point, not a required policy. A custom strategy can
combine multiple searches, use different filters, order by task-specific metadata,
or inspect body evidence before selecting candidates. For example:

```python
def retrieve(index, context):
    # Example: put hardware semantics before symptom patterns for this question.
    semantics = index.search("mbarrier", type="hardware")
    patterns = index.master(context["query"], type="pattern")
    return semantics + patterns
```

The runner removes duplicates and preserves first occurrence. If useful, the LLM
can implement round-robin merging or its own scoring in ordinary Python; no
separate composition command or fixed per-query quota is required. Independent
calls also work when reading one result determines the next question. Choose
based on evidence needs, not a target number of search methods used.

For a focused lookup or a scoring baseline, master remains directly available:

```bash
python3 scripts/query.py "mbarrier" --strategy master --run-dir "$search_run" --type hardware --compact
```

This uses the original fixed keyword score, not context-dependent relevance.
The named master strategy accepts `context.query` and CLI filters; it rejects
`decision` and the former `queries` plan key instead of ignoring an intended
custom decision. `index.master()` remains usable inside custom code.

## Strategy contract

Both `--strategy` and `--run-dir` are required. Select `master` or a Python file
that defines `retrieve(index, context)` and returns a **list or tuple of page IDs
or wiki-relative paths**, in presentation order. Unknown or globally excluded
pages fail explicitly. A failing strategy records an error without falling back
to master. Old `--strategy interleave` / `--route` calls must use custom code.

| API | Behavior |
| --- | --- |
| `index.search(query="", **filters)` | All OR/alias matches, no relevance scores or count cutoff; deterministic path order |
| `index.master(query="", **filters)` | All matches in master's score/path order, without a count cutoff |
| `index.metadata(id_or_path)` | Copied frontmatter for custom selection |
| `index.read(id_or_path)` | Body for Python logic, logged separately from LLM page reads |
| `index.keyword_score(id_or_path, query)` | Optional master score: title +10, metadata +5, body hits capped at 3; best alias variant per keyword |

Matching retains master's case-insensitive substring OR rules and aliases.
Quoted arguments still split on whitespace; use canonical terms for multiword
aliases and supplementary `rg -F` for literal phrases. Filters are `type`, `tag`,
`repo`, `language`, `architecture`, `symptom`, `confidence` and boolean `has_code`.
CLI filters constrain the entire index; per-search filters can only narrow it.
No task intent, architecture or single-GPU scope is inferred from natural language.

`context` is the JSON object from `--context` (empty if omitted). CLI keywords
override `context.query`; otherwise it defaults to an empty string. The template
always searches that base query, even when additional queries are supplied. The
runner adds `budget: {"max_chars": ..., "limit": ...}`. Other keys belong to the
strategy. Changing context inside Python does not change runner limits.

Relative strategy/context/run paths resolve from the caller's working directory;
the corpus resolves from the scripts' location. Strategies are ordinary local
Python executed with caller permissions, not a sandbox. Prefer self-contained
standard-library code for replay. Reads outside the index API are not traced.

## Output budget and relevance

The result header reports the effective budget and points to
`queries/<query-id>/result.json` under `--run-dir`. A bounded preview shows actual
index calls (query, rule, local filters and match counts), CLI-wide filters, and
selection counts: distinct matches, matches not proposed, and pages proposed
without a search. It is derived from execution, not from `decision.reason` or
other declared intent. A search's rule does not describe later custom reordering.
Full execution details and complete proposed/emitted/omitted IDs remain in the
receipt; a clipped preview ends with an ellipsis. All header text counts against
the same output budget as the candidate cards.

- `--max-chars` defaults to 16,384 and bounds candidate text plus its summary.
  The LLM may increase or decrease it per call within the experiment's total
  budget. It is a character budget, not a model-specific token estimate. A smaller
  budget truncates output; use query/selection changes to address irrelevant hits.
- For every strategy, there is no default candidate-count cap. Optional `--limit`
  is a maximum number of candidates to emit, with no multiplier or final-N role.
- The runner emits a prefix of strategy order. When the next complete card does
  not fit, it stops and records the remaining IDs. An oversized first card can
  yield zero candidates; use `--compact`, revise selection or increase the budget.
- `--compact` includes title, ID, path, architecture, matched terms and a short
  body excerpt. Default output includes additional metadata. Excerpts are first
  matches (or labeled opening previews), not generated article summaries.
- `--paths-only` sends candidate paths to stdout and the summary to stderr.
  Those paths alone provide little evidence for semantic ranking.

Incidental strategy stdout/stderr is kept out of candidate output. A bounded
diagnostic preview and its original character count are recorded in the run.
Architecture-filter diagnostics remain on stderr separately from candidate text.

The calling LLM judges applicability, chooses promising pages and revises its
judgments after reading. A strategy's numeric score or output order is not an
LLM assessment. Runtime task information guides the LLM's strategy generation;
Python does not automatically interpret natural-language intent.

For ablations, fix overall research budgets and include strategy generation and
revision in the cost. Per-query character limits do not cap cumulative queries
or full-page reads. Compare against the separate master checkout for the original
workflow; selecting `--strategy master` here isolates ordering but retains the
new excerpts, budgets and logging.
Assess which distinct questions were asked, which useful evidence was newly
read and which kernel changes it informed. Strategy-name counts alone do not
measure retrieval diversity: several focused calls may explore different
mechanisms, while an unchanged custom strategy may keep returning the same pages.

## Record retrieval and reading

`--run-dir` is required for strategy execution. Each invocation gets a unique
query ID and keeps its own files, so edits to `strategy.py` never overwrite a
previous strategy snapshot:

```text
<run-dir>/
  events.jsonl
  queries/<query-id>/
    request.json       # CLI arguments, script/corpus commits, runtime and hashes
    context.json
    aliases.yaml       # alias snapshot, when present in the corpus
    strategy.py        # exact executed source, including named strategies
    engine/            # retrieval and page-reader script snapshots
    result.json        # corpus hash, execution summary, searches, proposals, omissions
    output.txt
    strategy-output.txt  # only when the strategy printed diagnostics
  reads/<read-id>/
    receipt.json       # query association, file hashes, view modes
    output.txt
```

`result.json` separates matched-but-not-proposed pages from proposed-but-not-emitted
pages. It also separates Python `index.read()` calls from pages returned to the
LLM. A corpus hash identifies the loaded content; the corpus itself is not copied.
Script and corpus roots are recorded separately when a root override is used.
Replay needs the matching corpus and runtime dependencies as well as captured
strategy/context. Keep a fixed wiki checkout for performance ablations.

Use the query ID printed by the runner to link subsequent reads:

```bash
python3 scripts/get_page.py pattern-pipeline-stalls --run-dir <run-dir> --query-id <query-id>
python3 scripts/get_page.py hw-mbarrier --follow-sources --run-dir <run-dir> --query-id <query-id>
python3 scripts/get_page.py <artifact-path> --run-dir <run-dir> --query-id <query-id>
```

Reading logs distinguish full page, body-only, frontmatter-only, cited-source
preview and artifact output. Source previews are at most 500 characters; bundled
code may be limited to the first 200 KiB per file. The exact returned text is
saved. `--query-id` is optional for independent reads, which remain unlinked;
an unknown query ID is rejected rather than associated with a guessed query.

These records show what the tools returned, not what the model attended to.
Direct `cat`/`rg` reads are not intercepted. During ablations, use `get_page.py`
for the page and artifact content whose reading you want recorded, and retain
the model conversation for relevance judgments, selections and reasons.
