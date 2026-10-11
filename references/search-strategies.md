# Task-specific search strategies

The calling LLM writes an independent Python strategy for the current task.
`query.py` supplies stable retrieval primitives, executes that strategy and emits
candidate cards within a budget. The LLM assesses those cards and reads useful
pages. No model API is called by the scripts.

## Start a run

Keep generated code, task context and logs outside the shared wiki. This example
uses a temporary directory; choose a persistent experiment directory when results
need to survive temporary-file cleanup.

```bash
search_run=$(mktemp -d /tmp/kernelwiki-search.XXXXXX)
cp scripts/strategy_template.py "$search_run/strategy.py"
cat > "$search_run/context.json" <<'JSON'
{
  "task": "Inspect a single-GPU Blackwell GEMM with producer/consumer waits",
  "queries": [
    {"query": "pipeline", "type": "pattern"},
    {"query": "mbarrier", "type": "hardware"}
  ]
}
JSON
python3 ./scripts/query.py --strategy "$search_run/strategy.py" --context "$search_run/context.json" --run-dir "$search_run" --max-chars 12000 --compact
```

Adapt `retrieve()` before using the template for an actual task. The starter
interleaves all matches from the supplied queries and removes duplicates. That
policy is only an executable example: it assigns no relevance scores and can
still present low-value candidates. The LLM may change routes, filters, ordering
and selection, or use a single query. There are no fixed per-route quotas.

## Strategy contract

Define `retrieve(index, context)` and return a **list or tuple of page IDs or
wiki-relative paths**. The runner deduplicates while preserving your order.
Unknown pages and pages excluded by CLI filters are errors, not silent omissions.
An invalid strategy produces an error receipt; it does not fall back to a
different retrieval policy.

| API | Behavior |
| --- | --- |
| `index.search(query="", **filters)` | All matching page IDs, without relevance scoring or count truncation; path order is deterministic enumeration |
| `index.metadata(id_or_path)` | A copy of the page's frontmatter |
| `index.read(id_or_path)` | Body text for Python strategy logic; recorded as a strategy read |
| `index.keyword_score(id_or_path, query)` | Optional master heuristic: title +10, metadata +5, body hits capped at 3, best alias variant per keyword |

`search` uses master's case-insensitive substring OR matching and alias expansion.
Quoted phrases still split into words. Filters are `type`, `tag`, `repo`,
`language`, `architecture`, `symptom`, `confidence` and boolean `has_code`.
For example, `index.search("", tag="UMMA", architecture="B200")` uses the
existing aliases and architecture rules. CLI filters apply before the index is
provided to the strategy, so a route cannot broaden them.

There is no mandatory keyword-score cutoff in strategy mode. If using scores or
metadata to reduce a large pool, make that choice in the generated strategy.
Searching different mechanisms can expose additional candidates, but may also
displace relevant pages. Inspect omissions and follow page links when evidence
is missing rather than assuming every task needs multiple routes.

`context` is the JSON object supplied with `--context`, or an empty object. CLI
query words override `context["query"]`; otherwise it defaults to an empty string.
The runner sets `context["budget"]` to `{"max_chars": ..., "limit": ...}`.
Other keys are yours to define for operator, hardware, measurements and scope.
Changing the context inside Python does not change the runner's limits.

Strategies are ordinary local Python, executed with the caller's permissions;
the loader is not a sandbox. Keep the generated file self-contained with
standard-library imports so its captured source can be replayed. File accesses
or queries performed outside the index API are not automatically traced.

## Output budget and relevance

- `--max-chars` defaults to 16,000 and bounds candidate text plus its summary.
  It is a character budget, not a model-specific token estimate.
- With `--strategy`, there is no default candidate-count cap. Optional `--limit`
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

Without `--strategy`, ordinary queries use master's keyword ordering (path order
without keywords) and default `--limit 10`, also subject to `--max-chars`.
`--run-dir` can record this mode for a baseline experiment.

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
    strategy.py        # exact executed source; absent for built-in mode
    engine/            # retrieval and page-reader script snapshots
    result.json        # corpus hash, searches, proposals, output omissions
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
