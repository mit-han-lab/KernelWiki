# Task-specific search strategies

The calling LLM chooses a supplied strategy or writes an independent Python
strategy for the current task. `query.py` supplies stable retrieval primitives,
executes that choice and emits candidate cards within a budget. The LLM assesses
those cards, reads useful pages and revises the retrieval as evidence changes.
No model API is called by the scripts.

## Start a run

Keep generated code, task context and logs outside the shared wiki. Run the
following from the **optimization workspace**, using its writable `tmp/`.
Replace the wiki path with the checkout being evaluated. The scripts resolve the
corpus from their own location and user-supplied paths from the caller's working
directory. Neither queries nor page reads need access to system `/tmp`.

```bash
wiki_root="/path/to/KernelWiki"
search_run="$PWD/tmp/kernelwiki-search"
python3 "$wiki_root/scripts/query.py" --strategy interleave --route "pipeline --type pattern --rule master" --route "mbarrier --type hardware" --run-dir "$search_run" --max-chars 12000 --compact
```

The runner creates `search_run` automatically. No context file is required.
Use a new subdirectory for a separate experiment; query snapshots within each
run are already unique. If local `tmp/` is unavailable, pass another writable
experiment directory explicitly.

## Choose or compose a strategy

`--strategy` and `--run-dir` are required. Missing or failing strategies do not
silently switch to master. A bare keyword command from the earlier interface
must now explicitly select its policy.

| Selection | Behavior | When to consider it |
| --- | --- | --- |
| `--strategy master` | One query, master's score with path ties; filter-only results in path order | A focused keyword query or a baseline ordering |
| `--strategy interleave` | Round-robin distinct results from caller-defined routes, with no per-route quota | Compare mechanisms, symptoms or implementations without pooling every term into one score |
| `--strategy <file.py>` | Execute `retrieve(index, context)` written or adapted by the LLM | Custom filtering, ordering, merging or selection based on task evidence |

For example, GEMM pipeline waits can call for a symptom-pattern route and a
hardware-semantics route with different `type` filters, as above. Interleave
merges those pools without scoring all terms as a single query. Several
independent queries are equally useful when each result informs what to ask
next. A precise question about one mechanism may need only one master query.
Use the evidence needed to choose; there is no required interleave usage rate.

For a focused master query, reusing the variables above:

```bash
python3 "$wiki_root/scripts/query.py" "pipeline stalls" --strategy master --run-dir "$search_run" --limit 12 --compact
```

The number 12 is an example caller choice, not a default. Master preserves
title +10, metadata +5 and capped body scoring; it has **no implicit top-10 cap**
here. Use `--limit 10` to reproduce that count, subject to the output budget.
This strategy uses `context.query` or CLI keywords and CLI filters; it rejects
`context.queries` instead of silently ignoring a multi-query plan.

For interleave, repeat `--route "KEYWORDS [FILTERS] [--rule keyword|master]"`.
Each quoted value accepts the same filters as the main CLI, including
`--has-code`. Multiple words retain OR matching, not exact-phrase matching.
For example, `--route "tile pipeline --type technique"` searches either term
within technique pages. Filter-only routes such as `--route "--type hardware"`,
`--route "--type=hardware"` and `--route "--has-code"` are also valid.
Flags outside the quotes apply to every route; flags inside apply
only to that route. Route text is parsed as arguments, not executed as code.

Alternatively, `context.queries` is a list of route objects containing `query`,
optional filters and `rule`. `rule: "master"` explicitly scores that route;
`rule: "keyword"` (the route default) enumerates all matches in path order.
Duplicate pages do not consume a route's next turn. Without `queries`, one route
uses `context.query`. Route order is caller-chosen, not a semantic ranking.
Interleave does not understand relevance or guarantee better candidate order.
Its benefit is combining query directions and deduplicating the shared pool;
the LLM still selects evidence after reading.

To retain additional task details, a context file is optional. This is equivalent
to the routes in the start example, with extra task information:

```bash
mkdir -p "$search_run"
cat > "$search_run/context.json" <<'JSON'
{
  "task": "Inspect a single-GPU Blackwell GEMM with producer/consumer waits",
  "rationale": "Compare the measured stall with synchronization primitives",
  "queries": [
    {"query": "pipeline", "type": "pattern", "rule": "master"},
    {"query": "mbarrier", "type": "hardware", "rule": "keyword"}
  ]
}
JSON
python3 "$wiki_root/scripts/query.py" --strategy interleave --context "$search_run/context.json" --run-dir "$search_run" --max-chars 12000 --compact
```

Use either CLI routes or `context.queries` in a call; providing both is an error.
A context file without `queries` may accompany CLI routes to supply task details.
Positional query words cannot accompany `--route`, and the single-query `master`
strategy rejects routes. Built-in `interleave` also rejects positional query words
alongside `context.queries`; edit the JSON routes or omit the positional words.
Custom Python strategies may consume both `context.query` and `context.queries`.
This avoids silently ignoring part of the query plan in the built-in policies.

For custom logic, copy the template and edit `retrieve()` before running:

```bash
cp "$wiki_root/scripts/strategy_template.py" "$search_run/strategy.py"
python3 "$wiki_root/scripts/query.py" --strategy "$search_run/strategy.py" --route "pipeline --type pattern --rule master" --route "mbarrier --type hardware" --run-dir "$search_run" --compact
```

The starter calls `index.interleave()`; the LLM may replace it with
`index.master()`, combine rules or select candidates using metadata/body evidence.
There are no fixed route counts or candidate multipliers. Multiple queries can
expose additional evidence but also displace useful pages under a budget.

After reading, use the operator, hardware, dtype/layout and measured bottleneck
to judge relevance. For example, if a pipeline hypothesis is unsupported and
measurements point to tile tails, revise the queries toward tile scheduling;
do not just increase the same weights. Preserve the rationale, phase and prior
query ID in context or the conversation when useful. Python does not choose
queries from natural language or update the strategy by itself.

## Strategy contract

Define `retrieve(index, context)` and return a **list or tuple of page IDs or
wiki-relative paths**. The runner deduplicates while preserving your order.
Unknown pages and pages excluded by CLI filters are errors, not silent omissions.
An invalid strategy produces an error receipt; it does not fall back to a
different retrieval policy.

| API | Behavior |
| --- | --- |
| `index.search(query="", **filters)` | All matching page IDs, without relevance scoring or count truncation; path order is deterministic enumeration |
| `index.master(query="", **filters)` | All matches in master's score/path order, with no count truncation |
| `index.interleave(routes)` | Combine route objects with `rule="keyword"` or `rule="master"`, deduplicating while taking turns |
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
CLI routes become `context["queries"]`, using the same objects as JSON routes.
The log saves both the original route arguments and this normalized context.
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
measure retrieval diversity: several master calls may explore different
mechanisms, and repeated interleave calls may return the same pages.

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
