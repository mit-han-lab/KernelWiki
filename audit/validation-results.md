# Validation results

## Explicit strategy selection validation

Follow-up review reproduced and fixed a strategy-loader bug: dataclasses with
postponed annotations could not resolve their module. Strategies now execute
with a registered module, restoring the prior registry state on success or failure.

| Check | Result |
| --- | --- |
| Full unittest discovery | pass — 188 tests in 55.648 seconds |
| Focused retrieval suite | pass — 30 tests |
| Explicit master against original master | pass — 15 CLI path lists with the same limit; identical scores for 11 queries over 1,041 pages |
| Offline named/custom strategy smoke | pass — master, interleave and a dataclass-based custom strategy, each under 2,600 characters; 3 linked page reads |
| Page output against master | pass — 5 full/body/frontmatter/source-preview/code comparisons |
| Optimization-local paths | pass — relative strategy/context/run paths resolve in the caller's workspace; temporary-file allocation can be unavailable |
| Repository validation | pass — 1,041 pages, 989 source IDs, 37 bundles, 14 ledgers |
| Bundled YAML fallback | pass |
| Repository size | pass — 24.13 MiB, 1,595 files, 422 artifacts; ceiling 25 MiB |

Named `master`, `interleave` and generated Python files share one execution and
logging path. Every query requires `--strategy` and `--run-dir`; all candidate
count caps are explicit. Master scoring is preserved as an option. The examples
place generated files under the optimization workspace's writable `tmp/`.

The skill now asks the calling LLM to choose and revise queries, rules, candidate
selection and reading based on task evidence. The scripts make no model calls.
No GPU ablation or retrieval-quality claim is made. Validation uses public corpus
content and `.worklog/strategy-validation/` / `.worklog/strategy-review/`, with no
private task reads or writes.
The external skill validator still rejects the pre-existing `argument-hint` key;
frontmatter is byte-identical to master. Repository and documented-command checks
pass. Earlier validation below describes the preceding interface.

## Prior strategy retrieval validation

| Check | Result |
| --- | --- |
| `python3 -m unittest discover -s tests -p 'test_*.py'` | pass — 180 tests in 34.810 seconds |
| Strategy, output-budget and reading-receipt regression tests | pass — 22 tests |
| `python3 scripts/validate.py` | pass — 1,041 pages, 989 source IDs, 37 bundles and 14 candidate ledgers |
| `bash tests/check_yaml_fallback.sh` | pass — offline bundled YAML |
| Real-corpus strategy → bounded candidates → linked page read | pass under `python3 -S` — 8 proposals, 5 emitted, 3 omitted; 2,417 characters within a 2,600-character budget; 2 linked read receipts including source previews and full/truncated code |
| Built-in keyword/filter candidate ordering against `master:scripts/query.py` | pass — 15 CLI path lists identical; scores and OR recall identical for 11 queries across 1,041 pages |
| `get_page.py` output without logging against master | pass — 5 full/body/frontmatter/source-preview/code output comparisons match exactly |
| `python3 scripts/repo_size_check.py` | pass — 24.11 MiB, 1,593 files and 422 artifacts, below the 25 MiB ceiling |
| Python compilation and `git diff --check` | pass |

The runner executes an independently generated `retrieve(index, context)`
strategy. Index searches expose all matches without automatic scoring or a count
cutoff. The strategy chooses proposals and order; the runner deduplicates and
applies an optional candidate limit plus a character budget. Without a strategy,
ordinary keyword/alias/filter search retains master's ordering and default limit.

Each strategy query saves the executed source, context, engine snapshot, corpus
hash, searches, proposed/emitted/omitted candidates and output. Strategy body
reads and content returned by the page reader have separate records. Page-reading
receipts preserve source previews, metadata-only reads and truncated artifacts as
such; they are not evidence of model attention. Direct external reads are not
intercepted. The scripts make no model API calls.

The corpus (`wiki`, `sources`, `artifacts`, `data`, `queries`) and `grep_wiki.py`
remain unchanged from master. The existing page-reader behavior is extended with
optional reading receipts. Generated task context and run artifacts belong outside
the shared wiki. Only public corpus content and temporary validation runs were used; no private task files were read or modified.

The external skill-creator validator was rerun and rejects the pre-existing
`argument-hint` frontmatter key. The complete frontmatter is unchanged from master;
repository validation and documented-command tests pass. Retrieval and
logging checks do not measure LLM selection quality or kernel performance. No
GPU ablation was run for this change, and the historical upstream-evidence audit
below was not rerun.

## Historical corpus audit

Environment: isolated `/tmp/kernelwiki-factual-audit-venv-20260818` with
PyYAML 6.0.3. Final post-quorum rerun after clean Rounds 31 and 32:
2026-08-20 UTC.

## Baseline

The starting revision validated structurally, but `verify_verbatim.py` found
two semantic patch mismatches caused by upstream Git object-ID abbreviation
width, and unittest discovery failed with exit 5 because zero tests existed.
The initial size was 24.72 MiB / 25 MiB and core manifests contained 252 PRs.

## Final post-quorum matrix

| Command | Exit | Verdict | Exact result |
| --- | ---: | --- | --- |
| `python3 scripts/validate.py` | 0 | pass | 1,041 files; 989 source IDs; 37 verbatim bundles; 14 candidate ledgers; all valid |
| `python3 scripts/check_version_freshness.py` | 0 | pass | 0 warnings; 7 informational old-release messages |
| `python3 scripts/verify_verbatim.py --strict` | 0 | pass | 37 bundles; all verbatim/upstream-patch assets match upstream |
| `python3 scripts/check_dod_fixtures.py` | 0 | pass | contract v3; exact five-id roster; 3 active and 2 retired contracts; all assets/tombstones valid |
| `python3 scripts/verify_pr_architecture_upstream.py` | 0 | pass | 19 deterministic high-risk PRs re-derived live, including all 6 capped PRs, real-device/host-only `.cu` controls, and canonical receipt hashing |
| `python3 scripts/verify_core_prs.py` | 0 | pass | all 3 manifests byte-consistent; checksum `cde1aab41b70…`; 110 core PRs |
| `python3 scripts/repo_size_check.py` | 0 | pass | 24.68 MiB across 1,601 reviewable files; 422 artifact files; ceiling 25 MiB |
| `python3 scripts/generate-indices.py` | 0 | pass | 1,041 pages; all 7 query indices regenerated |
| `python3 -m unittest discover -s tests -p 'test_*.py'` | 0 | pass | 158 tests |
| `bash tests/check_freshness_offline.sh` | 0 | pass | freshness checker remains network-free |
| `git diff --check` | 0 | pass | no whitespace or conflict-marker errors |

Additional mandatory gates:

- numeric scanner: 3,812 candidates (1,121 supported, 2 corrected, 1,946 removed, 743 non-material), zero unresolved;
- architecture inventory: zero unsupported assignments, zero unexplained
  rewrites, zero unjustified unknowns; all 944 retained pages match the
  frozen-evidence receipt, and the live high-risk sample passes;
- scope: 2,692/2,692 baseline source PRs terminal; 944 retained and 1,748
  removed;
- narrative source-PR backlink scan: zero absent IDs;
- coverage: 1,091/1,091 files terminal, 100.0%, 36,457 examined claim units.

This complete matrix is rerun after any accepted independent-review finding and
again after the clean-round quorum.

Rounds 31 and 32 are consecutive exact-model clean reviews. The independent
review quorum is complete at 2/2, and this matrix is the required final rerun
after quorum.
