# Validation results

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

## Optional repository import extension

Current discovery includes 189 tests: the historical 158-test baseline plus
repository-import regression tests. The historical results above remain dated
records; extension validation is recorded separately below.

Extension validation: `python3 -m unittest discover -s tests -p 'test_*.py'`
passed with 189 tests. The optional importer also passes the existing offline
freshness and YAML fallback checks, full page validation, index regeneration,
and the size gate (approximately 24.15 MiB / 25 MiB). No GPU execution or
model invocation is performed by this test suite. Public presets require
separate reviewed snapshots; none is enabled by default.

## Repository-only import and overlap review update

Current discovery includes 200 tests: 158 baseline tests plus 42 import/extraction
tests. Added coverage checks skipped gitlinks, rejection of non-Git input, PR
code comparison and context invalidation, same-batch representative selection,
partial pending publication, compact retained versions and example-only scope.
The feature ships review examples without imported corpus pages or a populated
registry. Earlier receipts describe their dated implementation state.

Current discovery now includes 201 tests after adding existing-page variant
retention coverage. The earlier 200-test receipt remains historical.

Final validation passed: 201 tests (158 baseline + 43 import/extraction),
1,041 pages / 989 source IDs, 37 bundles, 14 ledgers, offline freshness, YAML
fallback and size gate (approximately 24.22 MiB / 25 MiB). A real pinned public
Quack review published one selected entry in an isolated temporary wiki, left
one entry pending, returned one query result and passed repeat apply. No import
outputs were added to the feature corpus; no Git commit or GPU execution occurred.

Current discovery includes 202 tests (158 baseline + 44 import/extraction).
The additional regression checks fresh-run reuse of unchanged distinct batch
comparisons and invalidation when comparison evidence changes.

Worktree import validation passed: 202 tests, full corpus validation and offline
freshness. The pinned public review admitted one kernel and left one pending.
A fresh prepare reused the unchanged review and retained one canonical page and
one full representative version. Tracked import outputs were checked out and
generated untracked outputs removed; staged implementation changes were preserved.
No Git commit or GPU execution occurred. Run receipts remain in the ignored cache.


Current discovery includes 205 tests (158 baseline + 47 import/extraction).
The three new regressions cover four reviewed variants selecting one common
representative, an unavailable or blocked representative, and conflicting
reciprocal representative choices. The focused overlap suite passed 15 tests.
The first full run passed 204 checks and failed only this ledger's stale test
count; the discovery receipts were then updated to match the added tests.
No imported corpus artifacts or Git commit were created.

The subsequent full run passed all 205 tests. Full corpus validation passed
1041 files, 989 source IDs, 37 asset bundles and 14 candidate ledgers; offline
freshness also passed. Validation logs remain in the ignored review cache.


Selection-stage update (2026-10-15): discovery includes 214 tests.
Nine new selection tests cover pending-only preparation, source-bound helper
exclusion, confirmed variants with one representative verdict/page, unresolved
coverage, invalid/cyclic/stale relationships, reviewer group splitting and
archived decisions, member-only external matching evidence, unchanged-group
reuse without inherited execution claims, explicit entry-kind filtering, and
real validation/publication in an isolated wiki with changed-selection apply
rejection. Historical per-candidate fixtures explicitly retain legacy mode.
The focused nine-test suite passed. The complete suite is checked separately.
The workspace corpus contains no import outputs and no Git commit was created.

Final selection-stage validation passed: all 214 tests in full discovery and all
nine focused selection tests (including pinned selection/decision reuse after
publication), full validation of 1041 files / 989 source IDs / 37 asset bundles /
14 candidate ledgers, and offline freshness. Logs: `.cache/repo-import/selection-validation/20261015T144413Z/`.
The real publication test used an isolated temporary wiki. No public repository
batch was re-reviewed, no workspace corpus import outputs were added, and no
Git commit was created.

The public selection pilot identified an additional generic-name matching defect.
Discovery now includes 215 tests. A new matching regression excludes unrelated
`kernel` names as tuning evidence while retaining same-body, same-file,
same-basename and actual tuning-suffix comparisons.

Pilot matching correction validation passed: all 215 tests in full discovery.
Unrelated generic `kernel` entry names no longer produce tuning hints. The pinned
Quack pilot retained 25 launchable representative candidates and reduced prepared
comparisons from 386 to 16 (14 batch, 2 existing-page hints), without deleting
saved evidence or changing the source snapshot. Detailed review is recorded in
ignored `.cache/repo-import/selection-pilot/20261015T144526Z/`; this is not a
corpus import or Git commit.

The pinned public Quack pilot completed independent external review and parent
verification: 25 representatives, 24 includes / 1 instrumentation-only exclusion,
zero pending, all snippet (21 source-reported / 3 experimental). Real isolated
index/schema/registry/size checks passed: 1066 files, 990 source IDs, 37 bundles,
14 ledgers, 24.95 MiB / 25 MiB. Outputs remain in ignored staging only. Detailed
receipts: `.cache/repo-import/selection-pilot/20261015T144526Z/`. The generic guide
now explains input-checkout versus wiki-root receipt paths, after this distinction
caused a review validation correction. No source execution or Git commit occurred.

The larger public-repository pilot adds per-scan dependency parsing and hash
caches, with 216 tests now discovered. The regression verifies that cyclic/shared
transitive receipts remain complete, each dependency is parsed once per scan,
and a subsequent source change does not reuse cached hashes.

Large run files use the optional C safe YAML loader when available, retaining
the bundled safe-reader fallback. Discovery now includes 217 tests; the added
reader regression checks matching alias/scalar/default semantics in both modes.

The large-repository performance corrections passed all 217 tests in full
discovery, the actual bundled-reader store tests (python -S), baseline validation
(1041 files / 989 source IDs / 37 bundles / 14 ledgers), and offline freshness.
Receipts: `.cache/repo-import/selection-vllm/20261016T014801Z/validation/`.
The fresh export retains the same 947 source entry locations. Historical parser
version differences changed dependency receipts for 718 entries and removed nine
parse failures; these are not attributed to caching and prior decisions are not
reused in this independent pilot. No workspace corpus import or commit occurred.

The large-repository pilot found repo-only PR comparisons against 203 vLLM
records without code assets. Discovery now includes 218 tests. PR overlap hints
require content, source-path or meaningful name evidence; same-repository records
remain in review context. Regression preserves exact-path, renamed-directory,
normalized-title and implementation-body matches.

Path/name/content-based matching passed all 218 tests in full discovery.
The final source-attribution narrowing was additionally checked by four matching
and 15 overlap tests; registry, PR-only attribution and reciprocal publication
gates remain enforced. Canonical-page repo attribution also needs an operation
clue; unrelated repo references remain available as context rather than forcing
all-pairs overlap decisions.

The large-repository selection-refresh trial now discovers 219 tests. All 10
selection tests and 56 import tests passed, including newly discovered duplicate
relationships, unaffected-review reuse and single-page generation. The first full
219-test run failed only the stale discovery-count receipt (218 other checks
passed); the count is corrected here before repeating full discovery.

The pinned large-repository selection covers 947 definitions: 632 representatives,
140 confirmed members and 175 support definitions. Its external partial checkpoint
reviews 51 representatives (4 include, 47 exclude), leaving 581 untouched and no
genuine evidence-gap deferrals. No full-repository review is claimed. The first
three-representative checkpoint passed isolated index and schema validation
(1045 files), but failed the unchanged size ceiling: 25.58 MiB / 25 MiB. This is
not a publication pass. Receipts and staged outputs remain in ignored cache;
the workspace corpus, registry and indices have not been imported or committed.

The final partial checkpoint also passed real isolated index/schema/registry
validation: 1046 files / 990 source IDs / 37 bundles / 14 ledgers. Its four canonical
pages and 14 selected origin/member records exceed the unchanged size gate:
26.04 MiB / 25 MiB. No gate bypass or actual apply occurred. Detailed receipt:
`.cache/repo-import/selection-vllm/20261016T014801Z/validation/final-checkpoint-staging/validation.yaml`.

Final full discovery passed all 219 tests (181.429s). The 10 focused selection
and 56 import tests also passed. The publication-size failure above remains
unresolved; passing software regression tests does not turn it into a publishable
full-repository review.

Discovery now includes 220 tests. The pinned FlashInfer selection trial exposed
missing CUDA definitions with mutually exclusive conditional brace openings.
The extraction fallback merges equal-depth arms without executing upstream code.
All six focused entry-extraction tests passed, including full conditional source
retention, fingerprint sensitivity and exclusion of adjacent unreviewed entries.
Full discovery passed all 220 tests (172.404s). The complete pinned rescan found
4,356 definitions after adding 15 CUDA entries; every original 4,341 candidate
record, ID and receipt is unchanged. The old inventory/selection were archived,
matching decisions retained and new entries left pending for source review.
No corpus import, Git staging or commit occurred.

Independent scan, selection, detailed-review and incremental-update workflows now discover 252 tests. The expanded checks cover immutable scan receipts, fresh-review isolation, list-only reports, compact publication records, explicit audit packs, content-bound update reuse and separately locked observation/publication state. Full discovery passed all 252 tests (184.988s); the 17 focused update checks, bundled-reader fallback, baseline page/asset/ledger validation and offline freshness also passed. An earlier run overlapped test additions and failed only the stale discovery-count receipt; the complete run above used the corrected receipt.

List-only admission also validates field-specific vocabulary, unique string lists and hardware-tag consistency before counting an implementation. Discovery now includes 253 tests; the added metadata regression is pending final execution.

Public-source acceptance exposed CUTLASS_GLOBAL kernel declarations omitted by literal __global__ scanning. Extraction now recognizes both markers and ignores qualifier-only macro definitions, with exact selected-snippet checks. A separate scan coverage contract blocks automatic reuse/deletion assertions across extraction changes while retaining stable per-entry receipts. Discovery now includes 256 tests; final execution is pending.

A multi-implementation PR now projects every linked canonical page into overlap review. Matching PR code can receive its own canonical page only after all linked canonical implementations have valid distinct comparisons; unknown/same relations and changed receipts still block it. Discovery now includes 257 tests; final execution is pending.

Changed skipped dependency pins conservatively invalidate reuse without fetching or scanning their contents. A real Gitlink fixture verifies this boundary. Discovery now includes 258 tests; final execution is pending.

Source-discovered overlap inputs now have receipt validation, mandatory refreshed hints, audit-pack replay and conservative update carry-forward. Bounded same-file CUDA declaration-prefix macros expose actual invocation bodies without scanning macro aliases as kernels. Discovery now includes 275 tests; final execution is pending.

Final implementation validation passed: all 275 discovered tests (200.313s), full baseline validation (1,041 files, 37 asset bundles, 14 ledgers), offline freshness and the unchanged 25 MiB size gate (24.44 MiB). Real pinned-source update acceptance reused 25 unchanged selection records and 23 valid decisions; a public parent-to-current update found 8 unchanged and 17 dependency-changed entries and reused no decisions in fresh-review mode. A refreshed two-implementation pilot passed isolated index generation, schema/registry validation and size checks (24.55 MiB). No real corpus publication occurred. Full public review materials remain in ignored external run storage. These checks validate the workflow; incomplete large-repository reviews are not counted as completed.

Review-material refresh now also archives the actual prior guide, instructions and representative targets alongside decisions and request bindings. The existing regrouping regression verifies that prompt changes preserve the original audit input while refreshing the current guide. The subsequent full run passed all 275 tests (189.038s).

Same-run source-discovered comparisons: 57 focused tests passed. Full discovery executed 281 tests (190.320s), with one failure because the regression ledger still recorded the old test count; the corrected full rerun passed all 281 tests (197.242s). Baseline validation and offline freshness passed after the implementation change. Actual preparation of three public-source candidate pairs generated six reciprocal mandatory comparisons, retained source/decision/prompt history and did not publish any kernels. All six new comparisons were resolved with valid source receipts, and parent verification plus partial decision validation passed. The remaining large-repository reviews are still incomplete.

Existing pages that explicitly cover different implementation branches can retain multiple scoped source relationships without creating duplicate pages. Four added regressions cover distinct literal target-page anchors, mandatory reciprocal comparisons, missing/forged/stale/overlapping scopes, one registry identity, preserved prose and evidence grades, list-only counts, and saved-decision replay. Discovery now includes 285 tests. The first full run executed all 285 checks (197.915s); its only failure was the stale discovery-count receipt in these ledgers. The corrected full gate is being rerun. The actual reviewed public multi-branch case admits 302 definitions into 226 effective groups, leaving only two source-defect decisions and one launch-reachability entry pending. No real wiki publication occurred.

The corrected complete discovery gate passed all 285 tests (202.323s). All 113 importer-focused checks, final four shared-page checks, baseline page/asset/ledger validation and offline freshness passed. Existing multi-branch source relationships passed actual list-only report and full audit-pack export; the real corpus remains unchanged.

An actual target-page edit reproduced stale source-attribution reuse even when branch anchors remained. Reuse now requires the registered existing-page content receipt to match the current page; externally changed prose cannot silently refresh old evidence. A new regression checks both automatic preparation and saved-manifest replay. Discovery now includes 286 tests; final gates are being run.

Final validation of target-page reuse passed all 286 tests (204.059s), all 42 focused admission/overlap/preset/update checks, baseline validation (1,041 files, 37 asset bundles, 14 ledgers), and offline freshness. Unchanged published source relationships remain reusable; the reproduced external-page edit now leaves both prior attributions pending. No real corpus content was changed.

Same-file object-like CUDA qualifier aliases now bind actual named functions, source lines and selected bodies instead of capturing a following struct or host helper. Four new regressions cover real functions, prototypes, unused/unsupported aliases, balanced attributes, local redefine/undef boundaries and exact snippets. Discovery includes 290 tests. The first complete run executed all 290 tests (209.597s); its only failure was the stale discovery-count receipt in these ledgers. The corrected full gate is being rerun. An isolated rescan of all 14 pinned public inputs preserved exact candidate records in 13 repositories; one input removed 13 pending false bindings and added 35 actual function definitions. Its immutable replacement scan keeps added definitions pending and retains only explicitly revalidated unchanged selection evidence. Baseline schema/assets/ledgers and offline freshness pass; no real corpus publication occurred.

The corrected complete gate passed all 290 tests (207.013s). All 41 focused scanner/snapshot/update tests passed (11.419s); baseline validation and offline freshness passed. The scan-coverage contract advances independently of stable entry fingerprints, blocking automatic reuse or deletion assertions across the coverage change. Real audit materials stay outside the commit and no kernel pages were published.

Unchanged selection refresh now batch-validates reusable decisions once, retaining full source/member/relationship checks and isolating invalid records through the existing individual validation fallback. Two new regressions cover one-pass reuse and independent valid-record retention after stale metadata, evidence or comparisons. Discovery includes 292 tests. All 116 importer checks passed (146.353s); the first complete run executed 292 tests (222.604s), with only the discovery-receipt check failing because these ledgers still listed 290. The corrected complete gate is being rerun. Actual unchanged public-source preparation preserved all 189 reviewed decisions exactly in 30.119s; baseline validation and offline freshness passed. No corpus content was published.

The corrected complete discovery gate passed all 292 tests (213.274s). All 116 importer checks, baseline page/asset/ledger validation and offline freshness passed. Actual unchanged-source selection preparation reused all 189 original reviewed decisions without altering their decision content; all full validation checks remain enabled. Real audit runs remain ignored and no kernel pages or populated registry are included.

Evidence-integrity update (2026-10-18): discovery includes 300 tests. Eight new regressions require actual existing files and nonempty 64-hex SHA-256 evidence for source decisions, helper selection, batch/page comparisons, existing-page attribution, reproducibility, discovered comparisons and update reuse. The focused eight-test suite and all 300 discovery tests passed (219.867s); baseline validation and offline freshness also passed. Real audit materials remain outside the committed corpus.

Balanced CUDA signature update (2026-10-18): discovery includes 304 tests. Four new regressions cover braced default parameters and nested constructors, same-file qualifier aliases, complete selected snippets with name/line binding, and conditional signatures plus unknown launch-attribute macros. The first complete 304-test run (264.386s) passed all implementation checks and failed only this documentation test-count receipt, which still listed 300. The corrected gate is being rerun. Actual fixed-commit rescans of all 14 public inputs preserve exact records in 12 repositories; two get immutable replacement scans with newly observed definitions pending source review. No real corpus publication occurred.

The corrected complete gate passed all 304 tests (236.440s). Baseline validation and offline freshness passed. Actual product rescans retain exact records in 12 public inputs and yield 10,027 raw definitions across all 14 pinned inputs; immutable replacements preserve previous scan receipts. A real unchanged-content v4-to-v5 update keeps all 25 observations but reuses zero selection decisions and disables deletion assertions. Public review is still in progress; raw definitions are not effective kernel counts. No corpus content was published.

Shared PR attribution update (2026-10-18): discovery includes 310 tests. Six new regressions cover a shared implementation cited by two existing pages, explicit canonical/source anchors, complete PR/page review, invalid anchors, unchanged registered attribution reuse and fresh review after an external page change. The gate preserves one import target and existing page prose; shared PR references alone do not establish equivalence. Real corpus publication remains disabled during acceptance.

The complete shared-PR attribution gate passed all 310 tests (233.770s). Baseline validation and offline freshness passed. The real source-reviewed FC1 port resolves to one existing canonical page with explicit evidence for both previously citing pages; acceptance generated decisions and reports only, without publishing corpus changes.

Generic module overlap update (2026-10-19): discovery includes 311 tests. The new
regression rejects title/basename hints for generic module names with leading
underscores, while preserving exact paths, retained upstream paths, source/body
hashes and same-file batch hints. All 311 tests passed (306.008s); baseline
validation and offline freshness passed. Real audit materials remain ignored,
and no corpus content was published.


Portable review archive update (2026-10-28): discovery includes 319 tests. Eight
new archive regressions cover cache-independent relocation, lossless YAML and
compression, read-only preset rejection, unchanged holds and confidence,
source/evidence receipt tampering, output isolation, and unsafe compressed paths
or links. The eight focused checks pass with both installed PyYAML and the
bundled fallback. Independent relocation of the real compressed batch verifies
all 14 repositories, 1,980 effective groups, 177 detailed-review defers and 34
initial-selection pending entries. Wiki validation (1,041 files, 37 asset bundles,
14 ledgers), offline freshness and the 40 MiB size gate pass; the working tree
is 36.41 MiB including the complete 11.84 MiB compressed review records. Original
run history is unchanged. No real Wiki corpus publication occurred.

Final complete discovery gate: all 319 tests passed (217.753s), including the two real-validator publication tests in isolated fixtures. Archive relocation, baseline validation, offline freshness and the size gate also pass. No real Wiki corpus or active import registry was changed.


Repeated real-repository acceptance update (2026-10-28): discovery includes 325 tests.
Five new regressions cover reciprocal batch distinctions becoming registry
matches after publication, unchanged-page receipts, changed source/page rejection,
absent prior comparisons, and unchanged versus edited existing-page evidence.
All five focused tests pass. See the final acceptance receipt below.

The initial real pinned Quack acceptance in an isolated Wiki produced 23 canonical pages from 23 representatives and one variant
member, excluding one helper. Same-run apply was byte-idempotent; two preset
reimports retained 23 current versions and the second preserved every page byte.
An unchanged-commit update retained 25 observations and reported 23 effective
groups with zero pending after select. All publication validators and kernel
queries passed. Original review records and main Wiki contents were unchanged;
confidence/reproducibility remained 20 source-reported, 3 experimental, and all
23 snippet. No GPU execution or publication into the main Wiki occurred.


Complete public-repository publication acceptance (2026-10-28): all 14 reviewed
fixed snapshots passed scan, selection, external-review validation, report,
build and apply in one separate persistent Wiki. Source-bound selection
refinements preserved the exact admitted definition set: 4,440 definitions,
1,988 per-repository effective groups and 1,949 canonical pages (1,941 newly
managed and eight existing pages retained). The 177 detailed-review defers and
34 selection-pending entries remained unadmitted. These refined acceptance
reviews are local audit material; the checked-in read-only archive is unchanged.

Every plan passed same-run reapplication, and all 14 completed plans were then
reapplied after the shared publication. The complete Wiki working-tree corpus
remained byte-identical. Final Wiki validation, offline freshness, size and
kernel-query coverage checks passed. Development regression passed all 325 tests
in 216.762s, including a multi-branch PR attribution regression with positive and
negative cases. FlashInfer's 3,490 matcher hints and their IDs were identical
before and after the ownership-set optimization. Two upstream WGMMA figure
anchors were verified against the official PTX ISA 9.4 document and added to the
offline link contract.

The development Wiki corpus, active registry and original review archive remain
unchanged. Generated acceptance pages, source checkouts, local orchestration
tools and run histories are outside the development checkout. No GPU execution
or performance verification was performed.
