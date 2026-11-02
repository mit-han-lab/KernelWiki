# Mandatory regression-test ledger

## Baseline

`python3 -m unittest discover -s tests -p 'test_*.py'` failed with exit 5
because the starting revision discovered zero tests. Architecture, generated
body, scope, metadata, query, and provenance guards were absent.

## Current full discovery gate — 2026-08-19 UTC

Command: `python3 -m unittest discover -s tests -p 'test_*.py'`

Result: **pass — 158 tests**.

Coverage includes:

- exact/family/unknown architecture classification; documented B200/GB200,
  B300/GB300, H100/H200/H800/H20/GH200, and A100 mappings; camel-case tokens;
  multiplier-prefixed products; portable CuTe atom names; generated CI-bot
  boilerplate; numeric architecture guards; underscore helpers; target comments;
  SM75/Turing and valid SM87/SM88 targets; rejection of nonexistent lower-SM
  suffix spellings; commented-out-code and range-boundary rejection; mixed
  exact/family evidence; the required seed cases;
- exact/family/unknown query hierarchy, exact-query exclusion disclosure,
  generated-index set equality, hand-authored family pages without a generated
  disposition, mixed pages appearing in both their exact and other-family
  lanes, family-caption precision, Blackwell canonical variants, and the
  documented exclusion of undispositioned non-PR empty lists from the
  validated source-PR unknown lane;
- free-text architecture aliases preserving `blackwell` as a family and
  resolving B200/GB200 to `sm100` and B300/GB300 to `sm103`, with validator
  rejection of alias collisions and product-map drift;
- title-subject EPLB/DeepEP/DualPipe exclusions, incidental body/test mentions,
  cross-device implementation paths, mixed distributed/local PRs, tests,
  configuration, license-only changes, host Python, and positive CUDA/PTX/DSL
  implementations;
- full qualifying-path preservation and metadata derivation after the former
  eighth-path cap; positive and negative auto-tagging including bare `fused`;
- deterministic upstream-only bodies; hashes; CRLF/trailing-space/truncation/
  conflict-marker canonicalization; injected prose rejection;
- refresh-time shared classifier behavior and defer-on-inaccessible-evidence;
- strong scan-kernel terms versus ordinary “linear scan” prose;
- release-snapshot upstream receipts that are recent, present, parseable, not
  older than any recorded release, and exactly equal to the committed release
  list (including missing/unreceipted/date-mismatch negative cases);
- the frozen 944-row architecture receipt plus the live GitHub sample gate,
  including negative controls in which an internally consistent fabricated
  page assignment must fail;
- upstream-patch canonicalization that ignores only Git object-ID abbreviation
  width while preserving hunk and mode discrimination;
- schema snippet requirements and positive/negative substantive-code checks;
- version-cutoff handling for old and newly obtained evidence.
- Definition-of-Done contract-v3 identity reconciliation, including failure
  when an active fixture and its colocated roster row are deleted together
  without a roster-preserving retirement tombstone.
- authoritative PR changed-file totals separated from REST-enumerated,
  complete-evidence, and displayed-path counts; complete pull-diff parsing;
  validator failures for inconsistent or hash-drifted receipts; a tail-only
  device-evidence fixture; and live checks for all six capped TensorRT-LLM PRs.
- added or modified `.cu` hunks require either an independent device construct
  or an immutable complete-file receipt whose content and policy-pattern
  digests validate; bare tile/warp identifiers and host launch syntax are
  negative cases alongside host-only, positive-device, stale-pattern, and live
  positive/host/tile-only controls.
- strict `.cu` scanning ignores comments/strings and rejects host
  `config.gridDim`/`config.blockDim` fields plus host-only `GemmType`, while
  retaining unqualified device builtin use and a real immutable-file device
  implementation;
- exact `cpu/` components never qualify for the GPU corpus, and weak Python
  tile/warp tuning text outside an explicit DSL path requires a content-hashed,
  current-pattern full-file Triton/CuTe DSL/TileLang verdict; comments, host
  shape arithmetic, negative/stale receipts, and positive DSL receipts are all
  covered;
- `upstream_files_sha256` ignores historical receipt blocks that the current
  policy does not require, preventing cache enrichment history from changing
  clean live derivation.
- the tcgen05 tutorial source and both propagated wiki references use the
  author-supported identity `Thien Tran (gau-nernst)` and reject the invented
  de-slugified form “Gau Nernst.”
- the extended and condensed FlashAttention-4 performance examples are
  semantically identical and paper-backed at 1,613 TFLOP/s without an invented
  maximizing shape; the skill's technique count equals both the generated
  index rows and on-disk technique pages.
- shipped navigation derives the 942-merged/2-closed PR status distribution,
  the single `verbatim` artifact mode, absence of contest `submissions`, and
  NVFP4/GEMV case-study facts from the current corpus; the FlashAttention §5
  label and source ordering are evidence-bounded, and the tutorial's 35%
  receipt reproduces v2b→v3 arithmetic.
- shipped ID link labels resolve to current frontmatter IDs, source-directory
  category descriptions equal the category sets on disk, and any arXiv URL is
  rejected unless it uses the `source-doc`/`paper` contract.
- the PTX source map pins its three corrected instruction-section locators and
  rejects the stale numbers; every PTX fragment in curated wiki/source-doc
  prose must also belong to the live-verified PTX ISA 9.3 anchor set;
- Blackwell-relevance guarantees derive Hopper and Blackwell family tokens plus
  every controlled exact `a`/`f` target from the shared policy, prove every
  Hopper-only wiki page is justified, preserve source-page exemption, exercise
  every derived value adversarially, and bind all four shipped descriptions to
  the executable validator scope;
- merged PRs require full 40-hex merge SHAs, non-merged PRs omit the field,
  and the generator both omits ephemeral test-merge objects for closed PRs and
  fails if a genuinely merged payload lacks its merge commit;
- the FlashAttention-4 source credits the complete six-author byline, while
  source body and propagated navigation avoid sole-authorship language.
- the Qwen3-Next source records the NVIDIA article's named author and rejects
  the unsupported organization-pair credit;
- the numeric ledger's 3,812 disposition rows and four status counts are
  mechanically recomputed and required to agree with all three current audit
  summaries;
- the complete concrete source-PR example in the schema reference must equal
  the named `pr-cutlass-2472` frontmatter, preventing mixed real/template data.
- the Modular source preserves its four-person byline, the NSA paper preserves
  first-author-plus-institutions attribution with DeepSeek-led prose, and every
  concrete shipped `grep_wiki.py`, `query.py`, and `get_page.py` command exits
  successfully with nonempty output;
- the CUTLASS 4.6.2/4.6.1 table preserves official rendered changelog labels
  and the checked-in/rendered distinction, while the Hugging Face community
  source preserves `Konstantin (apsys)` as its rendered author identity;
- current audit receipts derive the discovered test-method count and retained
  provenance-bundle count from disk, preventing same-day test or bundle totals
  from silently diverging across regression, validation, evidence, and strict
  verification ledgers.
- the current size-budget receipt is bound to the generated core-PR total and
  checksum, the 6,000-file limit is scoped to `artifacts/` in both executable
  and historical prose, and the JAX tutorial uses its rendered author credit
  without an unsupported corporate affiliation.
- every propagated NVFP4 scale description preserves the organizer's
  E4M3FNUZ-task-prose versus `torch.float8_e4m3fn`-reference-code discrepancy
  and rejects either signed encoding as a synonym for PTX `UE4M3`.
- both GPU Mode NVFP4 problem tuples and their three wiki consumers state that
  the organizer exposes per-16 scale tensors and permuted views, not a separate
  tensor-level/global scale operand.
- the reward-hacking post uses Natalia Kokoromyti's rendered participant byline
  and scopes GPU Mode as hosting publisher rather than co-author in both the
  source record and contest cross-reference.
- every shipped source/research consumer of that post-mortem is discovered and
  rejects organizer-authorship phrasing; every sentence naming Natalia
  Kokoromyti also rejects unsupported gendered third-person pronouns, requires
  the neutral source wording, and detects an adversarial gendered-form mutation;
- inclusion-policy AC-11 rejects the obsolete Triton no-tcgen05/TMEM phrase in
  raw YAML comments as well as parsed scalars, and Triton worked-example
  navigation points to the current opening-paragraph boundary rather than a
  removed subsection.

Corpus regeneration applied the shared policy to all 2,692 comparison-base
PRs: 944 retained, 1,748 removed, 33 family-only Blackwell, and 328 visible
unknown architecture dispositions. Candidate and scope row sets match exactly.

The generated-manifest verifier additionally regenerates `core-prs.yaml`,
`cute-dsl-universe.yaml`, and `triton-universe.yaml` in memory and compares
bytes. Current result: **pass**, 110 core PRs, checksum `cde1aab41b70…`.

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

The large-repository selection-refresh trial now discovers 219 tests. The new
regression covers an overlap discovered during detailed review: revise selection,
renew the affected representative, preserve an unrelated exclusion, and generate
one canonical page. All 10 focused selection tests and 56 import tests passed.
Review reuse now computes hints once per immutable refresh and skips deferred
records, which the existing reuse rules never accepted. The default decision
validation path still recomputes its hints.

The first full 219-test run passed 218 checks and failed the discovery-receipt
check because these two ledgers still listed 218. This entry corrects that count;
the complete discovery gate is rerun after the receipt update.

Final full discovery passed all 219 tests after the receipt correction (181.429s).
The real large-repository run corrected its selection to 632 representatives and
140 members, retaining unaffected verdicts. Partial generated content passed
isolated index/schema/registry checks but failed the existing 25 MiB size gate;
no workspace publication or Git commit occurred.

Discovery now includes 220 tests. The pinned FlashInfer trial exposed CUDA
definitions omitted when mutually exclusive preprocessor arms open corresponding
braces. A conservative fallback merges equal-depth arms when flat counting fails;
existing successful extents remain unchanged. The regression checks complete
conditional source retention, body-hash changes and separation of a following
unreviewed entry. All six entry-extraction tests passed. Full discovery passed
all 220 tests (172.404s). Rescanning the pinned public snapshot added 15 CUDA
definitions; all original 4,341 candidate records, IDs and receipts are identical.
Existing selection decisions were retained only after this complete comparison.

Independent scan, selection, detailed-review and incremental-update workflows now discover 252 tests. The expanded checks cover immutable scan receipts, fresh-review isolation, list-only reports, compact publication records, explicit audit packs, content-bound update reuse and separately locked observation/publication state. Full discovery passed all 252 tests (184.988s); the 17 focused update checks, bundled-reader fallback, baseline page/asset/ledger validation and offline freshness also passed. An earlier run overlapped test additions and failed only the stale discovery-count receipt; the complete run above used the corrected receipt.

List-only admission also validates field-specific vocabulary, unique string lists and hardware-tag consistency before counting an implementation. Discovery now includes 253 tests; the added metadata regression is pending final execution.

Public-source acceptance exposed CUTLASS_GLOBAL kernel declarations omitted by literal __global__ scanning. Extraction now recognizes both markers and ignores qualifier-only macro definitions, with exact selected-snippet checks. A separate scan coverage contract blocks automatic reuse/deletion assertions across extraction changes while retaining stable per-entry receipts. Discovery now includes 256 tests; final execution is pending.

A multi-implementation PR now projects every linked canonical page into overlap review. Matching PR code can receive its own canonical page only after all linked canonical implementations have valid distinct comparisons; unknown/same relations and changed receipts still block it. Discovery now includes 257 tests; final execution is pending.

Changed skipped dependency pins conservatively invalidate reuse without fetching or scanning their contents. A real Gitlink fixture verifies this boundary. Discovery now includes 258 tests; final execution is pending.

Source-discovered overlap inputs now have receipt validation, mandatory refreshed hints, audit-pack replay and conservative update carry-forward. Bounded same-file CUDA declaration-prefix macros expose actual invocation bodies without scanning macro aliases as kernels. Discovery now includes 275 tests; final execution is pending.

Final implementation validation passed: all 275 discovered tests (200.313s), full baseline validation (1,041 files, 37 asset bundles, 14 ledgers), offline freshness and the unchanged 25 MiB size gate (24.44 MiB). Real pinned-source update acceptance reused 25 unchanged selection records and 23 valid decisions; a public parent-to-current update found 8 unchanged and 17 dependency-changed entries and reused no decisions in fresh-review mode. A refreshed two-implementation pilot passed isolated index generation, schema/registry validation and size checks (24.55 MiB). No real corpus publication occurred. Full public review materials remain in ignored external run storage. These checks validate the workflow; incomplete large-repository reviews are not counted as completed.

Review-material refresh now also archives the actual prior guide, instructions and representative targets alongside decisions and request bindings. The existing regrouping regression verifies that prompt changes preserve the original audit input while refreshing the current guide. The subsequent full run passed all 275 tests (189.038s).

Source-discovered comparisons now also support two same-run candidates, with reciprocal mandatory hints, exact definition receipts, member projection and conservative updates of both endpoints. Six added regressions cover unresolved admission, tampering, duplicate/reversed pairs, contradictory reciprocal outcomes, initial groups, unchanged reuse, changed endpoints, possible renames and missing-target history. Discovery includes 281 tests. All 57 focused checks passed. The first full run executed 281 tests (190.320s); its only failure was this ledger's stale discovery-count receipt. The corrected full rerun passed all 281 tests (197.242s). Actual public-source preparation persisted three new candidate pairs and produced all six reciprocal mandatory hints without changing selection roles or publishing corpus content. The reviewer resolved all six with current source receipts; parent verification and partial decision validation passed, without claiming that the remaining repository review was complete.

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
