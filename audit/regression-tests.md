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
