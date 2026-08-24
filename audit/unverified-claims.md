# Unverified and disputed claims

## Open material claims

None. The numeric ledger contains zero unresolved material candidates, every
baseline source-PR scope and architecture decision is terminal, and no current
page states the original pilot uncertainties as fact.

## Resolved disputes

- TMEM is described with PTX's CTA-visible 512-column by 128-lane by 32-bit-cell
  model. A physical “bytes per SM” interpretation was not restored because the
  ISA is not evidence for that implementation claim.
- B300/GB300 are mapped to compute capability 10.3 only after locating them in
  NVIDIA's current CUDA GPU list; the classifier records that official URL.
- The exact file-by-file coverage denominator follows section 5 of the supplied
  audit plan. Generated queries, artifact provenance, and audit code are
  separately checked by set-equality, online provenance, and regression gates;
  they are not silently represented as part of the 1,091-file denominator.
  Root `CLAUDE.md` and `SKILL.md` are likewise outside that fixed denominator;
  focused tests now bind their factual examples and current counts to the
  in-scope schema, generated index, and on-disk pages.
- Claude round 1 is not quorum-eligible because it delegated work to Haiku,
  even though all of its findings were independently investigated.
- Claude round 2 is not quorum-eligible because its outer `modelUsage` includes
  Haiku despite the reviewer reporting a single-process Opus review; all ten
  findings were nevertheless independently investigated.
- Claude round 3 is exact-model valid (outer `modelUsage` contains only
  `claude-opus-5`) but non-clean. Its independent 2,692-page re-derivation
  reproduced every scope and architecture decision; all returned content and
  durability findings were nevertheless corrected before the next review.
- Claude round 4 is exact-model valid and non-clean. Twelve findings were
  accepted or accepted with a narrower evidence boundary; the proposed removal
  of `sm88` was rejected against NVIDIA's explicit NVCC target list. All
  accepted findings were corrected and revalidated before the next review.
- Claude round 5 is exact-model valid and non-clean. Its GB300/Blackwell alias
  finding and DoD-roster concern were confirmed and repaired. Its two omitted
  non-PR empty-architecture pages exposed an ambiguous index caption, not a
  mismatch under the validated-unknown query contract; the scope is now
  explicit and set-equality tested. Dead links copied inside the pinned CUTLASS
  verbatim reproduction remain attributed upstream text, not local claims.
- Claude round 6 is exact-model valid and non-clean. All six findings were
  confirmed and repaired: capped GitHub file listings now have honest totals
  and reconstruction receipts, FP4 scale overhead is arithmetically correct,
  README and DoD version statements match executable data, and PFLOPS variants
  enter the terminal numeric ledger. The clean-round streak remains zero.
- Claude round 7 is exact-model valid and non-clean. It correctly showed that
  the first reconstruction applied a narrower tail filter than the executable
  policy, falsely removing TensorRT-LLM PRs 12470 and 14291; both are restored
  from exact-count complete pull diffs. Its 5-versus-6 line count, NSA locator,
  and capped-removal recurrence findings are also corrected. The clean-round
  streak remains zero.
- Claude round 8 is exact-model valid and non-clean. Its four factual errors
  were confirmed: NSA's paper does not state BF16 for the 9.0x result, two
  current audit totals were stale, and the FlashAttention-4 paper title was
  wrong. Its DoD joint-deletion escape was also reproduced and closed by an
  independently pinned contract-v3 roster. The clean-round streak remains zero.
- Claude round 9 is exact-model valid and non-clean. Its modified-`.cu` finding
  was confirmed and broadened from a verified 15-page lower bound to an
  exhaustive 60-page correction. All 761 inconclusive modified CUDA paths now
  carry immutable PR-head content receipts, and both stored-receipt and live
  positive/negative controls pass. The clean-round streak remains zero.
- Claude round 10 is exact-model valid and non-clean. Its weak-token finding
  was confirmed: bare tile/warp variables made a host-only launcher satisfy the
  `.cu` device predicate. The correction also excludes host launch syntax as
  standalone proof, re-fetches 965 inconclusive files at immutable PR heads,
  and removes the 16 remaining false-positive FlashInfer pages. The live gate
  now includes FlashInfer PR 1398 as a tile-only negative control. The
  clean-round streak remains zero.
- Claude round 11 is not quorum-eligible because its outer `modelUsage`
  includes Haiku despite the inner report claiming a single-process Opus
  review. Its four findings were independently reproduced and corrected:
  host launch fields/type names no longer prove `.cu` device code, evidence
  hashing is canonical under the current policy, CPU paths are explicitly
  outside GPU scope, and ambiguous Python tuning hunks require immutable
  full-file DSL proof. The complete rerun removes 45 false-positive pages; the
  clean-round streak remains zero.
- Claude round 12 returned an inner clean report with complete coverage, but
  its outer usage record included a Haiku session-title request automatically
  made by Claude Code. The exact-model gate has no infrastructure exception,
  so the report receives no clean credit. A diagnostic identified and then
  suppressed that nonessential request.
- Claude round 13 was genuinely Opus-only, but disabling compaction at the
  standard 200K window caused `Prompt is too long` before a verdict. It is not
  a completed review and receives no clean credit. A diagnostic then verified
  the exact `claude-opus-5[1m]` extended-context variant with an Opus-only
  outer record.
- Claude round 14 is exact-model valid and non-clean. Its sole finding was
  independently confirmed: `gau-nernst` had been turned into the unsupported
  personal name “Gau Nernst.” The source and two wiki references now use the
  author's site- and account-supported identity, `Thien Tran (gau-nernst)`,
  with a focused recurrence test. The clean-round streak remains zero.
- Claude round 15 was stopped before verdict after the debug trace exposed
  three Haiku `web_fetch_apply` requests automatically made by Claude Code's
  WebFetch tool. They were not reviewer delegation, but the exact-model gate
  permits no exception. Subsequent reviews deny WebFetch/WebSearch and require
  direct `curl`/`gh` retrieval through the Opus-controlled Bash tool.
- Claude round 16 is exact-model valid and non-clean. It found a paper/blog
  value and shape mismatch in `CLAUDE.md` plus a stale technique count in
  `SKILL.md`. Both shipped-document errors were independently reproduced and
  repaired with focused cross-document/count regressions. The clean-round
  streak remains zero.
- Claude round 17 is exact-model valid and non-clean. Seven high-confidence
  navigation, citation, corpus-state, and receipt errors were independently
  confirmed; its medium-confidence FlashAttention ordering concern was also
  accepted because no primary evidence supports the ordering. All eight leads
  are corrected, deterministic coverage and numeric receipts are rebuilt, and
  the clean-round streak remains zero.
- Claude round 18 is exact-model valid and non-clean. Four shipped-document
  errors and one grouped paper-classification inconsistency were independently
  confirmed. The three named pages are the complete arXiv-backed blog set; all
  were migrated to `source-doc`/`paper`, references and indices regenerated,
  and a validator now rejects recurrence. The clean-round streak remains zero.
- Claude round 19 is exact-model valid and non-clean. Its four findings were
  independently confirmed: three PTX navigation locators were wrong, shipped
  prose overstated the wiki-only Blackwell-relevance rule, two closed PRs used
  a false merge-SHA sentinel, and a six-author post was credited to one author.
  The source maps, shipped contracts, PR generator/validator, metadata, and
  four focused recurrence tests are corrected. The clean-round streak remains
  zero.
- Claude round 20 is exact-model valid and non-clean. Two stale authored PTX
  anchors and one validator/test architecture-family divergence were confirmed;
  an independent all-fragment sweep also found and repaired a stale upstream
  locator in the compiled CUTLASS documentation source. Curated PTX anchors
  now have an offline resolution contract, Hopper/Blackwell values share the
  canonical policy, and the pattern schema accepts the relevance field the
  validator can require. The clean-round streak remains zero.
- Claude round 21 is exact-model valid and non-clean. Its two confirmed errors
  were independently reproduced: the Qwen3-Next source replaced a named author
  with organizations, and one present-tense numeric receipt remained at 3,806.
  Its medium-confidence schema-example lead was also accepted because concrete
  fields attached to real PR 2472 contradicted both the live pull and retained
  page. The example is now wholly exact rather than ambiguously templated, and
  three focused tests prevent recurrence. The clean-round streak remains zero.
- Claude round 22 is exact-model valid and non-clean. Its two confirmed errors
  were independently reproduced: the Modular source credited its publisher
  instead of its displayed four-person byline, and two shipped wiki-scoped
  grep examples returned no results. Its medium-confidence NSA attribution
  lead was also accepted after the arXiv API and title page confirmed fifteen
  authors across three institutions. The two source records, propagated prose,
  and all duplicated examples are corrected; three focused tests cover both
  attributions and execute every concrete shipped query command. The clean-round
  streak remains zero.
- Claude round 23 is exact-model valid and non-clean. Its confirmed CUTLASS
  changelog finding was independently reproduced: two cells documented as
  changelog labels instead said only “patch release.” Its attribution lead was
  also accepted after the Hugging Face community page identified Konstantin
  and handle `apsys` as the author. The date rows now preserve the rendered
  labels and transparently distinguish the missing checked-in 4.6.2 section;
  the author field follows the rendered name-plus-handle convention. Two
  focused tests prevent recurrence. The clean-round streak remains zero.
- Claude round 24 is exact-model valid and non-clean. Both confirmed receipt
  defects were independently reproduced: the current regression ledger said
  144 rather than 149 tests before the new recurrence test, and one historical
  verbatim log contradicted its own 87-bundle stdout with a forward expectation
  of 76 while the current strict verifier reports 37. The ledgers now record
  150 tests, preserve the dated 87-bundle capture as history, use a
  corpus-dependent success contract, and bind all current test/bundle totals
  to executable disk counts. The clean-round streak remains zero.
- Claude round 25 is exact-model valid and non-clean. Its three findings were
  independently reproduced: an undated runtime-consumed size receipt disagreed
  with the generated 110-core-PR manifest and retained an orphan pilot row; a
  historical memo applied the artifacts-only 6,000-file budget to the entire
  working tree; and the JAX tutorial's rendered “The JAX authors” credit had
  acquired an unsupported Google affiliation. The current size contract,
  historical scope, and source attribution are corrected, with three focused
  recurrence tests. The clean-round streak remains zero.
- Claude round 26 is exact-model valid and non-clean. Its sole finding was
  independently confirmed as an audit-introduced correction error: organizer
  task prose labels the per-16 NVFP4 scale E4M3FNUZ, organizer executable
  references construct `torch.float8_e4m3fn`, and PTX's `UE4M3` is a distinct
  unsigned 7-bit encoding. Seven propagated pages now disclose the source
  conflict instead of selecting or equating encodings, and a new cross-file
  regression protects that boundary. The clean-round streak remains zero.
- Claude round 27 is exact-model valid and non-clean. Its sole finding was
  independently confirmed as adjacent residue from the NVFP4 repair: the two
  organizer tasks expose only per-16 `sfa`/`sfb` scale tensors, while three
  wiki statements still attributed a separate tensor-level scale to them.
  Complete task, starter, and executable tuples now bound the corrected prose,
  and a new regression covers both contest pages and all three wiki consumers.
  The clean-round streak remains zero.
- Claude round 28 is exact-model valid and non-clean. Its sole attribution
  finding was independently confirmed: the reward-hacking article renders only
  Natalia Kokoromyti's participant byline, while the source record and contest
  cross-reference had elevated GPU Mode from host to co-author. Frontmatter and
  both prose locations now preserve the participant/publisher boundary, with a
  focused regression rejecting the former forms. The clean-round streak remains
  zero.
- Claude round 29 is exact-model valid and non-clean. Its three findings were
  independently reproduced: a third reward-hack consumer retained organizer
  authorship; a live inclusion-policy comment contradicted Triton 3.6 support,
  its own parsed scalar, and its cited page; and a worked example pointed to a
  removed pre-3.6 subsection. The consumer scan is now dynamic, AC-11 covers
  raw comments and parsed data, and navigation matches the actual opening
  paragraph. Two new regressions plus the broadened attribution guard cover
  these repairs. The clean-round streak remains zero.
- Claude round 30 is exact-model valid and non-clean. Its sole finding was
  independently confirmed: the reward-hack source inferred a gendered pronoun
  for Natalia Kokoromyti even though the cited article is first-person and the
  byline, leaderboard, and cited public profile provide no pronoun statement.
  The source and current receipts now use neutral author wording, and the
  dynamic consumer guard detects gendered third-person pronouns in sentences
  naming the author, including an adversarial recurrence. The clean-round
  streak remains zero.
- Claude round 31 is exact-model valid and clean. It independently reproduces
  the complete 1,091-file denominator, all scope/architecture/numeric totals,
  every mandatory gate, the Round-30 neutral-identity correction, and an
  exhaustive same-class attribution sweep with zero findings. It contributes
  the first of the two required consecutive clean reviews; the streak is one.
- Claude round 32 is exact-model valid and clean. In a fresh process it again
  reproduces the 1,091-file denominator, every scope/architecture/numeric
  total, all mandatory gates, the neutral-identity correction, and the
  same-class attribution sweep with zero findings. It is consecutive with
  Round 31 and supplies the second required clean review; the quorum is 2/2.

## Explicit residual limitations

- Source-reported performance measurements were checked for exact attribution
  and benchmark context but were not re-benchmarked. They remain labeled and
  scoped as source-reported.
- Partial code excerpts depend on their pinned surrounding implementation and
  are not represented as standalone benchmarks or kernels.
- Rolling documentation and live organizer/leaderboard pages are dated to the
  2026-08-18 access snapshot; later changes do not retroactively make that
  historical snapshot false.
- Upstream PR prose is preserved as an attributed report. The audit verifies
  that KernelWiki does not invent or broaden it; it does not independently
  reproduce every upstream author's experiment.
- The offline release checker proves equality with the committed release-list
  receipt and ages that receipt. It cannot discover a release published after
  2026-08-18 without a new authoritative refresh, so the limitation is stated
  rather than represented as live-network completeness.
- The 944-row architecture receipt embeds digests of the ignored frozen upstream
  snapshot and file-evidence corpus. A clean clone can detect page or policy
  drift from the committed receipt; external independence is supplied by the
  nineteen-case live deterministic GitHub sample gate. Reconstructing the complete frozen
  receipt still requires reacquiring the upstream evidence identified by those
  digests.
- GitHub's pull-files endpoint exposes at most 3,000 rows. All six affected PRs
  therefore disclose incomplete REST enumeration, while their complete pull
  diffs provide exact-count policy evidence. The live gate re-fetches those
  diffs; it does not represent the REST listing itself as complete.
- Complete-file receipts are frozen to the PR head and the current CUDA or
  Python-DSL signal pattern and are hashed only while that evidence is required
  by the current policy. They prove the policy verdict for that captured
  revision, not that an upstream file can never change later.
