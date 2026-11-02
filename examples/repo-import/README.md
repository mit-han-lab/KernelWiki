# Public repository review examples

These are external review materials, not imported wiki contents. They do not
create pages, populate a registry, enable a preset, or appear in kernel queries.
The examples reference upstream implementation files by pinned provenance.
Documentary evidence can include public source diffs. Each example pins a public
repository commit and records candidate fingerprints, evidence and a decision.

- `quack-softmax-review.yaml`: review selects Softmax backward as an independent
  implementation and explicitly distinguishes its forward entry. The forward
  candidate has no admission decision and remains pending in this example.
- `flashmla-pending-review.yaml`: review identifies sparse decode device code,
  but defers publication until its relationship to existing FlashMLA/sparse-MLA
  pages is established. Finding device code does not bypass overlap review.

Example IDs, comparison hints and wiki-context digests belong to their recorded
snapshots. Run `prepare` for your chosen repository and scope, inspect current
selection first, run `select`, inspect representative matches, then adapt the decisions using current candidate/match IDs and evidence.
Do not treat `example_only` as a runtime permission switch: the examples are
outside the active decision directory and the default preset file is empty.

A build may publish a reviewed subset while keeping unresolved candidates
pending. Actual publication generates kernel/source pages, selected-version
registry records and indices only in the explicitly chosen target wiki. The
feature itself ships none of those import outputs.

## Complete read-only review batch

[The 14-repository review archive](reviews/2026-10-16/README.md) keeps separate
scan inventories, confirmed selection, detailed verdicts and validated effective
lists. It also retains the prompts used and directly cited documentary evidence,
which may contain public diff excerpts. It preserves both detailed-review defers
and initial-selection pending entries. These records are review results, not
published Wiki pages or an enabled import preset.

Verify the archive offline from the repository root:

```bash
python3 scripts/archive_repo_reviews.py verify --archive examples/repo-import/reviews/2026-10-16
```

The verifier checks archive integrity and status preservation. It does not rerun
source review or execute kernels. Reusing a decision requires fresh preparation
and validation against the chosen source and current Wiki context.

## Adapting examples to selection

The saved examples predate the selection stage and deliberately contain only
review excerpts, not a complete inventory or a ready-to-run preset. Keep their
source observations as evidence to recheck, then use a fresh run's IDs and hashes.
For the Quack excerpt, a forward/backward pair must not be merged simply because
both are Softmax. Select both as representatives if independently launched;
leave the unreviewed forward entry pending at the detailed-review stage. The
FlashMLA representative still requires its existing-page comparisons.

The new selection file uses this structure (placeholders are not review results):

```yaml
version: 1
inventory_digest: <prepared inventory digest>
entries:
  - candidate_id: <representative ID>
    content_digest: <prepared content digest>
    role: representative
    rationale: <why this actual launchable implementation was selected>
    evidence: [{path: <source>, sha256: <hash>, locator: <entry and launcher>}]
  - candidate_id: <ordinary variant ID>
    content_digest: <prepared content digest>
    role: member
    representative: <representative ID>
    relation: variant
    difference_summary: <actual differences and why this is the same implementation>
    rationale: <why one representative is sufficient>
    evidence: [{path: <source>, sha256: <hash>, locator: <both compared entries>}]
  - candidate_id: <internal helper ID>
    content_digest: <prepared content digest>
    role: helper
    caller_basis: <actual callers, launch analysis, why internal-only>
    rationale: <support purpose>
    evidence: [{path: <source>, sha256: <hash>, locator: <definition and callers>}]
```

Keep every prepared entry, including unresolved ones with `role: pending`.
When sources differ, member evidence needs receipts for both source files and
relevant dependency/dispatch evidence. Suggestions may be discarded. After
`select`, edit only representative verdicts and preserve `selection_digest`.
Members need no duplicate full verdict; helpers need no detailed fact ledger.
