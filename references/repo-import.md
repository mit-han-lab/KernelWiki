# Optional repository kernel import

The importer is optional. It accepts a Git URL or a local Git repository root,
pins a commit, prepares review materials, and consumes decisions from an external
reviewer. It never invokes an AI model, executes downloaded kernels, or creates a
Git commit. An ordinary non-Git folder is rejected because it has no pinned
repository provenance.

Only files tracked by the input repository are scanned, including normal
subdirectories. Gitlinks are recorded in `snapshot.yaml.skipped_submodules` and
never initialized, fetched or scanned. There is no submodule expansion option.
Uncommitted worktree changes are excluded and recorded. CUDA candidates include
literal `__global__` and standard `CUTLASS_GLOBAL` declarations. A marker is
only a scan observation; external review still establishes target and launch
semantics. Qualifier-only macro definitions are not kernel implementations.
The scanner also recognizes bounded, same-file function-like declaration-prefix
macros with a literal entry name and an immediate body, plus object-like aliases
containing only CUDA qualifiers and known balanced launch attributes. Each alias
use is bound to its actual named function and physical source line; declarations,
prototypes and unrelated following structs are not kernels. It follows local
redefinitions and `#undef`, and retains the declaration plus selected body as
evidence. CUDA signatures balance complete parameter lists, including braced
default values and nested constructors; neither can replace the entry name or
truncate the selected source at an initializer. Conditional declarations retain
all source branches without choosing a preprocessing configuration.
This is not a C++ preprocessor: external macro wrappers, computed names
and arbitrary template instantiations are not enumerated. Reviewers must record
such coverage limitations rather than equating scan counts with complete runtime
kernel coverage.

## Commands

Run from the wiki root:

```bash
python3 scripts/import_repo.py prepare --repo /path/to/repo --revision HEAD
# First inspect launch relationships and edit selection.yaml independently.
python3 scripts/import_repo.py select --run <run-id>
# Review review-targets.yaml and matches.yaml; edit representative decisions.yaml.
python3 scripts/import_repo.py build --run <run-id>
python3 scripts/import_repo.py apply --run <run-id>
python3 scripts/import_repo.py status --run <run-id>
```

The compatibility `prepare` command scans and initializes a pending selection
draft; `select` validates that draft and prepares detailed materials. For explicit
stage separation use:

```bash
python3 scripts/import_repo.py scan --repo /path/to/repo --revision HEAD --fresh-review
python3 scripts/import_repo.py prepare-selection --run <run>
# External reviewer saves selection.yaml.
python3 scripts/import_repo.py validate-selection --run <run>
python3 scripts/import_repo.py prepare-review --run <run>
# External reviewer saves decisions.yaml.
python3 scripts/import_repo.py report --run <run> --out /persistent/reports/repo
```

`report` exports validated implementation lists without building or applying wiki
pages. A representative counts only after inclusion and all required comparisons
resolve; helpers and member variants are not separate effective kernels. Reports
validate field-specific vocabulary and separate unfinished work, genuine
deferrals and blocked includes. Counts are per
repository and must not be summed as a globally unique implementation total.
`--fresh-review` prevents old saved admission/exclusion decisions from populating
the new detailed draft. It does not conceal existing wiki overlap evidence.

Use `--config PATH` or `--wiki-root PATH` before the subcommand. Scan include and
exclude patterns use fnmatch; configure nested exclusion patterns explicitly
when required. Local input must be the repository root, not an arbitrary folder
inside it. A run ID or a full run-directory path is accepted by `--run`.

## Early selection and review size

`prepare` records every detected definition in `inventory.yaml` and creates a
pending `selection.yaml`. It does not yet generate detailed verdict templates or
all-pairs matching material. Same-file buckets and Python call sites are hints,
not automatic grouping or helper verdicts.

An external reviewer marks actual representatives, confirmed members, helpers
or pending entries. Helper exclusions require a short caller/launch explanation
and source evidence. Members require a direct representative, relation,
difference summary and evidence for both sources. Definitions with uncertain
status or relationships stay pending. See [the review guide](repo-import-review.md)
for the exact fields and review standard.

`select` validates complete coverage, hashes and relationships, then generates
`review-targets.yaml` and representative-only `decisions.yaml`. Helpers have no
full verdict or page. Members have compact relationship records, not independent
full reviews. Cross-group and existing-wiki matches discovered on any member
are retained on its representative. Only explicitly confirmed internal group
relations replace pairwise comparisons; suggestions cannot bypass review.

A reviewer can split/merge groups or change representatives at any time before
publication. Rerun `select`; old review files are archived. Unchanged groups may
reuse still-valid decisions; changed groups and comparisons need renewed review.
Unchanged decisions are revalidated together, including source receipts and
cross-candidate relationships, so a large completed review does not require a
full validation per decision. If an old record is invalid or conflicts with
another target, independent valid records can still be reused; unresolved
records remain pending.
Every evidence receipt must name an existing in-scope file, carry its complete
SHA-256 digest and provide a nonempty source locator. A missing file and an empty
digest never count as matching evidence, including during update reuse.
Archives retain the previous request, targets, decisions, matches, selection and
the actual review guide/instructions used in that round, including prompt-only
revisions that do not invalidate otherwise valid decisions.
Editing selection after build blocks apply. Full selection and scan history stay
in persistent run storage. Compact formal records retain admitted relationships,
necessary evidence, input digests and the immutable external run reference; they
never copy the entire selection, helper inventory or pending queue into the wiki.

Progress distinguishes scanned definitions, representatives, members, helpers
and pending work. A raw definition count is not an independent kernel count.
Unknowns remain pending even if unrelated reviewed representatives publish.

To omit Triton definitions, use a configuration override:

```yaml
version: 1
scan:
  entry_kinds: [cuda_global, cute_kernel]
```

The default includes CUDA C++, CuTe DSL and Triton. Changing this option is an
explicit scope choice, independent of helper filtering and representative
selection. `selection.enabled: false` supports legacy decision manifests and
restores the old per-candidate workflow; new runs use selection by default.

## Review and unresolved overlap

The run saves candidate entries, source/dependency hashes, unresolved imports,
parse failures, existing kernel and PR pages, code-asset receipts, matching hints,
and a snapshot of `references/repo-import-review.md`.

Hints cover registered implementations, existing kernel pages and PR assets,
and same-batch candidates. They are conservative comparisons, not proofs of
semantic equivalence. Same-repository PR records remain background context;
content, source paths or meaningful names establish prepared PR comparison hints.
They may include false positives and cannot guarantee that
all duplicate implementations are detected.

Every hint for an included representative needs an `overlap_resolutions` record:
`match_id`, `outcome` (`distinct` or `same_implementation`), rationale, and
content-bound evidence citing the matched page/file. For the same implementation,
choose a `representative` (candidate ID or canonical page ID). Missing or
unresolved relationships keep that candidate pending; unrelated reviewed
candidates may still publish. Deleting hints from saved materials cannot bypass
validation: current matching context is checked again.

For an existing PR with a linked kernel page, compare that canonical implementation
and reuse it when equivalent. A PR can contain several implementations: every
linked canonical page must be resolved, and a separate page is allowed only when
all of those implementations are source-proven distinct. If the PR only
records the implementation and no canonical kernel page exists, create one and
cite the PR as well as the repository snapshot. An existing PR is a source,
not a separate canonical kernel implementation.

Save source-discovered comparisons in `<run>/discovered-overlaps.yaml`, then run
`prepare-review` again. Do not edit generated `matches.yaml`:

```yaml
version: 1
entries:
  - candidate_id: candidate-...
    page_id: kernel-existing-example
    rationale: The source reveals a shared implementation lineage.
    input_evidence:
      - {path: src/example.cu, sha256: <source-hash>, locator: "example at line 42"}
    matched_evidence:
      - {path: wiki/kernels/example.md, sha256: <page-hash>, locator: kernel-existing-example}
```

Receipts must identify the actual input definition and current indexed page or
its code asset. For a newly discovered overlap between two scanned candidates,
replace `page_id` with `target_candidate_id` and cite the target definition in
`matched_evidence` (its repository-relative path, source hash and
`"entry at line N"` locator). One record creates mandatory comparisons in both
directions. Arguments, helpers, launch and output contracts still need review;
the record itself does not establish equivalence. When both entries belong to
one explicitly reviewed initial-selection group, that source-bound group relation
resolves their internal comparison; external comparisons project to its representative.
Unknown IDs, self comparisons, duplicate pairs (including reversed candidate
pairs), mixed target kinds and stale receipts fail validation.
The refreshed request exposes these as mandatory comparisons, including PR-linked
canonical pages and comparisons discovered on group members. Adding, changing or
removing this input after preparation blocks admission until materials refresh.
Updates rebind both ends of candidate comparisons. Changed or ambiguous endpoints
remain navigation requiring fresh review; missing endpoints stay in update history
and never imply that an existing wiki implementation should be deleted.

## One representative per implementation

Here, an existing kernel page means a page already in the wiki whose body is
maintained outside this importer. The term describes importer ownership,
independent of who authored the page. Importer-generated pages can be updated
through `extend_page`; existing-page attribution preserves the original body.

- `create_page` / `distinct`: create a canonical page for a genuinely different
  implementation, with resolved comparisons.
- `attach_source` / `exact_duplicate`: retain an unregistered existing kernel page,
  cite explicit equivalence evidence, and add repository attribution.
- `record_duplicate` / `exact_duplicate`: reuse a registered representative and
  add source attribution.
- `extend_page` / `new_version` or `variant`: select the incoming implementation
  as the new representative of the existing page.
- `retain_page` / `new_version` or `variant`: retain the existing representative;
  save only the incoming relationship and selection reason.
- `merge_candidate`: select another included candidate in the same batch as the
  representative; no additional canonical page is created for the merged entry.

Selection actions require `selection_reason`. Choose based on applicable
evidence, reproducibility, scope and teaching value, not commit recency alone.
Existing kernel page prose is never automatically replaced; body changes to such pages
require a separate reviewed page edit rather than `extend_page`.

Several different source branches already covered by one existing page may all
retain it. Review must identify distinct literal anchors in content-bound target
page receipts and resolve every pair's mandatory batch comparison as distinct.
Otherwise the group remains pending. This preserves one page/registry record and
separate source relationships; it does not merge different execution graphs or
allow ordinary versions to bypass representative selection.

When the same reviewed PR implementation is cited by several existing kernel
pages, explicitly select one canonical existing page. Review the PR and all
equivalent page hints as `same_implementation`, with that page as representative,
and cite literal shared-PR anchors in content-bound `target_evidence` for each
page. `retain_page` or `attach_source` records one attribution and preserves the
other pages. A shared PR reference does not automatically merge implementations.

The runtime registry stores one full representative version per implementation.
Other versions/variants retain compact origin, digest, relationship and reason
records; full scan and review history remains in persistent run storage. The
page shows the representative implementation plus a provenance table of member
paths, entries, pinned origins and relationship reasons. Each member retains its
own parameter/type limits; a table entry does not expand the representative's
supported dtype or shape range. Confidence, reproducibility and metrics come
from the representative's own evidence, not a discarded variant.

## Updating a pinned repository

```bash
python3 scripts/import_repo.py update --repo /path/to/repo --revision NEW_COMMIT --baseline /persistent/previous-run
python3 scripts/import_repo.py prepare-review --run <new-run>
```

`update.yaml` records unchanged, changed, added, missing and ambiguous entries,
plus possible renames. A rename/body match is a hint, not an equivalence decision.
The scan receipt binds an explicit `extractor_contract` independently of stable
per-entry fingerprint encoding. A coverage change requires a new immutable scan;
old results cannot silently authorize its additional definitions.
Source, dependency and explicit caller receipts must remain valid before a whole
selection group can be rebound. Removing a representative invalidates that group's
automatic reuse; the reviewer chooses any replacement. Detailed decisions require
current scope policy, group and comparison bindings, not just matching source text.

A changed skipped Gitlink pin also blocks reuse of dependency assumptions; its
contents are still never fetched or scanned.

Updates carry source-discovered comparisons forward as navigation hints for
unchanged, changed, ambiguous or possible-renamed successors. A carried hint is
not an identity decision. Only unchanged complete groups with valid original
source, target, policy and comparison receipts may reuse their reviewed verdicts.
Changed sources or targets require renewed comparison; missing targets and
successor ambiguities remain explicit in update history.

Missing observations authorize no automatic page deletion. Scan configuration or
extractor/coverage-contract changes, legacy scans without complete receipts, ambiguous overload identities
and parse failures disable deletion assertions and automatic reuse. Historical
commits remain valid provenance; removed sources may be discussed and retained.

The external per-repo observation index records the latest observed and latest
verified complete scan. Only a committed apply transaction records publication;
scan, initial selection, detailed review, report and build never advance it.
Incomplete publication cannot replace the complete published baseline. Use
`update --fresh-review` to prepare a comparison without reusing old review records.

## Persistence and publication

`data/repo-import.yaml` defines `state_dir` and `checkout_dir`, defaulting to
ignored `.cache/repo-import/`. Absolute paths are supported. Run history persists
until explicitly removed; retention and backups are user-managed. A run includes
snapshot, inventory, selection, review targets, matching, guide, decisions, optional `review-attachments/`,
staging, validation results, publication plan and backup/transaction journal.

Formal outputs are created only after a user executes import into a target wiki:
`sources/repos/`, `wiki/kernels/imported/`, registry and compact decisions under
`data/repo-import/`, license notices and query indices. Unedited pending templates
remain in run storage. The distributed feature has no populated registry,
imported kernel/source pages, or default enabled preset.

Apply first generates indices and runs page/registry and size validation in an
isolated copy. Publication uses a lock, before/after file hashes, backups and a
recoverable journal. Interrupted writes recover before context checks; concurrent
edits stop publication/recovery. A repeated successful apply does not write again.
Transaction status `committed` means file publication completed, not a Git commit.

Content changes invalidate old decisions. Changed wiki pages or PR code assets
require a new preparation. Previously admitted unchanged representatives may
reuse decisions if all current overlaps are resolved. Guide-version changes alone
do not invalidate prior decisions. Higher execution levels require committed,
content-bound evidence, and the importer performs no GPU validation. Scope/schema
policy receipts are bound separately from prompt wording; changing applicable
policy requires renewed review even when kernel text did not change.

Publication records a hash of the final canonical page in the registry. On a
repeat import, a former batch peer may now appear as a registry match. Its
distinctness decision is reusable only when both source bindings still match,
both original directions explicitly resolved the batch comparison as distinct,
and the published target page still matches that recorded hash. A missing
receipt, changed source/page or new comparison remains pending. Previously
reviewed distinct PR/page matches also retain their verdict only when the match
ID and all cited evidence receipts remain valid.

## Examples and presets

### Read-only review archives

`scripts/archive_repo_reviews.py export --batch <accepted-batch> --out
examples/repo-import/reviews/<batch>` packages an accepted multi-repository
review without building or applying an import. Optional repeated
`--public-source-catalog <saved-catalog>` arguments resolve historical source
receipts using recorded public Git origins or pinned public package members.
The exporter never fetches sources or calls a model. It verifies the accepted
decision/list bindings before relocating paths and refuses an existing output.

The archive separates inventory, selection, representative decisions, matching
inputs, effective lists, prompts and directly cited documentary evidence. Equal
dependency records use lossless standard YAML anchors. Original capture hashes
are recorded separately from normalized archive file hashes: relocating a path
changes document bytes and semantic digests. A snapshot of Wiki matching context
is retained; no active registry is populated.

`python3 scripts/archive_repo_reviews.py pack --archive <directory> --out
<new-directory>` creates a deterministic, lossless `records.tar.gz` and its
checksum manifest, retaining the input. The shipped complete batch uses this
format to keep the working tree small; compressed bytes remain subject to the
repository size budget. The verifier extracts into temporary storage, rejects
links, duplicate entries and paths escaping the archive, and checks both the
package hash and all inner records. Standard `tar -xzf` can extract a copy into
ignored storage for reading.

`python3 scripts/archive_repo_reviews.py verify --archive <directory>` checks
file hashes, candidate coverage, effective-group membership, unchanged metadata,
portable evidence bindings, and both stages of unresolved entries offline.
`archive_only: true` decisions are rejected by preset rebinding. The archive
contains no ready-to-apply decision manifest. Prepare the pinned source again,
recheck current matches and evidence, and validate before any future publication.

Raw iterative agent transcripts remain in the configured persistent run history.
`history://` references navigate that history; they are not admission evidence.
Public implementation bytes remain upstream, addressed by fixed Git commit or
package hash/member and their own source hash. Documentary diff excerpts can be
included, with original and normalized evidence hashes distinguished explicitly.

### Decision excerpts and optional presets

[Public review examples](../examples/repo-import/README.md) include Quack and
FlashMLA decisions and explanations without publication outputs. Adapt them to
fresh materials rather than blindly copying stale context or IDs.

The default `data/repo-import-presets.yaml` is empty. A user can explicitly add a
preset pinning `repo`, `revision`, `decision_manifest`, and optional scan config,
then run `python3 scripts/import_repo.py import --preset <name>`. With selection
enabled, the manifest must also contain the matching saved `selection`; legacy
examples require adaptation or explicit legacy mode. Export a portable full input
pack with `export-audit --run <run> --out /persistent/audit-pack.yaml`; this is
separate from the compact published manifest and does not publish content. Preset import
also restores source-discovered comparison inputs from that pack and validates
their receipts before reusing decisions. It
reuses valid saved review; changed candidates or unresolved overlaps stop it at
pending. There is no unconditional skip-review option.
