# External review of repository kernels

This guide is for an external agent or human reviewer. The importer never
executes this prompt, invokes a model, or requires a specific model or prompt
version. It prepares materials and validates saved decisions.

## Basis and scope

The existing `CLAUDE.md` and `references/schema.md` define wiki scope, confidence,
reproducibility and performance contracts. The archived Phase 2 plan supplies the
include/exclude/defer pattern, and current `scripts/pr_policy.py` refines positive
device evidence. Representative selection and content-bound entry allowlists
are adapted to repository input. This is an adapted guide, not a claim that all
historical prompt wording was preserved.

Review only the pinned input repository's tracked files. Submodules are listed
as skipped and are not part of its kernel inventory. A Git repo is required for
provenance. Review originals and callers; a file extension, function name or
`@triton.jit` decorator alone does not establish an actual launchable kernel.

## Selection before detailed review

Scan, selection and detailed decisions are independent saved artifacts. A scan
never establishes helper status, implementation identity or admission. The
importer generates materials and validates files; it never invokes this guide
as a model task. For a fresh evaluation, do not read prior runs, saved verdicts,
old groupings, correction notes or answer examples. Only the pinned inputs,
current materials and authoritative wiki contracts belong in that evaluation.

Start with `selection.yaml`, `inventory.yaml` and the pinned checkout. There is
no per-definition detailed review queue at this stage. Use one reviewer for the
repository; task splitting is optional, not a requirement of the process.

Keep selection lightweight. A reachable entry with no concrete local duplicate
clue needs purpose, launch and source evidence, not a complete detailed ledger.
Inspect the source differences needed to establish helper exclusions and member
relationships. Prefer representatives with applicable targets, evidence and
teaching value; never choose solely because an entry appeared first. Target
names alone do not split an implementation or prove binary compatibility.

Helpers do not enter detailed review or effective-kernel counts. A real launched
numerical reduction or tensor transformation still needs a purpose-based scope
decision; being called by another operator alone does not make it a helper.

Every scanned definition starts pending. Same-file buckets and Python direct-call
or grid-launch locations are navigation suggestions only. They do not establish
semantic identity, device-only status, or absence of an external launcher. Check
aliases, imports, dispatch, templates and actual callers as relevant.

Set one role per entry, keeping its candidate ID and content digest:

- `helper`: internal support only, with a short `caller_basis`, rationale and
  source receipts establishing the caller/launch relationship. No separate
  kernel verdict, full source-fact ledger or page is required. Follow helpers
  from a real kernel as needed. Pure setup entries should be excluded only when
  their inspected purpose warrants it; do not classify a real reduction or
  compute kernel as support merely because it is called from another operator.
- `representative`: an actual implementation selected for detailed review, with
  rationale and source receipts. A launch establishes a real entry, not a
  separate implementation family. For suspected local variants, explain the
  source differences that warrant separation or confirm a member relationship
  before detailed review; a blanket "do not assume equivalence" is unfinished
  relationship analysis. This is not an include verdict.
- `member`: a confirmed same-implementation duplicate, ordinary tuning variant or
  version. Set `representative` to a representative candidate ID, `relation` to
  `exact_duplicate`, `variant` or `new_version`, and `difference_summary` to the
  inspected differences and why they do not require a separate implementation.
  Cite both definitions and relevant dependencies/dispatch. Never establish
  identity from a body hash, file name, same operator or dtype/tile label alone.
- `pending`: unresolved launch status, scope or relationship. State what remains
  unknown. Do not force uncertain members into a group to reduce counts.

You may split groups, combine them, change the representative or ignore all
suggestions. Scripts do not automatically merge across source locations: the
current extractor cannot prove complete dependency/compile-context equivalence.
Repeated candidate enumeration is rejected by coverage/identity validation.

Run `python3 scripts/import_repo.py select --run <run>` after saving selection.
Then read `review-targets.yaml`, `matches.yaml` and the relevant sources. The full
`review-request.yaml` remains available for machine context and deeper inspection.
Only representatives have full verdicts in `decisions.yaml`; members retain the
selection comparison. Member-only matches to existing wiki/PR evidence and other
groups are projected onto their representative, so grouping cannot hide overlap.
Internal group comparisons are represented by saved member relationships instead
of repeated reciprocal verdicts.

If detailed inspection disproves a group, edit selection and rerun `select`.
Previous review files are archived under `review-attachments/selection-*`.
Unchanged groups with still-valid content/context/comparisons can reuse verdicts;
changed groups require renewed decisions. Do not edit prepared matches to make a
group pass. Helpers stay outside the detailed queue and unresolved definitions
stay pending. An excluded or deferred representative does not admit its members.

## Suggested prompt

After confirmed selection, read `review-targets.yaml`, `matches.yaml`, the pinned source checkout and the
existing kernel/PR pages and code evidence included in the context.

1. Identify actual device implementations. Inspect host launchers, template
   dispatch, macros and helpers. Exclude host-only wrappers, support helpers,
   unrelated tests and generated material. Report unresolved dependencies rather
   than treating them as proof of equivalence.
2. Preserve worthwhile operation families and genuinely different implementations.
   Follow Blackwell-first scope, existing Hopper relevance requirements,
   controlled vocabulary and English canonical page conventions. Do not guess
   an architecture when source evidence is missing.
3. Resolve every discovered overlap before inclusion. Compare registry entries,
   existing kernel pages, PR code assets and same-batch representatives. Cite the
   matched evidence and the input source. Decide `distinct` or
   `same_implementation`; explain mathematical behavior, signatures, helper and
   dependency differences. Renaming and repo names alone do not decide identity.
4. For the same implementation, choose one representative using applicable
   evidence, reproducibility, intended scope and teaching value. Do not simply
   prefer the newest commit. Different sources, newer versions and tuning
   variants share a single canonical kernel page. Keep the existing page when
   it is the better representative. Explain the selection reason.
5. Compare the implementations represented by every canonical page linked from
   a matching PR. A multi-implementation PR's source links do not establish that
   all of its code belongs to each page. Reuse a page representing the same
   implementation. A new canonical page may cite that PR only when every linked
   canonical implementation is source-compared as distinct; any unresolved or
   same-implementation canonical relation blocks a distinct page. If only PR
   source records exist, a new canonical page may cite that PR and the repo.
   Do not create another PR page or copy its prose as a second implementation.
6. Choose existing confidence values with a `confidence_basis`. Review approval
   and repo existence do not imply verified confidence. Choose only snippet,
   runnable or benchmarked reproducibility. Higher levels require applicable
   content-bound evidence. Identify upstream-reported versus locally-executed
   evidence. Never transfer discarded-version metrics to the selected version.
7. Use `defer` for unresolved relationships or insufficient evidence. Explain
   the next comparison or evidence needed. Only those candidates stay pending;
   unrelated resolved candidates may publish. A simple include cannot bypass
   overlap resolution. Do not fabricate provenance, correctness or performance.
8. Save structured decisions, with current content/context hashes and IDs.
   Preserve original agent input/output under `review-attachments/` when
   available. No prompt-version change alone invalidates saved admission.

For historical PR attribution, distinguish a pinned source snapshot from the
implementation actually changed by the PR. In a shallow clone, `git show` may
treat a boundary commit as a root and display existing files as entirely added.
Read the real first parent from the raw commit (`git cat-file -p`) and compare
the explicit parent and commit trees, or inspect a retained upstream patch.
Check the relevant function's actual changes; an unchanged neighboring function
does not establish that the PR introduced or modified that implementation.
Record missing parent trees, blobs or patches as provenance gaps, preserving
the source analysis already completed. Do not replace missing evidence with a
synthetic root diff, or describe available material not yet reviewed as missing.

Repository membership alone is context, not implementation-overlap evidence.
An existing canonical target may receive a newly established PR attribution even
when that PR currently links only other implementation branches. Review the target
as the same implementation through its own kernel or registry hint, and resolve
every currently linked canonical page as distinct. Missing comparisons or another
same-implementation linked page block this path; use the existing shared-source
canonical rules when several linked pages represent the same implementation.

Prepared PR hints use matching source paths (including historical directories),
code receipts or meaningful normalized names. Generic module names such as
`_cute.py`, `_kernels.py`, `_utils.py` and `default.cu` are not name-based overlap evidence,
including when unrelated modules both export `kernel` or `main`. Exact source
paths (including retained `key-files` paths), source/body hashes and explicit
discoveries still produce hints. A canonical page citing the repo
also needs an operation clue from its kernel types and the input path/entry;
repo attribution alone does not make every kernel a possible copy. All indexed pages remain in the
review context; inspect other relevant records when the source reveals another
relationship. A missing hint is not proof of distinctness. Record newly discovered
overlaps in `discovered-overlaps.yaml` at the run root, with `candidate_id` and
either an existing `page_id` or a same-run `target_candidate_id`,
a rationale, `input_evidence` and `matched_evidence` receipts. Use the format in
[the importer guide](repo-import.md#review-and-unresolved-overlap). Ask the
maintainer to run `prepare-review` before admitting them; never modify generated
matches or requests. All refreshed hints must then be resolved. Changing or
removing this comparison input also requires refreshed materials.

## Review procedure and completion

Build a source-fact ledger before writing verdicts. For every representative undergoing detailed review, record:

- the actual entry kind, definition and enclosing class/template;
- input, output and scratch tensors, mathematical transformation or diagnostic
  effects, and optional branches that change those contracts;
- a reachable launch site, including inherited methods or template dispatch;
- the helpers needed to justify the claimed computation or hardware behavior;
- source-declared architecture, operand/accumulator placement and synchronization
  when relevant; distinguish comments, inspected implementation and inference;
- source paths, hashes and precise line/function locators for these facts.

A class method signature is not a launch site. A helper name or an instruction
name is not evidence of where its operands or results are stored: follow the
allocations and helper implementation. Copying one candidate's description into
another is not an inspection. Do not invent a scope rule from a directory name;
an exclusion must connect the inspected purpose to the saved scope requirements.

Trace both sides of architecture conditions and the continuation into the
launcher. A condition that disables a hardware feature or selects a fallback is
not a minimum architecture requirement. A helper's early return does not reject
the whole operator unless its caller propagates that restriction. Distinguish
exact eligibility checks, compile-target declarations, feature selection and
fallbacks, and cite the branch that supports the claim.

Assess scope separately from architecture precision. The schema permits an empty
architecture list when an exact target is not supported by evidence; this alone
does not require deferral. Inspect project-level requirements, entry dispatch and
helpers for relevance without inventing an exact target matrix. A snippet-level
review does not require the build/CI matrix or execution receipts required for
higher reproducibility grades. A confirmed Hopper-only path still needs the
existing Blackwell relevance justification; an unknown target must not conceal
an inspected restriction. Defer concrete unresolved scope evidence, not just a
missing exact architecture label.

Use the ledger to compare each prepared hint. Generic entry names can produce
noisy hints; their existence alone does not establish equivalence or prevent
source comparison. Keep all hints and resolve them using the actual contracts.
Same mathematics does not by itself prove the same implementation, and different
architectures do not by themselves prove different implementations. Examine
device dataflow, instruction primitives, storage, pipeline and synchronization;
separate substantive changes from tuning, renaming or ordinary version updates.

The review is complete only when:

1. Every scanned definition has a selection role. Every representative has one
   decision and its own source-fact record; every member has an inspected
   difference/relationship record. Helpers need only selection evidence. Every
   required comparison for an include has been investigated.
2. Every representative include has a resolution for every prepared hint, and reciprocal batch
   resolutions and representative choices agree across the whole run.
3. Every exclude has a substantive scope reason backed by inspected code and
   callers. Every defer identifies a concrete unresolved relationship or missing
   evidence, what was examined, and what would resolve it.
4. Metadata, rationale, comparison notes and source facts agree. Vocabulary and
   evidence checks below pass, and remaining uncertainties are disclosed.

Work not yet performed is unfinished review, not evidence that the source is
unreviewable. Save a progress record and continue reading or split the task;
do not present a blanket defer because many comparisons remain as a completed
review. Genuine unresolved cases may remain deferred without forcing approval.

## Evidence and vocabulary checks

For each consequential factual claim, cite the inspected definition, launch,
helper or comparison location. Recheck input versus saved output, forward versus
backward, accumulator versus epilogue storage, and declarations versus supported
targets. Ensure prose does not contradict the metadata. Do not claim execution,
correctness or performance from static source inspection.

Read `data/tags.yaml`, `data/schemas.yaml`, `CLAUDE.md` and
`references/schema.md`. Construct separate allowed-value sets for each metadata
field using the same categories as `scripts/validate.py::validate_file`.
In particular, `tags` accepts the union of `hardware_features`, `techniques`,
`kernel_types` and `languages`; an architecture alias or a confidence value found
elsewhere in the vocabulary is not automatically a tag. Each other structured
vocabulary field accepts only its own category. List values must be unique. If
`hardware_features` is present, include all hardware-feature values used in `tags`.
Check types, confidence, reproducibility, architecture values and any conditional
scope/evidence requirements. Do not validate by searching for a word anywhere
in the vocabulary file.

Check current IDs, context/content bindings and evidence receipts with
`scripts/repo_import/decisions.py::validate_decisions` when available. This
read-only check does not replace full page/schema validation. The primary
reviewer or maintainer must also run the actual build and temporary-wiki
validation, including generated indices and the size gate, before publication.
The review agent must not publish unless separately authorized.

## External reviewer organization

Use one reviewer for a repository, performing selection and then detailed review.
No separate grouping agent or per-kernel worker is required. If a maintainer
explicitly splits work, assign complete confirmed groups and reconcile their
cross-group comparisons before assembling the saved decisions.

For an independent evaluation, start fresh with file-based context and exclude
previous verdicts, correction feedback, generated pages and repository-specific
answer examples. Provide the prepared sources and neutral format instructions,
not an expected admission count. Save the task and outputs under persistent run
attachments for audit.

## Decision output

The document has `version: 1`, the prepared `context_digest` and
`selection_digest` when selection is enabled, optional
`review_provenance`, and `decisions`. Write full rows only for representatives. Every row requires candidate/content IDs,
include/exclude/defer, a substantive rationale and source evidence. Evidence
receipts have path, SHA-256 and locator. Use current prepared values.

Receipt paths have explicit roots: top-level `decision.evidence` resolves from
the pinned input checkout. Within `overlap_resolutions`, a batch hint's receipts
resolve from that checkout; existing wiki, PR and registry hint receipts resolve
from the wiki root. Keep input-side comparison evidence in the top-level source
evidence or source-fact ledger when resolving an existing-page hint. The
resolution itself cites the matched wiki/page/asset receipt using its wiki-relative
path. A valid hash with the wrong path root still fails validation.

For each hint on an included candidate, add a resolution:

```yaml
overlap_resolutions:
  - match_id: <copied match-id>
    outcome: distinct
    rationale: <specific comparison explaining a different implementation>
    evidence:
      - path: <matched page or code path>
        sha256: <copied receipt>
        locator: <compared entry, source section or exact lines>
```

For `same_implementation`, also set `representative` to the selected candidate
ID or existing canonical page ID. A batch representative must be included and
must not itself merge into another candidate. Reciprocal decisions must agree;
multiple candidates cannot each create a page while declaring the same identity.
Do not edit or remove prepared hints. Build/apply checks current source and wiki
context, including PR code hashes.

With selection enabled, a newly confirmed relationship between representatives
must be recorded by revising `selection.yaml` and rerunning `select`. Keep one
representative and make the other a member, with comparison receipts. Do not
write a full `merge_candidate` verdict for a selected representative: member
merge records are generated internally from selection. The action remains
available for batches that explicitly disable selection. This keeps all group
relationships in one place and avoids chains of merged representatives.

If a matched PR has no linked canonical kernel page, a reviewed implementation
may use `action: create_page`, `relation: distinct` and resolve that PR hint as
`same_implementation` with `representative` set to its own candidate ID. Here
`distinct` means creating a canonical implementation identity, not denying its
relationship to the PR. Cite the historical implementation and the matched PR;
the generated kernel page will cite that PR automatically. A PR page ID is not
a canonical kernel target. If the PR already links to a kernel page, resolve
and reuse that canonical identity instead of creating another page.

Every include requires `action`, `relation`, `confidence_basis` and `metadata`.
The required metadata fields are `title`, `architectures`, `tags`, `kernel_types`,
`languages`, `confidence` and `reproducibility`. Use a string for the title and
grading fields, and lists for the vocabulary fields. An empty architecture list
is permitted when the source does not justify an exact architecture. The renderer
supplies `id`, `type`, `repo`, `related`, `sources` and `import_record`; do not
invent those generated values. Optional fields still follow `data/schemas.yaml`,
including conditional `blackwell_relevance` and `evidence_basis`. Non-snippet
reproducibility requires the content-bound receipts described above; benchmarks
also require actual performance claims. An empty claim list is valid for snippets.

## Publication actions

Existing kernel page refers here to a page already in the wiki whose body is
maintained outside this importer. It makes no assumption about the author or
the tool originally used to create it.

| Relationship | Action | Representative choice |
| --- | --- | --- |
| distinct | create_page | Incoming implementation gets its own page after all comparisons resolve. |
| exact_duplicate | record_duplicate | Existing registered page remains; add source attribution. |
| exact_duplicate of an existing kernel page | attach_source | Existing page remains; add attribution and registry identity. |
| new_version or variant | extend_page | Incoming version replaces the representative on the same managed page. |
| new_version or variant | retain_page | Existing representative remains; retain a compact incoming relation. |
| exact_duplicate, new_version or variant within a batch | merge_candidate | Another included candidate represents the implementation. |

Registered targets require `target_implementation_id`, `target_page_id` and
`target_version_id`. Batch merge requires `target_candidate_id`. Selection
through extend/retain/merge requires `selection_reason`.

For an unregistered existing kernel page, attach_source requires `target_page_id`,
`equivalence_basis` and `target_evidence` receipts in wiki or code assets. Existing
prose and reproducibility are preserved. Automatic body replacement of an existing kernel
page is not supported; such edits require a separate reviewed page edit.

An existing kernel page can also remain the representative of an incoming new_version or
variant: use retain_page with target_page_id, equivalence_basis, target_evidence,
and selection_reason, without a target implementation/version until registered.
If the same existing page explicitly covers several different implementation
branches, each incoming branch may use `retain_page` for that page. Each decision
must cite a different literal `anchor` in its `target_evidence` receipt for the
target page, explain the correspondence in `equivalence_basis`, and resolve a
mandatory reciprocal batch comparison as `distinct` against every other branch.
Missing scopes or comparisons keep the group pending. This does not turn duplicate
implementations into separate branches: equivalent versions still select one
representative. Publication preserves one existing page and one registry record,
with separate compact source relationships and their scoped receipts.
If several existing kernel pages cite the same implementation in one PR, choose
one canonical page through `retain_page` or `attach_source`. Resolve the common
PR and each equivalent page as `same_implementation`, naming that canonical
page as representative. `target_evidence` must include content-bound receipts
for both the canonical and other pages, with literal `anchor` text containing
the shared PR ID; explain the exact source correspondence in `equivalence_basis`.
Both pages must be linked by the reviewed PR hint. Shared citations alone never
decide equivalence, and unrelated implementations in that PR still need separate
comparisons. Only the chosen page receives the incoming attribution; the other
page is preserved. This does not merge or rewrite existing wiki pages.

Saved existing-page attribution is reusable only while its registered page receipt
matches the current page. An external page edit requires fresh review even when
the same anchor text survives. Importer-owned attribution updates renew that
receipt as part of the validated publication plan.
The registry stores the selected existing kernel page's receipt and actual classification;
incoming content is only a compact relationship record. A reviewed equivalent
copy whose fingerprints differ may use record_duplicate with explicit input and
target comparison evidence, not name-based equivalence.

The selected representative is the only full version stored in the runtime
registry. Nonselected versions preserve compact origin, content digest,
relationship and reason, with full review history in run storage. The registry
is an importer bookkeeping file, not an additional user-facing kernel page.

## Confidence and execution evidence

Use the existing wiki definitions for source-reported, inferred, experimental
and verified. Verified must also satisfy the existing validator's source and
`evidence_basis` contracts. Do not equate this with a local GPU correctness run.

Runnable requires `reproducibility_evidence` build, run and environment receipts.
Benchmarked additionally requires benchmark, inputs, method and results receipts,
plus valid `performance_claims`. Each receipt names a committed snapshot file,
SHA-256, locator and basis (`upstream-reported` or `locally-executed`). Local
reports must already belong to the reviewed snapshot; the importer does not
create commits or execute programs to obtain them.

Repository-specific onboarding examples are documented separately in
`examples/repo-import/README.md`. They are review materials, not preinstalled
kernels or active presets. Exclude these reviewed examples from an independent
evaluation; use the neutral output structure in this guide instead.

For selected members the importer expands the saved relationship into an internal
`merge_candidate` record after the representative is included. Such records use
`inferred` confidence and `snippet` reproducibility and carry no inherited timing
or execution claims. Only the representative determines page classification.
Legacy runs without selection retain their existing per-candidate contracts.
