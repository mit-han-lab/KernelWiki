# Public repository review archive

These are read-only review results for 14 public repositories. Sources were pinned
from main/master observations on 2026-10-16; later review completion does not
imply that these commits are the latest upstream revisions today. Full commits
and repository URLs are in the extracted `index.yaml` and each snapshot.

**No kernels have been published by this archive.** It creates no Wiki pages,
source pages, active registry, query indices or enabled presets. The temporary
review website and local checkout/history caches are not included.

## Results

| Repository | Effective groups | Include verdicts | Exclude | Review defer | Selection pending |
| --- | ---: | ---: | ---: | ---: | ---: |
| cudnn-frontend | 268 | 268 | 35 | 0 | 3 |
| cutlass | 226 | 227 | 34 | 2 | 1 |
| DeepGEMM | 20 | 20 | 4 | 3 | 0 |
| flash-attention | 43 | 43 | 2 | 1 | 2 |
| flash-linear-attention | 211 | 211 | 38 | 40 | 5 |
| flashinfer | 204 | 205 | 115 | 57 | 14 |
| FlashKDA | 2 | 2 | 0 | 0 | 0 |
| FlashMLA | 1 | 1 | 1 | 6 | 0 |
| quack | 23 | 23 | 0 | 0 | 0 |
| qutlass | 7 | 7 | 0 | 0 | 0 |
| sglang | 393 | 393 | 47 | 4 | 7 |
| TensorRT-LLM | 183 | 183 | 163 | 46 | 0 |
| tokenspeed | 138 | 138 | 48 | 18 | 2 |
| vllm | 261 | 261 | 245 | 0 | 0 |
| **Total** | **1980** | **1982** | **732** | **177** | **34** |

Include verdicts count representatives; effective groups count canonical
implementations. More than one admitted representative can resolve to the same
group. Ordinary variants remain members instead of duplicate effective kernels.

The `pending.yaml` list preserves all **211** holds: **177** detailed-review
defers and **34** initial-selection pending entries. They are not admitted.
Helpers are excluded from detailed review. Review completion means every selected
representative has an explicit verdict, including justified defers; it does not
mean every candidate passed. Confidence and reproducibility are copied unchanged.
No new GPU execution or correctness validation was performed during packaging.

## Contents and portability

[`records.tar.gz`](records.tar.gz) contains the complete portable YAML records.
[`archive.yaml`](archive.yaml) binds its SHA-256, uncompressed byte count, file
count and exact verification result. Compression preserves every byte. To read
the records from the repository root:

```bash
mkdir -p .cache/repo-import/review-readback
tar -xzf examples/repo-import/reviews/2026-10-16/records.tar.gz -C .cache/repo-import/review-readback
```

The files below are inside that archive. No source checkout is needed to read
them. The working-tree size budget includes the compressed archive; it is not
excluded from the size check.

- Each repository: `snapshot.yaml`, `inventory.yaml`, `selection.yaml`,
  `decisions.yaml`, `matches.yaml`, `discovered-overlaps.yaml`, `summary.yaml`,
  and `valid-kernels.yaml`. Scan, initial selection and detailed review stay separate.
- `prompts/`: the actual saved review instructions, deduplicated by original hash.
- `contexts/`: saved Wiki comparison context and policy receipts. A registry digest
  describes the reviewed context; this is not a populated active registry.
- `evidence/` and `evidence-index.yaml`: directly cited documentary evidence,
  including public source diffs where used (which can include added source).
  Upstream source trees and standalone implementation files are not vendored.
- `index.yaml`: normalized file hashes, original capture file/record hashes, pinned
  sources and exact result counts. Standard YAML aliases deduplicate repeated
  dependency records without discarding fields.

Source addresses use `git+https://github.com/<owner>/<repo>@<full-commit>/<path>`
or `package+https://files.pythonhosted.org/...#sha256=<package-hash>&path=<member>`.
They retain source SHA-256 receipts; package member bytes were checked against
the saved, hash-verified public wheels before packaging. `wiki://` addresses
refer to the reviewed Wiki context. Original relative upstream paths stay relative
to the pinned source, not to this archive directory.

Machine-local evidence paths become `archive://evidence/<original-hash>.yaml`.
The wrapped documentary content relocates path tokens; its original byte hash
and normalized archive file hash are separate. Original decision/selection/context
digests authenticate the original capture, not the normalized documents. Archive
file hashes bind the relocated records; this archive is not a runtime audit pack.

Iterative raw agent transcripts and superseded notebooks remain in persistent run
history outside Git. `history://` references navigate that external history;
`local-reference://` tokens identify uncopied local navigation. Neither is a
permission to admit a candidate. Directly cited structured evidence receipts are
preserved as portable archive/public-source references.

## Offline verification

From the KernelWiki repository root:

```bash
python3 scripts/archive_repo_reviews.py verify --archive examples/repo-import/reviews/2026-10-16
```

Verification checks all archived file hashes, selection coverage, representative
verdicts, admitted groups and members, unchanged confidence/reproducibility,
portable evidence receipts, and the complete hold list. It accepts compressed and extracted archives and works after relocating
this directory without the original cache or source checkouts. It is an integrity
check, not a fresh model review or GPU test.

## Reuse

`archive_only: true` marks the decision files as read-only; preset rebinding rejects
them. To reuse observations, prepare the desired Git repository and commit again,
revalidate source/dependency hashes and current Wiki overlaps, then adapt and
validate the decisions using the supported review flow. Changed source or unresolved
overlap relations still require review. Nothing here enables automatic publication.

## Packaging acceptance — 2026-10-28

All 319 repository tests pass, including eight archive regressions. The archive
checks also pass with the bundled YAML fallback. Independent relocation of the
complete compressed batch passes offline verification. Wiki validation retains
1,041 files, 37 asset bundles and 14 candidate ledgers; offline freshness and
the 40 MiB working-tree size gate pass. No real corpus was published.
