# Repository-local verification evidence

`local-snapshots/` contains the exact files and named source subtrees used by
the verifier reports. Nested Git metadata is intentionally excluded.

`MANIFEST.json` records the stable repository-relative path, revision labels,
file count, byte count, and SHA-256 digest for every copied evidence target.
Two complete CUTLASS checkout roots that were used only for repository-wide
searches are represented by the immutable upstream commit tree recorded in the
manifest; copying both full worktrees would have added roughly 200 MiB.

`historical-artifacts/` preserves six small files that remediation later
deleted or replaced but that original-review receipts still cite. Its separate
manifest records the source Git revision and SHA-256 for every restored file.

Exact host checks cited by receipts live in
`verification/tools/check_verification_evidence.py`. The aggregate audit checks structured
paths and commands, and can also fetch every current evidence URL and validate
NVIDIA fragments:

```bash
python3 verification/tools/audit_verification_evidence.py --network
```

Do not edit copied source files in place. Re-run the migration from the pinned
source checkouts when evidence must be refreshed, then validate the result:

```bash
python3 verification/tools/migrate_verification_evidence.py --check
```
