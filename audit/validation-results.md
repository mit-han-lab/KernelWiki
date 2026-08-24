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
