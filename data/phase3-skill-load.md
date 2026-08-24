# Phase 3 Skill-Load Empirical Audit (DEC-5)

> Generated: 2026-04-17
> Tooling: measured with Python on the Phase 3 Round 2 repo state.

## Question

The plan's DEC-5 decision asked whether `artifacts/` files blow up Claude Code's boot-time context, and whether an exclude mechanism (`.claude-skill-ignore`) is needed.

## Measurements

All numbers taken on the final Round 2 state, excluding `.git`, `.humanize`, `.codex`, `.venv`, and `__pycache__`.

| Metric | Value |
|---|---|
| `SKILL.md` size | 5,988 bytes (~6 KiB) |
| `SKILL.md` read time | 0.03 ms |
| Worst-case full-scan file count | 998 |
| Worst-case full-scan apparent bytes | 9.30 MiB |
| Worst-case full-scan time (`Path.rglob` + `stat`) | 25.07 ms |
| Typical skill-engagement read (`SKILL.md` + `references/*.md`, N=3) | 0.14 ms total |

## Interpretation

Claude Code's skill-discovery phase reads **only `SKILL.md`** at boot (and the YAML frontmatter therein). The 6 KiB `SKILL.md` loads in under 1 ms, so the artifact tree has **zero effect on skill boot cost**.

Even the pathological worst case — enumerating every file in the checkout — takes ~25 ms and touches under 10 MiB of apparent bytes. That is well inside Claude Code's typical startup budget and is a one-time cost per session.

When the skill is actually engaged, a typical query reads `SKILL.md` plus one or more files from `references/` (primer, schema, examples) — the measured sum is under 1 ms. Artifact files are only read on explicit `get_page.py --include-code` or `grep_wiki.py --ext` calls, and those are user-driven, not boot-time.

## Recommendation

**No exclude mechanism is needed.** The `artifacts/` tree does not participate in skill-boot scanning:

1. The Claude Code skill manifest is `SKILL.md` only; its `allowed-tools` / `description` fields are the boot payload.
2. Artifact contents reach the model only when the user explicitly asks for them (via `get_page.py --include-code`, `grep_wiki.py --ext`, or a direct `Read` call).
3. No `.claude-skill-ignore` file exists in the official Claude Code feature set; inventing one would be a phantom fix.

## Bound Compliance

The Phase 3 Round 2 working tree was 9.30 MiB across 998 files, under the
25 MiB AC-10 size ceiling with 15.7 MiB of size headroom. AC-10's separate
6,000-file budget applies only to files under `artifacts/`; this historical
memo did not preserve that artifact-file count, so it makes no historical
file-headroom claim. Re-running `scripts/repo_size_check.py` reports the current
working-tree size and enforces the current `artifacts/` file budget; it does
not enforce a whole-working-tree file-count budget.
