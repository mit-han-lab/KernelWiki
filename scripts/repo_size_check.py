#!/usr/bin/env python3
"""Report repo working-tree size and file count; fail if active ceiling exceeded.

Exclusion list: .git, .humanize, .codex, .cache, .worklog, __pycache__, node_modules.

`.worklog/` is explicitly reserved for ignored audit scratch state. Counting
downloaded evidence there would make the repository budget depend on a local
verification cache rather than reviewable repository content.

Reads data/phase3-size-budget.yaml when present to get the active ceiling.
If the budget file is absent, this script reports numbers but does not fail
(intermediate commits before the pilot don't need the size gate).

Exit codes:
  0 — within ceiling (or no budget file present)
  1 — active ceiling exceeded or file-count budget exceeded
  2 — invocation error

Usage:
  scripts/repo_size_check.py             # print and assert
  scripts/repo_size_check.py --ceiling-mib 60   # override active ceiling
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO_ROOT / "data" / "phase3-size-budget.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

EXCLUDE_TOP = {".git", ".humanize", ".codex", ".cache", ".worklog", "node_modules", ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDE_ANY = {"__pycache__"}

# Hard file-count budget (plan AC-10): 6000 under artifacts/
FILE_COUNT_BUDGET = 6000


def working_tree_bytes():
    total = 0
    count = 0
    for p in REPO_ROOT.rglob("*"):
        # Skip excluded top-level directories
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] in EXCLUDE_TOP:
            continue
        if any(part in EXCLUDE_ANY for part in parts):
            continue
        if p.is_file():
            try:
                total += p.stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def artifacts_file_count():
    if not ARTIFACTS_DIR.is_dir():
        return 0
    c = 0
    for p in ARTIFACTS_DIR.rglob("*"):
        if p.is_file():
            c += 1
    return c


def load_budget():
    if not BUDGET_PATH.is_file():
        return None
    try:
        return yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ceiling-mib", type=int, default=None,
                        help="Override the working-tree size ceiling (MiB)")
    args = parser.parse_args()

    total, total_files = working_tree_bytes()
    artifacts_files = artifacts_file_count()
    mib = total / (1024 * 1024)

    budget = load_budget()
    if args.ceiling_mib is not None:
        ceiling_mib = args.ceiling_mib
        ceiling_src = "CLI override"
    elif budget and "active_ceiling_mib" in budget:
        ceiling_mib = budget["active_ceiling_mib"]
        ceiling_src = "data/phase3-size-budget.yaml"
    else:
        ceiling_mib = None
        ceiling_src = None

    print(f"Working tree: {mib:.2f} MiB across {total_files} files "
          f"(artifacts/: {artifacts_files} files)")

    failed = False
    if ceiling_mib is not None:
        print(f"Active ceiling: {ceiling_mib} MiB (source: {ceiling_src})")
        if mib > ceiling_mib:
            print(f"FAIL: working tree {mib:.2f} MiB exceeds ceiling {ceiling_mib} MiB",
                  file=sys.stderr)
            failed = True
    else:
        print("Active ceiling: (no budget file present; skip size gate)")

    if artifacts_files > FILE_COUNT_BUDGET:
        print(f"FAIL: artifacts/ has {artifacts_files} files (> {FILE_COUNT_BUDGET} budget)",
              file=sys.stderr)
        failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
