#!/usr/bin/env python3
"""Print current corpus counts."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

from _wiki_root import WIKI_ROOT  # noqa: E402


def load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def count_files(*parts: str, pattern: str = "*.md") -> int:
    root = WIKI_ROOT.joinpath(*parts)
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def count_top_level_files(*parts: str, pattern: str = "*.yaml") -> int:
    root = WIKI_ROOT.joinpath(*parts)
    if not root.exists():
        return 0
    return sum(1 for p in root.glob(pattern) if p.is_file())


def count_prs_by_repo() -> Counter:
    prs_root = WIKI_ROOT / "sources" / "prs"
    counts: Counter[str] = Counter()
    if not prs_root.exists():
        return counts
    for repo_dir in prs_root.iterdir():
        if repo_dir.is_dir():
            counts[repo_dir.name] = sum(1 for _ in repo_dir.glob("PR-*.md"))
    return counts


def count_artifact_modes() -> Counter:
    counts: Counter[str] = Counter()
    for provenance in (WIKI_ROOT / "artifacts").rglob("PROVENANCE.yaml"):
        data = load_yaml(provenance)
        mode = data.get("asset_mode")
        if isinstance(mode, str) and mode:
            counts[mode] += 1
        else:
            counts["unknown"] += 1
    return counts


def main() -> int:
    pr_counts = count_prs_by_repo()
    artifact_modes = count_artifact_modes()
    source_pages = count_files("sources", pattern="*.md")
    wiki_pages = count_files("wiki", pattern="*.md")

    print("Corpus counts:")
    print(f"  Indexed source/wiki pages: {source_pages + wiki_pages}")
    print(f"  Source PR pages: {sum(pr_counts.values())}")
    print(f"  Wiki synthesis pages: {wiki_pages}")
    print(f"  Blog summaries: {count_files('sources', 'blogs', pattern='*.md')}")
    print(f"  Doc summaries: {count_files('sources', 'docs', pattern='*.md')}")
    print(f"  Contest pages: {count_files('sources', 'contests', pattern='*.md')}")
    print(f"  Candidate ledgers: {count_top_level_files('candidates', pattern='*.yaml')}")
    print(f"  Artifact bundles: {sum(artifact_modes.values())}")
    print(f"  Query indices: {count_top_level_files('queries', pattern='*.md')}")

    if pr_counts:
        print()
        print("Source PR pages by repo:")
        for repo, count in sorted(pr_counts.items(), key=lambda item: item[0].lower()):
            print(f"  {repo}: {count}")

    if artifact_modes:
        print()
        print("Artifact bundles by mode:")
        for mode, count in sorted(artifact_modes.items()):
            print(f"  {mode}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
