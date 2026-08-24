#!/usr/bin/env python3
"""Text search across wiki bodies and source PR descriptions.

Usage:
    grep_wiki.py "tcgen05" --only wiki
    grep_wiki.py "two-CTA" --only wiki
    grep_wiki.py "ping-pong" --context 3
    grep_wiki.py "nvfp4 block_scale" --any     # match if ANY word appears

Returns matching lines with file path, line number, and N context lines.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402


def iter_files(scope, exts=None):
    """Yield file paths under the given scope.

    Default: markdown files under wiki/sources. If `exts` is provided (a set
    of lowercase extensions including the leading dot, e.g. {'.cu', '.cuh'}),
    those extensions are ALSO searched across wiki/sources AND artifacts/.

    R32: when scope=='artifacts' the default search set includes common
    source-code extensions in addition to `.md`. Artifact bundles rarely
    contain markdown — the hits users expect live in `.cu` / `.py` /
    `.ptx` / `.patch` etc. files. Without this default, `--only artifacts`
    without `--ext` reports `No matches` even when the code bundles
    contain the pattern.
    """
    dirs = {
        "wiki": ["wiki"],
        "sources": ["sources"],
        "all": ["wiki", "sources"],
        "artifacts": ["artifacts"],
    }
    sub_list = dirs.get(scope, ["wiki", "sources"])
    # When --ext is used, also search artifacts/ unless scope explicitly excludes it
    if exts and "artifacts" not in sub_list and scope != "wiki" and scope != "sources":
        sub_list = sub_list + ["artifacts"]

    if scope == "artifacts" and not exts:
        # R32 default for --only artifacts: match every source-file
        # extension the Phase 3 emitters actually produce. Keeps `.md`
        # so MANIFEST/README-style notes are still searchable. `.sh`
        # added in R33 for shell fences emitted via EXT_MAP
        # bash/shell/sh -> .sh.
        search_exts = {
            ".md",
            ".cu", ".cuh", ".ptx",
            ".cpp", ".cxx", ".cc", ".c",
            ".h", ".hpp", ".hxx", ".inl",
            ".py", ".pyx",
            ".patch",
            ".txt",
            ".sh",
            ".yaml", ".yml", ".json",
        }
    else:
        search_exts = {".md"} | (exts or set())

    for sub in sub_list:
        base = WIKI_ROOT / sub
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() in search_exts:
                yield f


def grep_file(path, compiled_patterns, context, any_match):
    """Search a single file for the pattern(s). Returns list of (line_no, context_text) tuples."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    results = []
    for i, line in enumerate(lines):
        if any_match:
            matched = any(p.search(line) for p in compiled_patterns)
        else:
            matched = all(p.search(line) for p in compiled_patterns)
        if matched:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            snippet = "\n".join(
                f"{j+1}{'→' if j == i else ':'} {lines[j]}"
                for j in range(start, end)
            )
            results.append((i + 1, snippet))
    return results


def main():
    parser = argparse.ArgumentParser(description="Text search across Blackwell kernel wiki")
    parser.add_argument("patterns", nargs="+", help="Search pattern(s) — all must match a line unless --any is used")
    parser.add_argument("--only", choices=["wiki", "sources", "all", "artifacts"], default="all",
                        help="Restrict search scope (default: all)")
    parser.add_argument("--context", type=int, default=1, help="Context lines around each match (default 1)")
    parser.add_argument("--any", action="store_true", help="Match if ANY pattern matches a line (default: all must match)")
    parser.add_argument("--limit", type=int, default=20, help="Max files reported (default 20)")
    parser.add_argument("--files-only", action="store_true", help="Print only matching file paths")
    parser.add_argument("--ext", default=None, help="Comma-separated extra extensions to search (without dots), e.g. 'cu,cuh,ptx,py'; auto-expands scope to include artifacts/")
    args = parser.parse_args()

    compiled = []
    for p in args.patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            print(f"ERROR: invalid regex {p!r}: {e}", file=sys.stderr)
            print("       Hint: escape special chars ([](){}.*+?^$|\\) or quote the pattern.", file=sys.stderr)
            sys.exit(2)

    ext_set = None
    if args.ext:
        ext_set = {"." + e.strip().lstrip(".").lower() for e in args.ext.split(",") if e.strip()}

    matched_files = []
    for path in iter_files(args.only, exts=ext_set):
        hits = grep_file(path, compiled, args.context, args.any)
        if hits:
            matched_files.append((path, hits))

    matched_files = matched_files[:args.limit]

    if args.files_only:
        for path, _ in matched_files:
            print(path.relative_to(WIKI_ROOT))
        return

    if not matched_files:
        print("No matches.")
        return

    print(f"# {len(matched_files)} file(s) match")
    for path, hits in matched_files:
        print()
        print(f"## {path.relative_to(WIKI_ROOT)}  ({len(hits)} match{'es' if len(hits) != 1 else ''})")
        for line_no, snippet in hits[:5]:  # Cap per-file to 5 hits
            print(f"```")
            print(snippet)
            print(f"```")


if __name__ == "__main__":
    main()
