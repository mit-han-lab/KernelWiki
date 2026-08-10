#!/usr/bin/env python3
"""Replace one JSONL report entry with a reviewed JSON object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("entry", type=Path)
    args = parser.parse_args()

    replacement = json.loads(args.entry.read_text(encoding="utf-8"))
    path = replacement.get("path")
    if not isinstance(path, str) or not path:
        parser.error("entry must contain a non-empty path")

    rows = load_jsonl(args.report)
    matches = [index for index, row in enumerate(rows) if row.get("path") == path]
    if len(matches) != 1:
        parser.error(f"expected exactly one report entry for {path!r}, found {len(matches)}")

    rows[matches[0]] = replacement
    args.report.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"replaced path={path} report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
