#!/usr/bin/env python3
"""Make verification evidence paths repository-local and reproducible.

The verifier reports historically referred to ephemeral checkout paths.  This
tool copies only the files and subtrees that are named by evidence ``path``
fields, omits nested Git metadata, records content digests, and rewrites those
references to ``verification/evidence/local-snapshots``.

Two complete CUTLASS checkout roots were used only for repository-wide token
searches.  Copying them would add roughly 200 MiB, so those two evidence items
are changed to the immutable upstream commit tree instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFICATION_ROOT = REPO_ROOT / "verification"
SNAPSHOT_ROOT = VERIFICATION_ROOT / "evidence" / "local-snapshots"
MANIFEST_PATH = VERIFICATION_ROOT / "evidence" / "MANIFEST.json"
EPHEMERAL_PREFIX = "/" + "tmp" + "/"
MAX_COPY_BYTES = 25 * 1024 * 1024

CUTLASS_REVISION = "e406c186f510a15091cce01f782020ceb7ba8eb5"
CUTLASS_TREE_URL = (
    "https://github.com/NVIDIA/cutlass/tree/" + CUTLASS_REVISION
)
OVERSIZED_ROOTS = {
    EPHEMERAL_PREFIX + "kernelwiki-cutlass-dRC5i4/cutlass": CUTLASS_TREE_URL,
    EPHEMERAL_PREFIX + "kernelwiki-cutlass-4.5.0": CUTLASS_TREE_URL,
}
SOURCE_ALTERNATES = {
    EPHEMERAL_PREFIX + "kernelwiki-cutlass-4.5.0/README.md": (
        EPHEMERAL_PREFIX + "kernelwiki-cutlass-dRC5i4/cutlass/README.md"
    ),
}
IGNORED_NAMES = {".git", "__pycache__", ".pytest_cache"}


def verification_json_files() -> list[Path]:
    return sorted(
        path
        for path in VERIFICATION_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


def load_objects(path: Path) -> Iterable[Any]:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
    else:
        yield json.loads(path.read_text(encoding="utf-8"))


def evidence_paths(value: Any) -> Iterable[tuple[str, str | None]]:
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path.startswith(EPHEMERAL_PREFIX):
            revision = value.get("revision")
            yield path, revision if isinstance(revision, str) else None
        for child in value.values():
            yield from evidence_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_paths(child)


def source_path(recorded_path: str) -> Path:
    selected = SOURCE_ALTERNATES.get(recorded_path, recorded_path)
    return Path(selected)


def destination_path(recorded_path: str) -> Path:
    relative = recorded_path.removeprefix(EPHEMERAL_PREFIX)
    return SNAPSHOT_ROOT / relative


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        relative_parts = candidate.relative_to(path).parts
        if any(part in IGNORED_NAMES for part in relative_parts):
            continue
        yield candidate


def selected_file_union(paths: Iterable[str]) -> set[Path]:
    files: set[Path] = set()
    for recorded_path in paths:
        if recorded_path in OVERSIZED_ROOTS:
            continue
        source = source_path(recorded_path)
        if not source.exists():
            raise FileNotFoundError(
                f"evidence source is unavailable: {recorded_path}"
            )
        files.update(iter_files(source))
    return files


def copy_evidence_path(recorded_path: str) -> None:
    if recorded_path in OVERSIZED_ROOTS:
        return
    source = source_path(recorded_path)
    destination = destination_path(recorded_path)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*sorted(IGNORED_NAMES)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_content_stats(path: Path) -> tuple[str, int, int]:
    files = list(iter_files(path))
    total_bytes = sum(candidate.stat().st_size for candidate in files)
    if path.is_file():
        return sha256_file(path), 1, total_bytes

    tree_digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(sha256_file(candidate)))
    return tree_digest.hexdigest(), len(files), total_bytes


def content_record(recorded_path: str, revisions: set[str]) -> dict[str, Any]:
    if recorded_path in OVERSIZED_ROOTS:
        return {
            "source_key": recorded_path.removeprefix(EPHEMERAL_PREFIX),
            "resolution": "immutable-upstream-tree",
            "url": OVERSIZED_ROOTS[recorded_path],
            "revisions": sorted(revisions),
            "reason": "Complete checkout omitted by the repository size guard.",
        }

    source = source_path(recorded_path)
    destination = destination_path(recorded_path)
    digest, file_count, total_bytes = local_content_stats(source)
    kind = "file" if source.is_file() else "directory"

    record: dict[str, Any] = {
        "source_key": recorded_path.removeprefix(EPHEMERAL_PREFIX),
        "path": destination.relative_to(REPO_ROOT).as_posix(),
        "kind": kind,
        "files": file_count,
        "bytes": total_bytes,
        "sha256": digest,
        "revisions": sorted(revisions),
    }
    alternate = SOURCE_ALTERNATES.get(recorded_path)
    if alternate:
        record["restored_from_equivalent_checkout"] = alternate.removeprefix(
            EPHEMERAL_PREFIX
        )
    return record


def rewrite_verification_files(paths: Iterable[Path]) -> int:
    changed = 0
    stable_prefix = "verification/evidence/local-snapshots/"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        updated = text
        for oversized_path, url in OVERSIZED_ROOTS.items():
            pattern = re.compile(
                r'"path"(\s*:\s*)"' + re.escape(oversized_path) + r'"'
            )
            updated = pattern.sub(
                lambda match: '"url"' + match.group(1) + json.dumps(url),
                updated,
            )
        updated = updated.replace(EPHEMERAL_PREFIX, stable_prefix)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def migrate() -> None:
    json_files = verification_json_files()
    revisions_by_path: dict[str, set[str]] = {}
    for json_file in json_files:
        for obj in load_objects(json_file):
            for recorded_path, revision in evidence_paths(obj):
                revisions_by_path.setdefault(recorded_path, set())
                if revision:
                    revisions_by_path[recorded_path].add(revision)

    if not revisions_by_path:
        print("No ephemeral evidence paths found; nothing to migrate.")
        return

    files = selected_file_union(revisions_by_path)
    copy_bytes = sum(path.stat().st_size for path in files)
    if copy_bytes > MAX_COPY_BYTES:
        raise RuntimeError(
            f"curated evidence copy is {copy_bytes} bytes, over the "
            f"{MAX_COPY_BYTES}-byte guard"
        )

    for recorded_path in sorted(revisions_by_path):
        copy_evidence_path(recorded_path)

    records = [
        content_record(path, revisions_by_path[path])
        for path in sorted(revisions_by_path)
    ]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "kernelwiki-local-evidence-v1",
        "copy_policy": (
            "Exact evidence files and named subtrees are copied without nested "
            "Git metadata; oversized whole-checkout searches use immutable "
            "upstream commit trees."
        ),
        "copied_unique_files": len(files),
        "copied_bytes": copy_bytes,
        "entries": records,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    changed = rewrite_verification_files(json_files)
    print(
        f"Migrated {len(revisions_by_path)} evidence paths; copied "
        f"{len(files)} unique files ({copy_bytes} bytes); rewrote "
        f"{changed} verification files."
    )


def check() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest.get("entries", []):
        recorded = entry.get("path")
        if not isinstance(recorded, str):
            continue
        path = REPO_ROOT / recorded
        if not path.exists():
            raise FileNotFoundError(f"manifest target is missing: {recorded}")
        digest, file_count, total_bytes = local_content_stats(path)
        expected = (entry.get("sha256"), entry.get("files"), entry.get("bytes"))
        actual = (digest, file_count, total_bytes)
        if actual != expected:
            raise RuntimeError(
                f"manifest mismatch for {recorded}: expected={expected}, "
                f"actual={actual}"
            )
        checked += 1

    for json_file in verification_json_files():
        text = json_file.read_text(encoding="utf-8")
        if EPHEMERAL_PREFIX in text:
            raise RuntimeError(f"ephemeral path remains in {json_file}")

    print(
        f"Validated {checked} copied evidence entries and found no ephemeral "
        "paths in structured verification records."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify copied content digests and structured evidence paths",
    )
    args = parser.parse_args()
    if args.check:
        check()
    else:
        migrate()


if __name__ == "__main__":
    main()
