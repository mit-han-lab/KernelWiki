#!/usr/bin/env python3
"""Build an exact-text Wiki before/after/reason JSONL report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


FRONTMATTER = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
BODY_LOCATOR = re.compile(r"^body:L(\d+)-L(\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def queue_index(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{source}: queue entry is missing path")
        if path in result:
            raise ValueError(f"{source}: duplicate path {path!r}")
        result[path] = row
    return result


def claim_index(
    rows: list[dict[str, Any]], source: Path
) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for page in rows:
        path = page.get("path")
        for claim in page.get("claims", []):
            claim_id = claim.get("claim_id")
            if not isinstance(path, str) or not isinstance(claim_id, str):
                raise ValueError(f"{source}: claim is missing path or claim_id")
            if claim_id in result:
                raise ValueError(f"{source}: duplicate claim_id {claim_id!r}")
            result[claim_id] = (path, claim)
    return result


def split_page(text: str) -> tuple[str, str]:
    match = FRONTMATTER.match(text)
    if not match:
        raise ValueError("Wiki page is missing YAML frontmatter")
    return match.group(1), text[match.end() :]


def top_level_fields(frontmatter: str) -> list[str]:
    fields: list[str] = []
    for line in frontmatter.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):(?:\s|$)", line)
        if match:
            fields.append(match.group(1))
    return fields


def extract_frontmatter_field(frontmatter: str, field: str) -> str | None:
    lines = frontmatter.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if re.match(rf"^{re.escape(field)}:(?:\s|$)", line):
            start = index
            break
    if start is None:
        return None
    end = start + 1
    while end < len(lines):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:(?:\s|$)", lines[end]):
            break
        end += 1
    return "\n".join(lines[start:end]).strip()


def extract_fields(text: str, fields: list[str]) -> str | None:
    frontmatter, _ = split_page(text)
    parts = [extract_frontmatter_field(frontmatter, field) for field in fields]
    return "\n\n".join(part for part in parts if part) or None


def locator_fields(locator: str, before_text: str, after_text: str) -> list[str]:
    if not locator.startswith("frontmatter:"):
        return []
    description = locator.split(":", 1)[1]
    before_frontmatter, _ = split_page(before_text)
    after_frontmatter, _ = split_page(after_text)
    candidates = top_level_fields(before_frontmatter)
    for field in top_level_fields(after_frontmatter):
        if field not in candidates:
            candidates.append(field)
    fields = [
        field
        for field in candidates
        if re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(field)}(?![A-Za-z0-9_-])",
            description,
        )
    ]
    return fields or candidates


def unit_index(queue_page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {unit["id"]: unit for unit in queue_page.get("claim_units", [])}


def ordered_unit_ids(queue_page: dict[str, Any], requested: list[str]) -> list[str]:
    requested_set = set(requested)
    known = [unit["id"] for unit in queue_page.get("claim_units", [])]
    missing = requested_set.difference(known)
    if missing:
        raise ValueError(
            f"{queue_page['path']}: unknown coverage units {', '.join(sorted(missing))}"
        )
    return [unit_id for unit_id in known if unit_id in requested_set]


def extract_units(
    text: str, queue_page: dict[str, Any], requested: list[str]
) -> tuple[str | None, list[str]]:
    frontmatter, body = split_page(text)
    body_lines = body.splitlines()
    units = unit_index(queue_page)
    metadata_parts: list[str] = []
    metadata_fields: list[str] = []
    body_ranges: list[tuple[int, int]] = []
    for unit_id in ordered_unit_ids(queue_page, requested):
        locator = units[unit_id]["locator"]
        body_match = BODY_LOCATOR.match(locator)
        if body_match:
            start, end = map(int, body_match.groups())
            if body_ranges and start <= body_ranges[-1][1] + 1:
                body_ranges[-1] = (body_ranges[-1][0], max(body_ranges[-1][1], end))
            else:
                body_ranges.append((start, end))
            continue
        if locator.startswith("frontmatter:"):
            field = locator.split(":", 1)[1]
            metadata_fields.append(field)
            part = extract_frontmatter_field(frontmatter, field)
            if part and part not in metadata_parts:
                metadata_parts.append(part)
            continue
        raise ValueError(f"{queue_page['path']}: unsupported locator {locator!r}")
    body_parts = [
        "\n".join(body_lines[start - 1 : end]).strip()
        for start, end in body_ranges
    ]
    parts = metadata_parts + [part for part in body_parts if part]
    return "\n\n".join(parts) or None, metadata_fields


def git_show(repository_root: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        raise ValueError(f"cannot read {path!r} at {revision}: {result.stderr.strip()}")
    return result.stdout


def validate_body(text: str, queue_page: dict[str, Any], label: str) -> None:
    _, body = split_page(text)
    observed = hashlib.sha256(body.encode("utf-8")).hexdigest()
    expected = queue_page.get("body_sha256")
    if observed != expected:
        raise ValueError(
            f"{queue_page['path']}: {label} body hash {observed} != queue {expected}"
        )


def evidence_urls(
    receipt: dict[str, Any],
    original_claims: dict[str, tuple[str, dict[str, Any]]],
    current_claims: dict[str, tuple[str, dict[str, Any]]],
) -> list[str]:
    evidence = list(receipt.get("evidence", []))
    original = original_claims.get(receipt["original_claim_id"])
    if original:
        evidence.extend(original[1].get("evidence", []))
    for claim_id in receipt.get("replacement_claim_ids", []):
        evidence.extend(current_claims[claim_id][1].get("evidence", []))
    urls: list[str] = []
    for item in evidence:
        url = item.get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    verification_dir = repository_root / "verification"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=verification_dir / "before-after-report.jsonl",
    )
    args = parser.parse_args()

    metadata = load_json(verification_dir / "baseline-metadata.json")
    baseline_revision = metadata["repository_commit"]
    baseline_queue = queue_index(
        load_jsonl(verification_dir / "baseline-queue.jsonl"),
        verification_dir / "baseline-queue.jsonl",
    )
    current_queue = queue_index(
        load_jsonl(verification_dir / "current-queue.jsonl"),
        verification_dir / "current-queue.jsonl",
    )
    original_claims = claim_index(
        load_jsonl(verification_dir / "original-report.jsonl"),
        verification_dir / "original-report.jsonl",
    )
    current_claims = claim_index(
        load_jsonl(verification_dir / "verification-report.jsonl"),
        verification_dir / "verification-report.jsonl",
    )
    receipts = [
        receipt
        for receipt in load_jsonl(verification_dir / "remediation-log.jsonl")
        if str(receipt.get("path", "")).startswith("wiki/")
    ]

    baseline_texts: dict[str, str] = {}
    current_texts: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for receipt in receipts:
        path = receipt["path"]
        before_page = baseline_queue[path]
        after_page = current_queue[path]
        if path not in baseline_texts:
            baseline_texts[path] = git_show(repository_root, baseline_revision, path)
        if path not in current_texts:
            current_texts[path] = (repository_root / path).read_text(encoding="utf-8")
        before_text = baseline_texts[path]
        after_text = current_texts[path]
        validate_body(before_text, before_page, "baseline")
        validate_body(after_text, after_page, "current")

        original_unit_ids = list(receipt.get("original_unit_ids", []))
        if original_unit_ids:
            before_statement, metadata_fields = extract_units(
                before_text, before_page, original_unit_ids
            )
            after_unit_ids: list[str] = []
            for claim_id in receipt.get("replacement_claim_ids", []):
                if claim_id not in current_claims:
                    raise ValueError(f"replacement claim {claim_id!r} is missing")
                claim_path, claim = current_claims[claim_id]
                if claim_path != path:
                    raise ValueError(
                        f"{receipt['receipt_id']}: replacement claim belongs to {claim_path}"
                    )
                after_unit_ids.extend(claim.get("covers", []))
            after_statement, _ = extract_units(after_text, after_page, after_unit_ids)
            current_metadata = extract_fields(after_text, metadata_fields)
            if current_metadata:
                after_statement = (
                    f"{current_metadata}\n\n{after_statement}"
                    if after_statement
                    else current_metadata
                )
        else:
            fields = locator_fields(receipt["locator"], before_text, after_text)
            before_statement = extract_fields(before_text, fields)
            after_statement = extract_fields(after_text, fields)

        if before_statement is None:
            raise ValueError(f"{receipt['receipt_id']}: no exact baseline text found")

        records.append(
            {
                "path": path,
                "before": {"statement": before_statement},
                "after": {"statement": after_statement},
                "reason": {
                    "statement": receipt["action_reason"],
                    "urls": evidence_urls(receipt, original_claims, current_claims),
                },
            }
        )

    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        f"wrote records={len(records)} paths={len({record['path'] for record in records})} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
