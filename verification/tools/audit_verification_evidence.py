#!/usr/bin/env python3
"""Audit verification evidence for stable paths, exact commands, and live URLs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from html import unescape
from pathlib import Path
from urllib.parse import urldefrag


ROOT = Path(__file__).resolve().parents[2]
VERIFICATION = ROOT / "verification"
EPHEMERAL_PREFIX = "/" + "tmp" + "/"
PLACEHOLDER = re.compile(
    r"<(?:displayed|each|all|exact|evaluate|compare|parse|resolve|scope|wiki|official|candidate|four|ACL|AST|the\b)[^>]*>",
    re.IGNORECASE,
)


def walk(value: object):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reviewed_claims(lane: str) -> list[dict]:
    claims = []
    for path in sorted((VERIFICATION / "reviewed" / lane).glob("*.json")):
        claims.extend(json.loads(path.read_text()).get("claims", []))
    return claims


def evidence_items(record: dict) -> list[dict]:
    return [item for item in record.get("evidence", []) if isinstance(item, dict)]


def string_values(record: dict) -> list[str]:
    return [value for value in walk(record) if isinstance(value, str)]


def direct_kinds(record: dict) -> set[str]:
    kinds = set()
    for item in evidence_items(record):
        for kind in ("path", "url", "command"):
            if isinstance(item.get(kind), str):
                kinds.add(kind)
    return kinds


def curl(url: str, include_body: bool) -> tuple[int, str, str]:
    command = [
        "curl",
        "-L",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "180",
        "-A",
        "Mozilla/5.0 KernelWiki evidence audit",
        "-sS",
        "-w",
        "\n__KERNELWIKI_STATUS__%{http_code}\t%{url_effective}",
    ]
    if not include_body:
        command.extend(["-o", "/dev/null"])
    command.append(url)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        return 0, url, completed.stderr.strip()
    body, marker, trailer = completed.stdout.rpartition("\n__KERNELWIKI_STATUS__")
    if not marker:
        return 0, url, "missing curl status trailer"
    status, final_url = trailer.split("\t", 1)
    return int(status), final_url, body


def html_ids(body: str) -> set[str]:
    return {
        unescape(match)
        for match in re.findall(r"\bid=[\"']([^\"']+)[\"']", body, flags=re.IGNORECASE)
    }


def network_audit(urls: list[str]) -> tuple[list[dict], list[dict]]:
    failures = []
    fragment_failures = []
    page_cache: dict[str, tuple[int, str, str]] = {}
    for url in urls:
        base, fragment = urldefrag(url)
        check_fragment = bool(fragment) and "docs.nvidia.com" in base
        if base not in page_cache:
            page_cache[base] = curl(base, include_body=check_fragment)
        status, final_url, body = page_cache[base]
        if not 200 <= status < 400:
            failure = {"url": url, "status": status, "final_url": final_url}
            if status == 0:
                failure["error"] = body
            failures.append(failure)
            continue
        if check_fragment:
            if not body:
                status, final_url, body = curl(base, include_body=True)
                page_cache[base] = status, final_url, body
            if fragment not in html_ids(body):
                fragment_failures.append(
                    {"url": url, "fragment": fragment, "status": status, "final_url": final_url}
                )
    return failures, fragment_failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true", help="fetch every unique current evidence URL")
    args = parser.parse_args()

    receipts = load_jsonl(VERIFICATION / "remediation-log.jsonl")
    claims = reviewed_claims("current")
    original_claims = reviewed_claims("original")
    receipt_tmp = [
        row["receipt_id"]
        for row in receipts
        if any(EPHEMERAL_PREFIX in value for value in string_values(row))
    ]
    claim_tmp = [
        row["claim_id"]
        for row in claims
        if any(EPHEMERAL_PREFIX in value for value in string_values(row))
    ]
    original_tmp = [
        row["claim_id"]
        for row in original_claims
        if any(EPHEMERAL_PREFIX in value for value in string_values(row))
    ]

    placeholder_commands = []
    missing_results = []
    local_paths = []
    missing_local_paths = []
    for group, records, id_key in (
        ("receipt", receipts, "receipt_id"),
        ("current", claims, "claim_id"),
        ("original", original_claims, "claim_id"),
    ):
        for row in records:
            for item in evidence_items(row):
                command = item.get("command")
                if isinstance(command, str):
                    if PLACEHOLDER.search(command):
                        placeholder_commands.append({"group": group, "id": row[id_key], "command": command})
                    if "result" not in item:
                        missing_results.append({"group": group, "id": row[id_key], "command": command})
                path = item.get("path")
                if isinstance(path, str) and not Path(path).is_absolute():
                    local_paths.append(path)
                    if not (ROOT / path).exists():
                        missing_local_paths.append({"group": group, "id": row[id_key], "path": path})

    urls = sorted(
        {
            item["url"]
            for row in claims
            for item in evidence_items(row)
            if isinstance(item.get("url"), str) and item["url"].startswith(("http://", "https://"))
        }
    )
    url_failures: list[dict] = []
    fragment_failures: list[dict] = []
    if args.network:
        url_failures, fragment_failures = network_audit(urls)

    direct_kind_counts = Counter(
        "+".join(sorted(direct_kinds(row))) or "none"
        for row in [*receipts, *claims, *original_claims]
    )
    report = {
        "receipts": len(receipts),
        "current_claims": len(claims),
        "original_claims": len(original_claims),
        "receipt_absolute_tmp_records": len(receipt_tmp),
        "current_absolute_tmp_claims": len(claim_tmp),
        "original_absolute_tmp_claims": len(original_tmp),
        "placeholder_commands": len(placeholder_commands),
        "commands_missing_result": len(missing_results),
        "repo_local_evidence_references": len(local_paths),
        "missing_repo_local_evidence": len(missing_local_paths),
        "current_unique_evidence_urls": len(urls),
        "network_checked": args.network,
        "url_failures": len(url_failures),
        "nvidia_fragment_failures": len(fragment_failures),
        "direct_evidence_kind_counts": dict(sorted(direct_kind_counts.items())),
        "details": {
            "receipt_absolute_tmp_records": receipt_tmp,
            "current_absolute_tmp_claims": claim_tmp,
            "original_absolute_tmp_claims": original_tmp,
            "placeholder_commands": placeholder_commands,
            "commands_missing_result": missing_results,
            "missing_repo_local_evidence": missing_local_paths,
            "url_failures": url_failures,
            "nvidia_fragment_failures": fragment_failures,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    high_issue_count = sum(
        (
            len(receipt_tmp),
            len(claim_tmp),
            len(original_tmp),
            len(placeholder_commands),
            len(missing_results),
            len(missing_local_paths),
            len(url_failures),
            len(fragment_failures),
        )
    )
    return 1 if high_issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
