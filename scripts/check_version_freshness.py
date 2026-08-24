#!/usr/bin/env python3
"""Offline freshness checker for the KernelWiki version-claim registry.

Reads checked-in inputs only:
  - data/tool-versions.yaml      (release-of-record snapshot)
  - data/tool-release-receipt.yaml (dated authoritative-list receipt)
  - data/version-claims.yaml     (per-claim metadata)
  - candidates/<repo>.yaml       (ledger searched_at)
  - data/refresh-cutoff.yaml     (optional; round's refresh cutoff date)
  - plan-phase*.md               (read first 3 lines for supersession marker)

Prints structured warnings; exits 0 in advisory mode (default), 1 in --strict
mode when any warning fires.

This script is required to be NETWORK-FREE. A static-analysis check (in CI)
rejects any future PR that imports HTTP libraries or shells out to git/gh
for remote fetches; see tests/check_freshness_offline.sh for the exact grep.
Do not import or shell out to network tools from here.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CANDIDATES_DIR = REPO_ROOT / "candidates"

DEFAULT_VERSION_STALENESS_DAYS = 180
DEFAULT_LEDGER_STALENESS_DAYS = 30


def load_yaml(path):
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_iso(value):
    """Return a date from a YAML scalar value or raise ValueError. Accepts
    a string, a datetime.date, or a datetime.datetime (PyYAML may decode
    YYYY-MM-DD as date and YYYY-MM-DD HH:MM as datetime)."""
    if hasattr(value, "isoformat") and not isinstance(value, str):
        # PyYAML already gave us a date / datetime.
        if hasattr(value, "date"):
            return value.date()
        return value
    return date.fromisoformat(str(value))


def is_superseded(plan_path):
    """Return True if the first three lines of plan_path contain a
    supersession marker pointing to a successor plan. Honors AC-9 per
    DEC-7 (per-file header mode)."""
    if not plan_path.is_file():
        return False
    head = plan_path.read_text(encoding="utf-8").splitlines()[:3]
    return any(re.search(r"^>\s*Superseded by", line, re.IGNORECASE) for line in head)


def check_tool_versions(tool_versions, today, version_staleness_days):
    """Yield (severity, message) tuples for each tool/release entry."""
    threshold = today - timedelta(days=version_staleness_days)
    for tool in tool_versions.get("tools", []):
        checked_at = tool.get("upstream_checked_at")
        checked_date = None
        if checked_at is None:
            yield "warn", f"tool-versions: {tool['tool']} missing upstream_checked_at"
        else:
            try:
                checked_date = parse_iso(checked_at)
                if checked_date < threshold:
                    age_days = (today - checked_date).days
                    yield "warn", (
                        f"tool-versions: {tool['tool']} upstream_checked_at="
                        f"{checked_date.isoformat()} ({age_days}d ago, >= "
                        f"{version_staleness_days}d threshold)"
                    )
            except ValueError:
                yield "warn", (
                    f"tool-versions: {tool['tool']} has unparseable "
                    f"upstream_checked_at={checked_at!r}"
                )
        for rel in tool.get("releases", []):
            name = rel.get("name", "?")
            ra = rel.get("released_at")
            if ra in (None, "needs-verification"):
                yield "warn", f"tool-versions: {tool['tool']} {name} has released_at={ra!r} (needs-verification)"
                continue
            try:
                ra_date = parse_iso(ra)
            except ValueError:
                yield "warn", f"tool-versions: {tool['tool']} {name} has unparseable released_at={ra!r}"
                continue
            if checked_date is not None and ra_date > checked_date:
                yield "warn", (
                    f"tool-versions: {tool['tool']} {name} released_at="
                    f"{ra_date.isoformat()} is newer than upstream_checked_at="
                    f"{checked_date.isoformat()}"
                )
            if ra_date < threshold:
                age_days = (today - ra_date).days
                yield "info", f"tool-versions: {tool['tool']} {name} released {age_days}d ago (>= {version_staleness_days}d staleness threshold)"


def _registry_release_pairs(tool):
    pairs = set()
    for release in tool.get("releases", []) or []:
        name = str(release.get("name", ""))
        try:
            released_at = parse_iso(release.get("released_at")).isoformat()
        except (TypeError, ValueError):
            released_at = str(release.get("released_at"))
        pairs.add((name, released_at))
    return pairs


def _receipt_release_pairs(tool_name, receipt_tool):
    pairs = set()
    for index, row in enumerate(receipt_tool.get("releases", []) or []):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{tool_name} releases[{index}] must be [name, released_at]")
        pairs.add((str(row[0]), parse_iso(row[1]).isoformat()))
    return pairs


def check_release_receipt(tool_versions, receipt):
    """Require the registry to equal its independently committed dated receipt.

    This offline comparison detects unreceipted additions, omissions, and date
    edits. It intentionally does not claim to discover upstream releases newer
    than the receipt's captured_at date.
    """
    if not isinstance(receipt, dict):
        yield "warn", "tool-release-receipt: missing or invalid receipt"
        return
    receipt_tools = receipt.get("tools")
    if not isinstance(receipt_tools, dict):
        yield "warn", "tool-release-receipt: tools must be a mapping"
        return

    registry_tools = {
        str(tool.get("tool")): tool
        for tool in tool_versions.get("tools", []) or []
        if isinstance(tool, dict) and tool.get("tool")
    }
    for tool_name, registry_tool in sorted(registry_tools.items()):
        receipt_tool = receipt_tools.get(tool_name)
        if not isinstance(receipt_tool, dict):
            yield "warn", f"tool-release-receipt: missing receipt for {tool_name}"
            continue
        registry_checked = str(registry_tool.get("upstream_checked_at", ""))
        receipt_checked = str(receipt_tool.get("upstream_checked_at", ""))
        if registry_checked != receipt_checked:
            yield "warn", (
                f"tool-release-receipt: {tool_name} upstream_checked_at mismatch "
                f"registry={registry_checked!r} receipt={receipt_checked!r}"
            )
        try:
            receipt_pairs = _receipt_release_pairs(tool_name, receipt_tool)
        except (TypeError, ValueError) as exc:
            yield "warn", f"tool-release-receipt: {exc}"
            continue
        registry_pairs = _registry_release_pairs(registry_tool)
        missing = sorted(receipt_pairs - registry_pairs)
        unreceipted = sorted(registry_pairs - receipt_pairs)
        if missing:
            yield "warn", f"tool-release-receipt: {tool_name} registry missing receipted releases {missing}"
        if unreceipted:
            yield "warn", f"tool-release-receipt: {tool_name} has unreceipted releases {unreceipted}"

    extra_receipts = sorted(set(receipt_tools) - set(registry_tools))
    if extra_receipts:
        yield "warn", f"tool-release-receipt: receipts without registry tools {extra_receipts}"


def check_version_claims(claims_data, tool_versions, today, version_staleness_days):
    """Yield (severity, message) tuples for each version-claim registry entry."""
    if claims_data is None:
        return
    threshold = today - timedelta(days=version_staleness_days)
    # Build a release-name set from tool-versions for forward resolution.
    known_releases = set()
    for tool in tool_versions.get("tools", []):
        for rel in tool.get("releases", []):
            known_releases.add(rel.get("name"))
    for claim in claims_data.get("claims", []) or []:
        cid = claim.get("id", "?")
        verified_at = claim.get("last_verified_at")
        verified_release = claim.get("last_verified_release")
        if verified_at is None:
            yield "warn", f"version-claims: {cid} missing last_verified_at"
        else:
            try:
                va_date = parse_iso(verified_at)
                if va_date < threshold:
                    age_days = (today - va_date).days
                    yield "warn", f"version-claims: {cid} last_verified_at={va_date.isoformat()} ({age_days}d ago, >= {version_staleness_days}d threshold)"
            except ValueError:
                yield "warn", f"version-claims: {cid} unparseable last_verified_at={verified_at!r}"
        if verified_release and verified_release not in known_releases:
            yield "warn", f"version-claims: {cid} last_verified_release={verified_release!r} not in data/tool-versions.yaml"


def check_ledger_freshness(today, ledger_staleness_days, refresh_cutoff):
    """Yield (severity, message) tuples for each candidate ledger whose
    searched_at is older than the threshold (or older than the refresh
    cutoff, whichever is tighter)."""
    if not CANDIDATES_DIR.is_dir():
        return
    threshold = today - timedelta(days=ledger_staleness_days)
    for ledger_file in sorted(CANDIDATES_DIR.glob("*.yaml")):
        data = load_yaml(ledger_file)
        if not isinstance(data, dict):
            continue
        sa = data.get("searched_at")
        if sa is None:
            yield "warn", f"ledger {ledger_file.name}: missing searched_at"
            continue
        try:
            sa_date = parse_iso(sa)
        except ValueError:
            yield "warn", f"ledger {ledger_file.name}: unparseable searched_at={sa!r}"
            continue
        if refresh_cutoff and sa_date != refresh_cutoff:
            yield "info", f"ledger {ledger_file.name}: searched_at={sa_date.isoformat()} disagrees with refresh-cutoff {refresh_cutoff.isoformat()}"
        elif not refresh_cutoff and sa_date < threshold:
            age_days = (today - sa_date).days
            yield "warn", f"ledger {ledger_file.name}: searched_at={sa_date.isoformat()} ({age_days}d ago, >= {ledger_staleness_days}d threshold)"


def check_supersession(today):
    """Yield (severity, message) tuples for each plan-phase*.md not
    properly marked superseded. Honors AC-9 per DEC-7."""
    for plan_path in sorted(REPO_ROOT.glob("plan-phase*.md")):
        # plan-phase4.md is the current round and should NOT have a supersession header.
        if plan_path.name == "plan-phase4.md":
            if is_superseded(plan_path):
                yield "warn", f"{plan_path.name}: current-round plan should not be superseded"
            continue
        if not is_superseded(plan_path):
            yield "warn", f"{plan_path.name}: missing supersession header in first 3 lines (AC-9)"


def main():
    parser = argparse.ArgumentParser(description="Offline freshness checker for KernelWiki")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any warning fires (default: advisory exit 0).")
    parser.add_argument("--today", default=None,
                        help="Override today's date for testing (YYYY-MM-DD).")
    parser.add_argument("--version-staleness-days", type=int,
                        default=DEFAULT_VERSION_STALENESS_DAYS)
    parser.add_argument("--ledger-staleness-days", type=int,
                        default=DEFAULT_LEDGER_STALENESS_DAYS)
    args = parser.parse_args()

    if args.today:
        today = date.fromisoformat(args.today)
    else:
        today = date.today()

    tool_versions = load_yaml(DATA_DIR / "tool-versions.yaml") or {}
    release_receipt = load_yaml(DATA_DIR / "tool-release-receipt.yaml")
    claims = load_yaml(DATA_DIR / "version-claims.yaml") or {}
    refresh_cutoff_data = load_yaml(DATA_DIR / "refresh-cutoff.yaml")
    refresh_cutoff = None
    if refresh_cutoff_data and "cutoff_date" in refresh_cutoff_data:
        try:
            refresh_cutoff = parse_iso(refresh_cutoff_data["cutoff_date"])
        except ValueError:
            print(f"WARN: data/refresh-cutoff.yaml::cutoff_date is unparseable: "
                  f"{refresh_cutoff_data['cutoff_date']!r}", file=sys.stderr)

    findings = []
    findings.extend(check_tool_versions(tool_versions, today, args.version_staleness_days))
    findings.extend(check_release_receipt(tool_versions, release_receipt))
    findings.extend(check_version_claims(claims, tool_versions, today, args.version_staleness_days))
    findings.extend(check_ledger_freshness(today, args.ledger_staleness_days, refresh_cutoff))
    findings.extend(check_supersession(today))

    warns = [m for s, m in findings if s == "warn"]
    infos = [m for s, m in findings if s == "info"]

    print(f"check_version_freshness.py — today={today.isoformat()}, "
          f"version-staleness={args.version_staleness_days}d, "
          f"ledger-staleness={args.ledger_staleness_days}d")
    print(f"  {len(warns)} warnings, {len(infos)} info messages")
    for m in warns:
        print(f"  WARN: {m}")
    for m in infos:
        print(f"  INFO: {m}")

    if args.strict and warns:
        print(f"\n--strict: {len(warns)} warning(s) -> exit 1", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
