#!/usr/bin/env python3
"""Refresh candidate ledgers and emit deterministic search-result artifacts.

For each tracked repo, runs `gh search prs` with the keywords already documented in
each ledger's `keywords_used` field, then:

  - Updates `candidates/<repo>.yaml::searched_at` to today's date
    (or the explicit `--cutoff` flag).
  - Writes (or updates) `data/refresh-cutoff.yaml` with `cutoff_date`.
  - Writes (or updates) `data/refresh-search-results.yaml` with per-repo
    `pr_numbers_seen` (sorted) plus `last_pr_date_seen`.

Outputs are byte-stable for identical inputs (deterministic ordering;
tested by re-running the script twice and diffing).

This script makes `gh` calls to GitHub Search API and is therefore NOT
intended to be invoked by `validate.py` or `check_version_freshness.py`
— those scripts must remain offline. It is a refresh-time utility,
similar to `scripts/fetch_pr_diff.py` and `scripts/generate-pr-pages.py`.

Usage:
  python3 scripts/refresh_candidate_ledger.py --cutoff 2026-04-27
  python3 scripts/refresh_candidate_ledger.py --cutoff 2026-04-27 --dry-run
  python3 scripts/refresh_candidate_ledger.py --cutoff 2026-04-27 --repos cutlass,sglang
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

from pr_policy import classify_scope

REPO_ROOT = Path(__file__).parent.parent
CANDIDATES_DIR = REPO_ROOT / "candidates"
DATA_DIR = REPO_ROOT / "data"
REFRESH_CUTOFF_PATH = DATA_DIR / "refresh-cutoff.yaml"
REFRESH_RESULTS_PATH = DATA_DIR / "refresh-search-results.yaml"

REPO_SLUG_TO_FULL = {
    "cutlass": "NVIDIA/cutlass",
    "sglang": "sgl-project/sglang",
    "vllm": "vllm-project/vllm",
    "flashinfer": "flashinfer-ai/flashinfer",
    "pytorch": "pytorch/pytorch",
    "deepgemm": "deepseek-ai/DeepGEMM",
    "flash-attention": "Dao-AILab/flash-attention",
    "tensorrt-llm": "NVIDIA/TensorRT-LLM",
    "cccl-cub": "NVIDIA/cccl",
    "triton": "triton-lang/triton",
    "tilelang": "tile-ai/tilelang",
    "thunderkittens": "HazyResearch/ThunderKittens",
    "tilekernels": "deepseek-ai/TileKernels",
    "quack": "Dao-AILab/quack",
}


import time as _time


def gh_search_prs(repo_full, keywords, cutoff_date, per_keyword_limit=30,
                  sleep_between_kw=2.5):
    """Run `gh search prs` once per keyword and union the results
    (deduplicated by PR number). `gh search` does not support OR-joined
    keyword queries reliably, so the script makes per-keyword calls.

    GitHub Search API rate limit is 30/min. Sleeping ~2.5s between
    keyword queries keeps the script comfortably under that ceiling.

    Filters to merged PRs whose closedAt is on or before cutoff_date."""
    if not keywords:
        return []
    cutoff_str = cutoff_date.isoformat()
    seen = {}
    for kw in keywords:
        cmd = [
            "gh", "search", "prs",
            "--repo", repo_full,
            "--state", "closed",
            "--merged",
            "--limit", str(per_keyword_limit),
            "--json", "number,title,closedAt,url",
            kw,
        ]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"WARN: gh search failed for {repo_full} kw={kw!r}: {e}", file=sys.stderr)
            _time.sleep(sleep_between_kw)
            continue
        rows = json.loads(out)
        for r in rows:
            closed = (r.get("closedAt") or "")[:10]
            if closed and closed <= cutoff_str:
                num = r["number"]
                if num not in seen:
                    seen[num] = {
                        "number": num,
                        "title": r.get("title", ""),
                        "closedAt": closed,
                    }
        _time.sleep(sleep_between_kw)
    return list(seen.values())


def fetch_pr_evidence(repo_full, number):
    """Fetch the authoritative PR description and complete changed-file payload."""
    try:
        pr_raw = subprocess.check_output(
            ["gh", "api", f"repos/{repo_full}/pulls/{number}"], text=True, timeout=60
        )
        pr = json.loads(pr_raw)
        files = []
        for page in range(1, 31):
            raw = subprocess.check_output(
                [
                    "gh", "api",
                    f"repos/{repo_full}/pulls/{number}/files?per_page=100&page={page}",
                ],
                text=True,
                timeout=60,
            )
            payload = json.loads(raw)
            if not isinstance(payload, list):
                return None
            files.extend(payload)
            if len(payload) < 100:
                return pr, files
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return None


def classify_search_hit(repo_full, hit, evidence_fetcher=fetch_pr_evidence):
    """Classify a refresh hit with the same positive-evidence policy as generation."""
    evidence = evidence_fetcher(repo_full, hit["number"])
    if evidence is None:
        return "defer", "authoritative PR evidence unavailable; needs-triage", []
    pr, files = evidence
    decision = classify_scope(pr.get("title") or hit.get("title", ""), pr.get("body") or "", files)
    return (
        "include" if decision.retain else "exclude",
        decision.reason,
        [item.get("filename") for item in files if item.get("filename")],
    )


def update_ledger(ledger_path, cutoff_date, search_results, repo_full=None):
    """Update a candidates/<repo>.yaml file to set searched_at == cutoff_date
    AND merge new PR numbers from search_results into the ledger as
    `decision: defer` rows. Existing rows are not touched.

    AC-5 negative test requires every `pr_numbers_seen` entry to appear in
    the ledger's `prs[*].number` set after refresh. New entries land as
    `decision: defer` to make later triage explicit.
    """
    data = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return
    data["searched_at"] = cutoff_date.isoformat()
    existing_nums = {row.get("number") for row in (data.get("prs") or [])
                     if isinstance(row, dict)}
    new_rows = []
    for hit in search_results:
        num = hit["number"]
        if num in existing_nums:
            continue
        decision, reason, paths = classify_search_hit(repo_full or data.get("repo", ""), hit)
        new_rows.append({
            "number": num,
            "title": hit.get("title", ""),
            "date": hit.get("closedAt", ""),
            "decision": decision,
            "reason": reason,
            "files_changed": paths,
        })
    if new_rows:
        data["prs"] = (data.get("prs") or []) + new_rows
        # Update top-level summary counts
        new_inc = sum(1 for r in data["prs"] if str(r.get("decision","")).lower() == "include")
        new_exc = sum(1 for r in data["prs"] if str(r.get("decision","")).lower() == "exclude")
        new_def = sum(1 for r in data["prs"] if str(r.get("decision","")).lower() == "defer")
        data["total_candidates"] = new_inc + new_exc + new_def
        data["included"] = new_inc
        data["excluded"] = new_exc
        data["deferred"] = new_def
    out = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True)
    ledger_path.write_text(out, encoding="utf-8")
    return len(new_rows)


def write_cutoff(cutoff_date):
    out = (
        "## Round-N refresh cutoff record. Schema: data/schemas.yaml ::\n"
        "## refresh-cutoff. Updated by scripts/refresh_candidate_ledger.py.\n"
        "##\n"
        "## All candidates/*.yaml::searched_at fields must equal this date\n"
        "## after a refresh round (per AC-5).\n"
    )
    out += yaml.safe_dump({"cutoff_date": cutoff_date.isoformat()}, sort_keys=False, default_flow_style=False)
    REFRESH_CUTOFF_PATH.write_text(out, encoding="utf-8")


def write_search_results(cutoff_date, per_repo):
    """Write data/refresh-search-results.yaml byte-stably."""
    repos = []
    for repo_slug in sorted(per_repo.keys()):
        rows = per_repo[repo_slug]
        pr_numbers = sorted([r["number"] for r in rows])
        last_pr_date = max([r["closedAt"] for r in rows], default="")
        repos.append({
            "repo_slug": repo_slug,
            "searched_at": cutoff_date.isoformat(),
            "cutoff_date_used": cutoff_date.isoformat(),
            "pr_numbers_seen": pr_numbers,
            "last_pr_date_seen": last_pr_date,
        })
    payload = {"cutoff_date": cutoff_date.isoformat(), "repos": repos}
    out = (
        "## AC-5 refresh search-results record. Schema: data/schemas.yaml ::\n"
        "## refresh-search-results. Generated by\n"
        "## scripts/refresh_candidate_ledger.py. Byte-stable for identical\n"
        "## query inputs (pr_numbers_seen sorted ascending; repos keyed by\n"
        "## repo_slug ascending).\n"
    )
    out += yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True)
    REFRESH_RESULTS_PATH.write_text(out, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True,
                        help="Refresh cutoff date (YYYY-MM-DD).")
    parser.add_argument("--repos", default=None,
                        help="Comma-separated repo slugs to refresh "
                             "(default: all from REPO_SLUG_TO_FULL).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write any files; print the search results.")
    args = parser.parse_args()

    cutoff_date = date.fromisoformat(args.cutoff)
    repo_slugs = (args.repos.split(",") if args.repos
                  else list(REPO_SLUG_TO_FULL.keys()))

    per_repo = {}
    for slug in repo_slugs:
        repo_full = REPO_SLUG_TO_FULL.get(slug)
        if not repo_full:
            print(f"WARN: unknown repo slug {slug!r}", file=sys.stderr)
            continue
        ledger_path = CANDIDATES_DIR / f"{slug}.yaml"
        if not ledger_path.is_file():
            print(f"WARN: no ledger for {slug} at {ledger_path}", file=sys.stderr)
            per_repo[slug] = []
            continue
        ledger_data = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
        keywords = ledger_data.get("keywords_used") or []
        print(f"  {slug}: searching {repo_full} with {len(keywords)} keywords...")
        rows = gh_search_prs(repo_full, keywords, cutoff_date)
        per_repo[slug] = rows
        print(f"    -> {len(rows)} merged PRs found within cutoff window")
        if not args.dry_run:
            added = update_ledger(ledger_path, cutoff_date, rows, repo_full)
            if added:
                print(f"    -> +{added} new defer-rows merged into ledger")

    if args.dry_run:
        print("\nDry-run mode; no files written.")
        return

    # Merge with existing data/refresh-search-results.yaml so partial
    # refreshes (e.g. --repos cutlass,sglang) accumulate instead of
    # overwriting other repos' results.
    if REFRESH_RESULTS_PATH.is_file():
        existing = yaml.safe_load(REFRESH_RESULTS_PATH.read_text(encoding="utf-8")) or {}
        for row in existing.get("repos", []) or []:
            slug = row.get("repo_slug")
            if slug and slug not in per_repo:
                per_repo[slug] = [
                    {"number": n, "title": "", "closedAt": row.get("last_pr_date_seen", "")}
                    for n in row.get("pr_numbers_seen", []) or []
                ]

    write_cutoff(cutoff_date)
    write_search_results(cutoff_date, per_repo)
    print(f"\nWrote {REFRESH_CUTOFF_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REFRESH_RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
