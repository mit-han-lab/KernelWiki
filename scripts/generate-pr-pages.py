#!/usr/bin/env python3
"""Generate PR source pages from candidate ledger YAML files.

Reads candidates/{repo}.yaml, fetches PR details from GitHub API,
and creates sources/prs/{repo_slug}/PR-{number}.md for each included PR.

Usage:
    python3 scripts/generate-pr-pages.py candidates/cutlass.yaml
    python3 scripts/generate-pr-pages.py --all
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import yaml
from datetime import date
from pathlib import Path

from pr_policy import (
    BODY_CONTRACT,
    classify_scope,
    derive_architectures,
    derive_metadata,
    normalize_files,
    render_generated_body,
    upstream_files_sha256,
)

REPO_ROOT = Path(__file__).parent.parent
TAGS_PATH = REPO_ROOT / "data" / "tags.yaml"
PR_PAGE_SKIPPED_PATH = REPO_ROOT / "data" / "pr-page-skipped.yaml"

# Load controlled vocabulary
with open(TAGS_PATH, encoding="utf-8") as f:
    TAGS_DATA = yaml.safe_load(f)

VALID_HW = set(TAGS_DATA.get("hardware_features", []))
VALID_TECHNIQUES = set(TAGS_DATA.get("techniques", []))
VALID_KT = set(TAGS_DATA.get("kernel_types", []))
VALID_LANGS = set(TAGS_DATA.get("languages", []))
ALL_TAGS = VALID_HW | VALID_TECHNIQUES | VALID_KT | VALID_LANGS

# Keyword -> tag mappings for auto-tagging from PR titles/files
KW_TO_TAGS = {
    "tcgen05": "tcgen05", "tmem": "tmem", "tma": "tma", "clc": "clc",
    "nvfp4": "nvfp4", "fp8": "fp8", "fp4": "fp4", "block_scale": "block-scale",
    "block-scale": "block-scale", "mbarrier": "mbarrier", "wgmma": "wgmma",
    "2sm": "2sm-cooperative", "2cta": "2sm-cooperative", "cta_group": "2sm-cooperative",
}
KW_TO_KT = {
    "gemm": "gemm", "attention": "attention", "moe": "moe", "fmha": "flash-attention",
    "flash_attention": "flash-attention", "flash-attention": "flash-attention",
    "mla": "mla", "gemv": "gemv", "grouped_gemm": "grouped-gemm",
    "grouped-gemm": "grouped-gemm", "decode": "decode", "prefill": "prefill",
    "sparse_attention": "sparse-attention", "sparse-attention": "sparse-attention",
    "gated_delta": "gated-delta-net", "quantiz": "quantization",
    "topk": "topk", "top-k": "topk", "scan": "scan", "reduce": "reduction",
    "reduction": "reduction", "sort": "sort", "radix": "sort",
}
KW_TO_TECH = {
    "warp_special": "warp-specialization", "warp-specialization": "warp-specialization",
    "persistent": "persistent-kernel", "swizzl": "swizzling",
    "pipeline": "pipeline-stages", "epilogue": "epilogue-fusion",
    "tile_schedul": "tile-scheduling", "double_buffer": "double-buffering",
    "fusion": "kernel-fusion", "fused": "kernel-fusion",
    "topk": "top-k-selection", "top-k": "top-k-selection",
    "scan": "parallel-scan", "stream-k": "stream-k", "streamk": "stream-k",
}
KW_TO_LANG = {
    ".cu": "cuda-cpp", ".cuh": "cuda-cpp", "cuda": "cuda-cpp",
    "cute_dsl": "cute-dsl", "cute-dsl": "cute-dsl", "cutedsl": "cute-dsl",
    "triton": "triton", ".ptx": "ptx", "ptx": "ptx",
    "python": "python", "tilelang": "tilelang",
}

EXCLUDE_TITLE_PATTERNS = [
    r'\[ci\]', r'\[doc\]', r'\[docs\]', r'bump', r'typo', r'format',
    r'lint', r'\bnit\b', r'revert', r'readme', r'changelog', r'pre-commit',
    r'ruff', r'deprecat', r'release\s+note', r'committers\.md',
]


import subprocess


def gh_api(endpoint):
    """Call GitHub API via authenticated gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def fetch_pr(repo, number):
    """Fetch PR details via gh CLI."""
    return gh_api(f"repos/{repo}/pulls/{number}")


def fetch_pr_files(repo, number):
    """Fetch complete changed-file metadata, including available patches."""
    files = []
    for page in range(1, 31):
        data = gh_api(f"repos/{repo}/pulls/{number}/files?per_page=100&page={page}")
        if data is None:
            return []
        if not isinstance(data, list):
            return []
        files.extend(data)
        if len(data) < 100:
            return files
    raise RuntimeError(f"PR {repo}#{number} exceeds GitHub's 3000 changed-file API limit")


def is_kernel_related(title, files, body=""):
    """Compatibility wrapper around the shared evidence-based scope policy."""
    decision = classify_scope(title, body, files)
    return decision.retain, decision.reason


def auto_tag(title, files, body="", scope=None):
    """Compatibility wrapper returning shared positive-evidence metadata."""
    metadata = derive_metadata(title, body, files, scope)
    return (
        metadata["tags"],
        metadata["hardware_features"],
        metadata["kernel_types"],
        metadata["techniques"],
        metadata["languages"],
    )


def generate_page(repo, pr_data, files, inclusion_reason, captured_at, preserved=None):
    """Generate markdown page content for a PR."""
    repo_slug = repo.split("/")[1]
    number = pr_data["number"]
    title = pr_data["title"]
    author = pr_data["user"]["login"]
    date = pr_data["created_at"][:10]
    url = pr_data["html_url"]
    merge_sha = pr_data.get("merge_commit_sha")
    body = pr_data.get("body") or ""
    normalized_files = normalize_files(files)
    scope = classify_scope(title, body, normalized_files)
    archs, architecture_disposition, architecture_evidence = derive_architectures(
        title, body, normalized_files
    )
    tags, hw_features, kernel_types, techniques, languages = auto_tag(
        title, normalized_files, body, scope
    )

    all_paths = [item["filename"] for item in normalized_files]
    evidence_paths = list(scope.evidence_paths)
    for row in architecture_evidence:
        locator = row.get("locator", "")
        if locator.startswith("changed-path:"):
            evidence_paths.append(locator.removeprefix("changed-path:"))
        elif locator.startswith("added-patch:"):
            evidence_paths.append(locator.removeprefix("added-patch:"))
    display_paths = list(dict.fromkeys([*all_paths[:25], *evidence_paths]))
    authoritative_changed_files = int(pr_data.get("changed_files") or len(all_paths))
    if authoritative_changed_files < len(all_paths):
        raise ValueError(
            f"PR {repo}#{number} reports changed_files={authoritative_changed_files} "
            f"but {len(all_paths)} file records were supplied"
        )
    enumerated_count = int(
        pr_data.get("changed_files_enumerated_count", len(all_paths))
    )
    changed_files_listing_complete = bool(
        pr_data.get(
            "changed_files_listing_complete",
            enumerated_count == authoritative_changed_files,
        )
    )
    upstream_body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    rendered_body, excerpt_sha256 = render_generated_body(
        body, display_paths, authoritative_changed_files
    )

    # Build frontmatter
    status = "merged" if pr_data.get("merged") or pr_data.get("merged_at") else str(
        pr_data.get("state") or "closed"
    ).lower()
    if status == "merged" and not merge_sha:
        raise ValueError(f"PR {repo}#{number} is merged but has no merge_commit_sha")
    fm = {
        "id": f"pr-{repo_slug}-{number}",
        "repo": repo,
        "pr": number,
        "title": title,
        "author": author,
        "date": date,
        "url": url,
        "source_category": "upstream-code",
        "architectures": archs,
        "architecture_disposition": architecture_disposition,
        "architecture_evidence": architecture_evidence,
        "tags": tags,
        "techniques": techniques,
        "hardware_features": hw_features,
        "kernel_types": kernel_types,
        "languages": languages,
        "captured_at": captured_at,
        "status": status,
        **({"merge_sha": str(merge_sha)} if status == "merged" else {}),
        "inclusion_reason": scope.reason,
        "scope_disposition": scope.disposition,
        "scope_evidence": {"rule": scope.rule, "paths": list(scope.evidence_paths)},
        "changed_files_count": authoritative_changed_files,
        "changed_files_enumerated_count": enumerated_count,
        "changed_files_listing_complete": changed_files_listing_complete,
        "changed_paths": display_paths,
        "changed_paths_complete": (
            changed_files_listing_complete
            and len(display_paths) == authoritative_changed_files
        ),
        "body_contract": BODY_CONTRACT,
        "upstream_body_locator": f"{url} (PR description)",
        "upstream_body_sha256": upstream_body_sha256,
        "upstream_excerpt_sha256": excerpt_sha256,
        "upstream_files_sha256": upstream_files_sha256(normalized_files),
        "upstream_patches_complete": pr_data.get(
            "patches_complete",
            all(item.get("patch") is not None for item in normalized_files),
        ),
    }
    if pr_data.get("changed_files_evidence_method"):
        fm.update({
            "changed_files_evidence_count": len(all_paths),
            "changed_files_evidence_complete": (
                len(all_paths) == authoritative_changed_files
            ),
            "changed_files_evidence_method": pr_data["changed_files_evidence_method"],
            "changed_files_evidence_receipt": pr_data["changed_files_evidence_receipt"],
        })
        fm["scope_evidence"]["path_source"] = pr_data[
            "changed_files_evidence_method"
        ]
    preserved = preserved or {}
    if preserved.get("artifact_dir"):
        fm["artifact_dir"] = preserved["artifact_dir"]

    content = "---\n"
    content += yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    content += "---\n\n"
    content += rendered_body

    return content


def load_skip_audit():
    """Load data/pr-page-skipped.yaml, returning a dict mapping
    (repo, pr_number) -> row. Used to merge new skips with existing rows."""
    if not PR_PAGE_SKIPPED_PATH.is_file():
        return {}
    data = yaml.safe_load(PR_PAGE_SKIPPED_PATH.read_text(encoding="utf-8")) or {}
    rows = data.get("rows") or []
    return {(row["repo"], row["pr_number"]): row for row in rows}


def write_skip_audit(audit_map):
    """Write data/pr-page-skipped.yaml deterministically: rows sorted
    by (repo, pr_number) ascending; byte-stable across re-runs for
    identical inputs."""
    rows = [audit_map[k] for k in sorted(audit_map.keys())]
    payload = {"rows": rows}
    out = "## AC-4 PR-page skip audit. Schema: data/schemas.yaml ::\n"
    out += "## pr-page-skipped-audit. Sort key: (repo, pr_number) ascending.\n"
    out += "## Byte-stable for identical inputs (regen-deterministic).\n"
    out += "##\n"
    out += "## Stages owned by scripts/generate-pr-pages.py:\n"
    out += "##   pre-fetch         -> gh PR fetch returned no data\n"
    out += "##   is-kernel-related -> file-allowlist check excluded the PR\n"
    out += "##\n"
    out += yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=200, allow_unicode=True)
    PR_PAGE_SKIPPED_PATH.write_text(out, encoding="utf-8")


def record_skip(audit_map, repo, pr_number, stage, reason, captured_at):
    """Record (or update) a skip-audit row for one ledger include row."""
    repo_slug = repo.split("/")[1]
    audit_map[(repo, pr_number)] = {
        "pr_id": f"pr-{repo_slug}-{pr_number}",
        "repo": repo,
        "pr_number": pr_number,
        "stage": stage,
        "reason": reason,
        "recorded_at": captured_at,
    }


def process_ledger(ledger_path, max_pages=None, captured_at=None, audit_map=None):
    """Process a candidate ledger and generate PR pages."""
    with open(ledger_path, encoding="utf-8") as f:
        ledger = yaml.safe_load(f)

    # Handle both formats (repo field or inferred from filename)
    repo = ledger.get("repo")
    if not repo:
        slug = Path(ledger_path).stem
        repo_map = {
            "cutlass": "NVIDIA/cutlass", "sglang": "sgl-project/sglang",
            "vllm": "vllm-project/vllm", "flashinfer": "flashinfer-ai/flashinfer",
            "pytorch": "pytorch/pytorch", "deepgemm": "deepseek-ai/DeepGEMM",
            "flash-attention": "Dao-AILab/flash-attention",
            "tensorrt-llm": "NVIDIA/TensorRT-LLM", "cccl-cub": "NVIDIA/cccl",
            "triton": "triton-lang/triton", "tilelang": "tile-ai/tilelang",
            "thunderkittens": "HazyResearch/ThunderKittens",
            "tilekernels": "deepseek-ai/TileKernels", "quack": "Dao-AILab/quack",
        }
        repo = repo_map.get(slug, f"unknown/{slug}")
    repo_slug = repo.split("/")[1]
    outdir = REPO_ROOT / "sources" / "prs" / repo_slug
    outdir.mkdir(parents=True, exist_ok=True)

    # Get existing PR numbers to avoid re-fetching
    existing = set()
    for p in outdir.glob("PR-*.md"):
        try:
            existing.add(int(p.stem.split("-")[1]))
        except (ValueError, IndexError):
            pass

    # Get included candidates (handle both 'prs' and 'candidates' keys, both cases)
    prs_key = "prs" if "prs" in ledger else "candidates"
    candidates = ledger.get(prs_key, [])

    included = []
    for c in candidates:
        decision = str(c.get("decision", "")).lower()
        if decision == "include":
            num = c["number"]
            if num not in existing:
                included.append(c)

    if captured_at is None:
        captured_at = date.today().isoformat()
    else:
        # Smoke check: must parse as ISO YYYY-MM-DD
        date.fromisoformat(captured_at)

    print(f"\n{repo}: {len(included)} new PRs to process ({len(existing)} already exist)")
    print(f"  captured_at = {captured_at}")
    if max_pages:
        print(f"  Target: stop after generating {max_pages} pages")

    generated = 0
    skipped = 0

    for i, candidate in enumerate(included):
        if max_pages and generated >= max_pages:
            print(f"  Reached target of {max_pages} generated pages")
            break
        number = candidate["number"]
        title = candidate.get("title", "")

        # Fetch PR details (gh CLI with auth = 5000/hour limit)
        pr_data = fetch_pr(repo, number)
        if not pr_data:
            skipped += 1
            if audit_map is not None:
                record_skip(audit_map, repo, number, "pre-fetch",
                            "gh pr fetch returned no data", captured_at)
            continue
        files = fetch_pr_files(repo, number)

        # Re-triage with file data
        is_kernel, reason = is_kernel_related(pr_data.get("title", title), files, pr_data.get("body") or "")
        if is_kernel is False:
            skipped += 1
            if audit_map is not None:
                record_skip(audit_map, repo, number, "is-kernel-related",
                            reason, captured_at)
            continue

        inclusion_reason = reason if is_kernel else "deferred-semantic"
        content = generate_page(repo, pr_data, files, inclusion_reason, captured_at)

        outpath = outdir / f"PR-{number}.md"
        outpath.write_text(content, encoding="utf-8")
        generated += 1

        if generated % 10 == 0:
            print(f"  Generated {generated} pages so far...")

        # Progress marker (gh auth supports 5000/hour)
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(included)}")

    print(f"  Done: {generated} generated, {skipped} skipped, {len(existing)} pre-existing")
    return generated


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_pages = None
    captured_at = None
    for a in sys.argv[1:]:
        if a.startswith("--max="):
            max_pages = int(a.split("=")[1])
        elif a.startswith("--captured-at="):
            captured_at = a.split("=", 1)[1]
            date.fromisoformat(captured_at)  # smoke check

    audit_map = load_skip_audit()

    if "--all" in sys.argv:
        ledger_dir = REPO_ROOT / "candidates"
        for ledger_file in sorted(ledger_dir.glob("*.yaml")):
            process_ledger(ledger_file, max_pages, captured_at, audit_map)
    elif args:
        process_ledger(args[0], max_pages, captured_at, audit_map)
    else:
        print("Usage: python3 scripts/generate-pr-pages.py candidates/cutlass.yaml [--max=N] [--captured-at=YYYY-MM-DD]")
        print("       python3 scripts/generate-pr-pages.py --all [--max=N] [--captured-at=YYYY-MM-DD]")
        print("       (default captured_at = today's date)")
        return

    write_skip_audit(audit_map)


if __name__ == "__main__":
    main()
