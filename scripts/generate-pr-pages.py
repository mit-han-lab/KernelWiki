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

from pr_architectures import infer_architectures

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
    ".cu": "cuda-cpp", ".cuh": "cuda-cpp",
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


def fetch_pr_body_text(repo, number):
    """Fetch GitHub's plain-text PR body representation via GraphQL."""
    owner, name = repo.split("/", 1)
    query = """
      query($owner: String!, $name: String!, $number: Int!) {
        repository(owner: $owner, name: $name) {
          pullRequest(number: $number) { bodyText }
        }
      }
    """
    try:
        result = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}",
             "-f", f"owner={owner}", "-f", f"name={name}",
             "-F", f"number={number}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout)
        pr = ((payload.get("data") or {}).get("repository") or {}).get("pullRequest")
        return None if pr is None else str(pr.get("bodyText") or "")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def fetch_pr_files(repo, number):
    """Fetch the complete changed-file list via paginated GitHub REST."""
    try:
        result = subprocess.run(
            ["gh", "api", "--paginate", "--jq", ".[].filename",
             f"repos/{repo}/pulls/{number}/files?per_page=100"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def is_kernel_related(title, files):
    """Check if PR is kernel-related based on title + changed files."""
    title_lower = title.lower()

    # Title-based exclusion
    for pat in EXCLUDE_TITLE_PATTERNS:
        if re.search(pat, title_lower):
            return False, "excluded by title pattern"

    # File-based inclusion
    kernel_exts = {".cu", ".cuh", ".ptx"}
    kernel_dirs = {"cutlass/", "csrc/", "kernel", "triton/", "cute/", "gemm",
                   "attention", "moe", "inductor/", "tensor_core", "mma",
                   "scaled_mm", "quantiz", "flash_attention", "sdpa"}

    has_kernel_file = False
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext in kernel_exts:
            has_kernel_file = True
            break
        for kd in kernel_dirs:
            if kd in f.lower():
                has_kernel_file = True
                break

    # Semantic signals in title
    semantic_kws = ["kernel", "sm100", "blackwell", "tcgen05", "tmem", "nvfp4",
                     "fp8", "fp4", "gemm", "attention", "moe", "mla", "cutlass",
                     "flashinfer", "deepgemm", "flashmla", "triton", "fmha",
                     "inductor", "sdpa", "flash_attention", "scaled_mm",
                     "block_scale", "quantiz", "tma", "b200", "cuda 13"]
    has_semantic = any(kw in title_lower for kw in semantic_kws)

    if has_kernel_file:
        return True, "kernel file changes"
    if has_semantic and files:
        return True, "semantic match with file changes"
    if has_semantic and not files:
        return None, "semantic match but no file data"  # defer
    return False, "no kernel signals"


def auto_tag(title, files):
    """Auto-detect tags from title and changed files."""
    text = (title + " " + " ".join(files)).lower()

    tags = set()
    hw_features = set()
    kernel_types = set()
    techniques = set()
    languages = set()

    for kw, tag in KW_TO_TAGS.items():
        if kw in text:
            hw_features.add(tag)
            tags.add(tag)

    for kw, kt in KW_TO_KT.items():
        if kw in text:
            kernel_types.add(kt)
            tags.add(kt)

    for kw, tech in KW_TO_TECH.items():
        if kw in text:
            techniques.add(tech)
            tags.add(tech)

    for kw, lang in KW_TO_LANG.items():
        if kw in text:
            languages.add(lang)

    # Filter to valid vocabulary only
    tags = tags & ALL_TAGS
    hw_features = hw_features & VALID_HW
    kernel_types = kernel_types & VALID_KT
    techniques = techniques & VALID_TECHNIQUES
    languages = languages & VALID_LANGS

    # Extensions are language evidence regardless of other topical tokens.
    # Do not infer CUDA C++ from a bare "cuda" substring: Python files such
    # as test_cudagraph_trees.py are not CUDA C++ source.
    for f in files:
        lowered = f.lower()
        if lowered.endswith((".cu", ".cuh", ".cc", ".cpp", ".cxx", ".h", ".hpp")):
            languages.add("cuda-cpp")
        if lowered.endswith(".py"):
            languages.add("python")
            if "triton" in lowered:
                languages.add("triton")

    return sorted(tags), sorted(hw_features), sorted(kernel_types), sorted(techniques), sorted(languages)


def render_upstream_body(title, body, files):
    """Render a compact body containing only deterministic upstream material."""
    if not body.strip():
        return "\n".join([
            "## Upstream description status",
            "",
            "GitHub `bodyText` is empty.",
            "",
            "> Local status statement; empty `bodyText` identity: frontmatter hash; source: `url`.",
        ])
    raw_excerpt = body.strip()[:160]
    excerpt = "\n".join(line.rstrip() for line in raw_excerpt.splitlines()).rstrip()
    lines = [
        "## Upstream description excerpt",
        "",
        excerpt,
        "",
        "> GitHub `bodyText` prefix; local line-end whitespace normalized. Full `bodyText`/file identities: frontmatter hashes; source: `url`.",
    ]
    return "\n".join(lines)


def load_existing_frontmatter(path):
    """Load fields that must survive an in-place refresh."""
    if not path.is_file():
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", path.read_text(encoding="utf-8"), re.DOTALL)
    if not match:
        return {}
    try:
        payload = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def generate_page(repo, pr_data, files, inclusion_reason, captured_at, existing_fm=None):
    """Generate markdown page content for a PR."""
    repo_slug = repo.split("/")[1]
    id_repo_slug = "deepgemm" if repo_slug.lower() == "deepgemm" else repo_slug
    number = pr_data["number"]
    title = pr_data["title"]
    author = (pr_data.get("user") or {}).get("login") or "unknown"
    date = pr_data["created_at"][:10]
    url = pr_data["html_url"]
    merge_sha = (pr_data.get("merge_commit_sha") or "")[:8]
    body = pr_data.get("body") or ""
    merged = bool(pr_data.get("merged_at"))
    status = "merged" if merged else str(pr_data.get("state") or "closed").lower()

    # Determine architectures from explicit PR evidence; an empty list is more
    # accurate than assigning SM100 to architecture-neutral work.
    archs = infer_architectures(title, body, files)

    tags, hw_features, kernel_types, techniques, languages = auto_tag(title, files)

    # Ensure tags include all kernel_types and hw_features
    all_tags = set(tags)
    for kt in kernel_types:
        if kt in ALL_TAGS:
            all_tags.add(kt)
    for hw in hw_features:
        if hw in ALL_TAGS:
            all_tags.add(hw)
    tags = sorted(all_tags & ALL_TAGS)

    # Build frontmatter
    fm = {
        "id": f"pr-{id_repo_slug}-{number}",
        "repo": repo,
        "pr": number,
        "title": title,
        "author": author,
        "date": date,
        "url": url,
        "source_category": "upstream-code",
        "architectures": archs,
        "tags": tags,
        "techniques": techniques if techniques else [],
        "hardware_features": hw_features if hw_features else [],
        "kernel_types": kernel_types if kernel_types else [],
        "languages": languages,
        "captured_at": captured_at,
        "status": status,
        "inclusion_reason": inclusion_reason,
        "changed_paths": files[:5],
        "changed_paths_total": len(files),
        "changed_paths_truncated": len(files) > 5,
        "upstream_body_text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "upstream_files_sha256": hashlib.sha256(
            "\n".join(files).encode("utf-8")
        ).hexdigest(),
    }
    if merged:
        fm["merge_sha"] = merge_sha
    if existing_fm and existing_fm.get("artifact_dir"):
        fm["artifact_dir"] = existing_fm["artifact_dir"]

    content = "---\n"
    content += yaml.safe_dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False, width=1000)
    content += "---\n\n"
    content += render_upstream_body(title, body, files)

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
    out += "##   bodyText-fetch    -> GitHub GraphQL bodyText fetch returned no data\n"
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


def process_ledger(ledger_path, max_pages=None, captured_at=None, audit_map=None,
                   refresh_existing=False, only_prs=None):
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
            if only_prs is not None and num not in only_prs:
                continue
            if num not in existing or refresh_existing:
                included.append(c)

    if captured_at is None:
        captured_at = date.today().isoformat()
    else:
        # Smoke check: must parse as ISO YYYY-MM-DD
        date.fromisoformat(captured_at)

    action = "selected PRs to refresh/process" if refresh_existing else "new PRs to process"
    print(f"\n{repo}: {len(included)} {action} ({len(existing)} already exist)")
    if refresh_existing:
        print(f"  captured_at = {captured_at} for new pages; existing pages preserve their recorded value")
    else:
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
        outpath = outdir / f"PR-{number}.md"
        existing_fm = load_existing_frontmatter(outpath) if refresh_existing else {}
        page_captured_at = existing_fm.get("captured_at", captured_at)

        # Fetch PR details (gh CLI with auth = 5000/hour limit)
        pr_data = fetch_pr(repo, number)
        if not pr_data:
            skipped += 1
            if audit_map is not None:
                record_skip(audit_map, repo, number, "pre-fetch",
                            "gh pr fetch returned no data", captured_at)
            continue
        body_text = fetch_pr_body_text(repo, number)
        if body_text is None:
            skipped += 1
            if audit_map is not None:
                record_skip(audit_map, repo, number, "bodyText-fetch",
                            "GitHub GraphQL bodyText fetch returned no data", captured_at)
            continue
        pr_data["body"] = body_text
        files = fetch_pr_files(repo, number)

        # Re-triage with file data
        is_kernel, reason = is_kernel_related(title, files)
        if is_kernel is False:
            skipped += 1
            if audit_map is not None:
                record_skip(audit_map, repo, number, "is-kernel-related",
                            reason, captured_at)
            continue

        inclusion_reason = reason if is_kernel else "deferred-semantic"
        content = generate_page(
            repo, pr_data, files, inclusion_reason, page_captured_at, existing_fm
        )

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
    refresh_existing = "--refresh-existing" in sys.argv
    only_prs = None
    for a in sys.argv[1:]:
        if a.startswith("--max="):
            max_pages = int(a.split("=")[1])
        elif a.startswith("--captured-at="):
            captured_at = a.split("=", 1)[1]
            date.fromisoformat(captured_at)  # smoke check
        elif a.startswith("--pr="):
            only_prs = {int(value) for value in a.split("=", 1)[1].split(",") if value}

    audit_map = load_skip_audit()

    if "--all" in sys.argv:
        ledger_dir = REPO_ROOT / "candidates"
        for ledger_file in sorted(ledger_dir.glob("*.yaml")):
            process_ledger(ledger_file, max_pages, captured_at, audit_map,
                           refresh_existing, only_prs)
    elif args:
        process_ledger(args[0], max_pages, captured_at, audit_map,
                       refresh_existing, only_prs)
    else:
        print("Usage: python3 scripts/generate-pr-pages.py candidates/cutlass.yaml "
              "[--max=N] [--captured-at=YYYY-MM-DD] "
              "[--refresh-existing] [--pr=N[,N...]]")
        print("       python3 scripts/generate-pr-pages.py --all [--max=N] [--captured-at=YYYY-MM-DD]")
        print("       (default captured_at = today's date)")
        return

    write_skip_audit(audit_map)


if __name__ == "__main__":
    main()
