#!/usr/bin/env python3
"""Strict online check of high-risk PR policy outputs against live GitHub APIs.

The offline architecture receipt detects ordinary page drift but shares local
policy and frozen inputs with generation. This gate supplies an external
boundary: it re-fetches a deterministic retained/removed sample, requires an
exact paginated changed-file count unless GitHub's documented 3,000-file cap is
reached, preserves the pull object's authoritative total, and re-derives scope,
architecture, metadata, hashes, and paths before comparing the current page.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from pr_policy import (
    _path_is_distributed_implementation,
    _path_is_non_implementation,
    classify_scope,
    cuda_translation_unit_device_signal,
    device_code_pattern_sha256,
    derive_architectures,
    derive_metadata,
    cuda_requires_complete_evidence,
    python_dsl_languages,
    python_dsl_pattern_sha256,
    python_requires_complete_evidence,
    parse_git_diff_files,
    upstream_files_sha256,
)


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = (
    ("tilelang/PR-2198.md", "tile-ai/tilelang", 2198),       # SM75 title boundary
    ("vllm/PR-29901.md", "vllm-project/vllm", 29901),       # SM75 vs SM80 context
    ("flashinfer/PR-1503.md", "flashinfer-ai/flashinfer", 1503),  # SM87 guard
    ("cutlass/PR-2995.md", "NVIDIA/cutlass", 2995),         # B200 product mapping
    ("TensorRT-LLM/PR-10987.md", "NVIDIA/TensorRT-LLM", 10987),
    ("TensorRT-LLM/PR-12937.md", "NVIDIA/TensorRT-LLM", 12937),  # >3000 cap
    ("TensorRT-LLM/PR-12612.md", "NVIDIA/TensorRT-LLM", 12612),  # >3000 cap
    ("TensorRT-LLM/PR-13652.md", "NVIDIA/TensorRT-LLM", 13652),  # >3000 cap
    ("TensorRT-LLM/PR-12470.md", "NVIDIA/TensorRT-LLM", 12470),  # tail .cu
    ("TensorRT-LLM/PR-13505.md", "NVIDIA/TensorRT-LLM", 13505),  # cap removed
    ("TensorRT-LLM/PR-14291.md", "NVIDIA/TensorRT-LLM", 14291),  # tail headers
    ("vllm/PR-15354.md", "vllm-project/vllm", 15354),       # host-only new .cu
    ("flashinfer/PR-958.md", "flashinfer-ai/flashinfer", 958),    # binding-only .cu
    ("flashinfer/PR-994.md", "flashinfer-ai/flashinfer", 994),    # distributed exclusion
    ("DeepGEMM/PR-83.md", "deepseek-ai/DeepGEMM", 83),      # GemmType-only host .cu is negative
    ("sglang/PR-3033.md", "sgl-project/sglang", 3033),      # modified host-only .cu
    ("flashinfer/PR-1398.md", "flashinfer-ai/flashinfer", 1398),  # tile-only host launcher
    ("sglang/PR-6101.md", "sgl-project/sglang", 6101),      # immutable full .cu is true device code
    ("flashinfer/PR-1389.md", "flashinfer-ai/flashinfer", 1389),  # canonical mixed receipt digest
)


def frontmatter(path: Path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL)
    if not match:
        raise ValueError(f"missing frontmatter: {path}")
    return yaml.safe_load(match.group(1)) or {}


def github_json(endpoint: str):
    command = ["gh", "api"]
    command.append(endpoint)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"gh api failed: {endpoint}")
    return json.loads(result.stdout)


def fetch_full_diff_files(repo: str, pr: int):
    """Fetch and parse the complete pull diff outside the capped files API."""
    url = f"https://github.com/{repo}/pull/{pr}.diff"
    result = subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "120", url],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"curl failed: {url}")
    return parse_git_diff_files(result.stdout)


def _fetch_bytes(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "KernelWiki-factual-audit"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def attach_complete_file_evidence(repo: str, head_sha: str, files):
    """Re-fetch inconclusive CUDA/Python files at the immutable pull head."""
    candidates = []
    for item in files:
        path = item.get("filename") or item.get("path") or ""
        if _path_is_non_implementation(path) or _path_is_distributed_implementation(path):
            continue
        if cuda_requires_complete_evidence(item):
            kind = "cuda"
        elif python_requires_complete_evidence(item):
            kind = "python-dsl"
        else:
            continue
        url = (
            f"https://raw.githubusercontent.com/{repo}/{head_sha}/"
            + urllib.parse.quote(path, safe="/")
        )
        candidates.append((item, url, kind))
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(_fetch_bytes, url): (item, kind)
            for item, url, kind in candidates
        }
        for future in as_completed(futures):
            item, kind = futures[future]
            content = future.result()
            text = content.decode("utf-8", errors="replace")
            if kind == "cuda":
                item.update({
                    "complete_file_evidence_complete": True,
                    "complete_file_sha256": hashlib.sha256(content).hexdigest(),
                    "complete_file_device_signal": cuda_translation_unit_device_signal(text),
                    "complete_file_device_pattern_sha256": device_code_pattern_sha256(),
                })
            else:
                languages = list(python_dsl_languages(text))
                item.update({
                    "complete_file_python_dsl_evidence_complete": True,
                    "complete_file_python_dsl_sha256": hashlib.sha256(content).hexdigest(),
                    "complete_file_python_dsl_signal": bool(languages),
                    "complete_file_python_dsl_languages": languages,
                    "complete_file_python_dsl_pattern_sha256": python_dsl_pattern_sha256(),
                })
    return files


def attach_complete_cuda_evidence(repo: str, head_sha: str, files):
    """Backward-compatible name for the complete CUDA/Python evidence gate."""
    return attach_complete_file_evidence(repo, head_sha, files)


def fetch_upstream(repo: str, pr: int):
    pull = github_json(f"repos/{repo}/pulls/{pr}")
    files = []
    page = 1
    while True:
        batch = github_json(
            f"repos/{repo}/pulls/{pr}/files?per_page=100&page={page}"
        )
        if not isinstance(batch, list):
            raise RuntimeError(f"{repo}#{pr}: GitHub files response is not a list")
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    expected_count = int(pull.get("changed_files") or 0)
    listing_complete = len(files) == expected_count
    if not listing_complete and not (expected_count > 3000 and len(files) == 3000):
        raise RuntimeError(
            f"{repo}#{pr}: incomplete GitHub file pagination: "
            f"expected {expected_count}, received {len(files)}"
        )
    evidence_files = files
    if not listing_complete:
        evidence_files = fetch_full_diff_files(repo, pr)
        if len(evidence_files) != expected_count:
            raise RuntimeError(
                f"{repo}#{pr}: full pull diff count mismatch: "
                f"expected {expected_count}, received {len(evidence_files)}"
            )
    head = (pull.get("head") or {}).get("sha")
    if any(
        cuda_requires_complete_evidence(item) or python_requires_complete_evidence(item)
        for item in evidence_files
    ):
        if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40,64}", head):
            raise RuntimeError(f"{repo}#{pr}: missing immutable pull head SHA")
        attach_complete_file_evidence(repo, head, evidence_files)
    return pull, files, listing_complete, evidence_files


def derive_upstream_record(pull, files, listing_complete=None, evidence_files=None):
    title = pull.get("title") or ""
    body = pull.get("body") or ""
    evidence_files = evidence_files if evidence_files is not None else files
    scope = classify_scope(title, body, evidence_files)
    architectures, disposition, evidence = derive_architectures(
        title, body, evidence_files
    )
    metadata = derive_metadata(title, body, evidence_files, scope)
    total = int(pull.get("changed_files") or len(files))
    if listing_complete is None:
        listing_complete = len(files) == total
    all_paths = [item["filename"] for item in evidence_files]
    evidence_paths = list(scope.evidence_paths)
    for row in evidence:
        locator = row.get("locator", "")
        if locator.startswith("changed-path:"):
            evidence_paths.append(locator.removeprefix("changed-path:"))
        elif locator.startswith("added-patch:"):
            evidence_paths.append(locator.removeprefix("added-patch:"))
    display_paths = list(dict.fromkeys([*all_paths[:25], *evidence_paths]))
    result = {
        "title": title,
        "retain": scope.retain,
        "scope_rule": scope.rule,
        "scope_paths": list(scope.evidence_paths),
        "architectures": architectures,
        "architecture_disposition": disposition,
        "architecture_evidence": evidence,
        "metadata": metadata,
        "upstream_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "upstream_files_sha256": upstream_files_sha256(evidence_files),
        "changed_files_count": total,
        "changed_files_enumerated_count": len(files),
        "changed_files_listing_complete": listing_complete,
        "changed_paths": display_paths,
        "changed_paths_complete": listing_complete and len(display_paths) == total,
    }
    if evidence_files is not files:
        result.update({
            "changed_files_evidence_count": len(evidence_files),
            "changed_files_evidence_complete": len(evidence_files) == total,
            "changed_files_evidence_method": "github-pull-diff",
            "changed_files_evidence_receipt": "audit/pr-file-cap-reconstruction.json",
            "scope_path_source": "github-pull-diff",
        })
    return result


def compare_local_page(label: str, fm, expected):
    errors = []
    if not expected["retain"]:
        if fm is not None:
            errors.append(f"{label}: upstream policy removes PR but local page exists")
        return errors
    if fm is None:
        return [f"{label}: upstream policy retains PR but local page is missing"]

    comparisons = {
        "title": expected["title"],
        "architectures": expected["architectures"],
        "architecture_disposition": expected["architecture_disposition"],
        "architecture_evidence": expected["architecture_evidence"],
        "upstream_body_sha256": expected["upstream_body_sha256"],
        "upstream_files_sha256": expected["upstream_files_sha256"],
        "changed_files_count": expected["changed_files_count"],
        "changed_paths": expected["changed_paths"],
        "changed_paths_complete": expected["changed_paths_complete"],
        "scope_disposition": "retained",
    }
    if not expected["changed_files_listing_complete"] or (
        "changed_files_enumerated_count" in fm
        or "changed_files_listing_complete" in fm
    ):
        comparisons.update({
            "changed_files_enumerated_count": expected["changed_files_enumerated_count"],
            "changed_files_listing_complete": expected["changed_files_listing_complete"],
        })
    if "changed_files_evidence_count" in expected:
        for field in (
            "changed_files_evidence_count",
            "changed_files_evidence_complete",
            "changed_files_evidence_method",
            "changed_files_evidence_receipt",
        ):
            comparisons[field] = expected[field]
    for field, value in comparisons.items():
        if fm.get(field) != value:
            errors.append(f"{label}: live-upstream mismatch in {field}")
    scope_evidence = fm.get("scope_evidence") or {}
    if scope_evidence.get("rule") != expected["scope_rule"]:
        errors.append(f"{label}: live-upstream mismatch in scope_evidence.rule")
    if scope_evidence.get("paths") != expected["scope_paths"]:
        errors.append(f"{label}: live-upstream mismatch in scope_evidence.paths")
    if scope_evidence.get("path_source") != expected.get("scope_path_source"):
        errors.append(f"{label}: live-upstream mismatch in scope_evidence.path_source")
    for field in ("tags", "hardware_features", "kernel_types", "techniques", "languages"):
        if fm.get(field) != expected["metadata"][field]:
            errors.append(f"{label}: live-upstream mismatch in {field}")
    return errors


def main():
    errors = []
    for relative, repo, pr in SAMPLES:
        label = f"{repo}#{pr}"
        page = ROOT / "sources" / "prs" / relative
        try:
            pull, files, listing_complete, evidence_files = fetch_upstream(repo, pr)
            expected = derive_upstream_record(
                pull, files, listing_complete, evidence_files
            )
            fm = frontmatter(page) if page.is_file() else None
            errors.extend(compare_local_page(label, fm, expected))
            suffix = (
                ""
                if listing_complete
                else (
                    f" ({len(files)}/{expected['changed_files_count']} API-listed; "
                    f"{len(evidence_files)} full-diff paths)"
                )
            )
            print(
                f"checked {label}: {'retained' if expected['retain'] else 'removed'}{suffix}"
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: {exc}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(SAMPLES)} deterministic PR samples against live GitHub evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
