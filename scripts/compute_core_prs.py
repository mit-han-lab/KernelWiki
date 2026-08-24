#!/usr/bin/env python3
"""Compute the deterministic Phase 3 core-PR set + CuTe DSL / Triton universes.

Sources of truth (union, deduplicated, sorted):
  A. Graph-closure of wiki/**.md frontmatter `sources:` that point to a PR ID.
  B. PR IDs referenced anywhere inside any sources/contests/**/*.md body.
  C. PRs in data/cute-dsl-universe.yaml that `captured: true` after applying
     the cute-dsl lane of data/inclusion-policy.yaml.
  D. PRs in data/triton-universe.yaml that `captured: true` after applying
     the triton lane of data/inclusion-policy.yaml.
  E. Additions from data/core-prs-allowlist.yaml (subtract exclusions).

Outputs (all written to data/):
  - core-prs.yaml        (the captured set; deterministic ordering; sha256)
  - cute-dsl-universe.yaml  (all cute-dsl PRs, captured or skipped + reason)
  - triton-universe.yaml    (all triton PRs,    captured or skipped + reason)

Re-running on an unchanged corpus produces byte-identical output files
(AC-1 deterministic regeneration check).
"""

from __future__ import annotations

import hashlib
import re
import sys
from fnmatch import fnmatch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = REPO_ROOT / "sources"
WIKI = REPO_ROOT / "wiki"
DATA = REPO_ROOT / "data"


def extract_frontmatter(md_path):
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None, ""
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", text, re.DOTALL)
    if not m:
        return None, text
    try:
        return yaml.safe_load(m.group(1)), m.group(2)
    except yaml.YAMLError:
        return None, m.group(2)


def load_all_prs():
    """Load the complete set of PR pages under sources/prs/.

    Returns a dict keyed by PR ID: {id: {frontmatter..., '_path': str}}.
    """
    prs = {}
    for md in sorted((SOURCES / "prs").rglob("PR-*.md")):
        fm, _ = extract_frontmatter(md)
        if not fm or "id" not in fm:
            continue
        rec = dict(fm)
        rec["_path"] = str(md.relative_to(REPO_ROOT))
        prs[fm["id"]] = rec
    return prs


def graph_closure_from_wiki():
    """Collect every PR ID referenced in any wiki/**.md frontmatter `sources:` list."""
    ids = set()
    for md in sorted(WIKI.rglob("*.md")):
        fm, _ = extract_frontmatter(md)
        if not fm:
            continue
        for sid in (fm.get("sources") or []):
            if isinstance(sid, str) and sid.startswith("pr-"):
                ids.add(sid)
    return ids


def contest_referenced_prs():
    """Collect PR IDs explicitly listed in a contest page's frontmatter
    `referenced_prs:` field.

    R30: switched from a full-text regex scan to an explicit structured
    list. The regex-over-body approach picked up PR IDs mentioned in
    prose (e.g. `pr-sglang-21239` appearing in a `code_unavailable_reason`
    string), which pulled unrelated PRs into the generated core set.
    Curators who want a contest page to bring a PR into core must now
    declare it explicitly:

        ---
        ...
        referenced_prs:
          - pr-cutlass-2466
          - pr-vllm-23696
        ---
    """
    ids = set()
    for md in sorted((SOURCES / "contests").rglob("*.md")):
        fm, _ = extract_frontmatter(md)
        if not isinstance(fm, dict):
            continue
        for pid in fm.get("referenced_prs") or []:
            if isinstance(pid, str) and pid.startswith("pr-"):
                ids.add(pid)
    return ids


def path_matches_any(paths, globs):
    return any(fnmatch(p, g) for p in paths for g in globs)


# Matches scripts/fetch_pr_diff.py SKIP_GLOBS: a PR whose every changed
# path falls under one of these globs will produce an empty key-files/
# bundle downstream. Keep in sync with fetch_pr_diff.SKIP_GLOBS.
_FETCH_SKIP_GLOBS = (
    "tests/**", "**/tests/**",
    "test/**", "**/test/**",
    "*_test.cu", "*_test.cpp", "*_tests.py", "**/test_*.py",
    "**/*_test.py", "**/*_tests.py",
    "**/conftest.py",
    "benchmark/**", "benchmarks/**", "bench/**",
    "**/benchmark/**", "**/benchmarks/**", "**/bench/**",
    "**/bench_*.py", "**/*_bench.py", "**/*_benchmark.py",
    "docs/**", "**/docs/**",
    "**/README*", "**/CHANGELOG*", "**/release_notes*",
    "**/LICENSE*", "**/NOTICE*",
    ".github/**", "**/.github/**",
    "ci/**", "**/ci/**",
)


def all_paths_would_be_skipped_by_fetch(paths):
    """Return True iff every path in `paths` matches fetch_pr_diff's
    SKIP_GLOBS (tests / benchmarks / docs / ci). Such a PR would ship
    only diff.patch — no key-files/ — so capturing it just pollutes the
    corpus. Empty `paths` returns False (no evidence of a noise PR).

    Trusts the source page's `changed_paths` as ground truth. Any stale
    or incomplete source pages must be backfilled from GitHub (see
    Round 24's one-shot backfill) so this helper stays self-contained:
    compute_manifests() must not depend on `artifacts/` state, which
    would break regeneration in sparse clones.
    """
    if not paths:
        return False
    return all(any(fnmatch(p, g) for g in _FETCH_SKIP_GLOBS) for p in paths)


_STRONG_CUTE_GLOBS = (
    "cute/**", "**/cute/**",
    "cute_dsl/**", "**/cute_dsl/**",
    "**/cutlass.cute.**",
    "examples/cute_dsl/**", "**/examples/cute_dsl/**",
    "examples/python/CuTeDSL/**", "**/examples/python/CuTeDSL/**",
    "tools/cute/**", "**/tools/cute/**",
    "**/*cutedsl*.py",
    "**/*cutedsl*.cu",
)

_KERNEL_OR_EXAMPLE_GLOBS = _STRONG_CUTE_GLOBS + (
    "examples/**/*.cu", "**/examples/**/*.cu",
    "examples/**/*.cuh", "**/examples/**/*.cuh",
    "examples/**/*.hpp", "**/examples/**/*.hpp",
)


def apply_cute_dsl_policy(policy, pr):
    """Return (captured: bool, skipped_reason: str|None)."""
    rules = policy.get("cute-dsl", {}) or {}
    langs = set(pr.get("languages") or [])
    tags = set(pr.get("tags") or [])
    changed_paths = pr.get("changed_paths") or []

    # Cute-DSL path signals. Two tiers:
    #   STRONG: paths that literally name cute or cutedsl (e.g.
    #           `include/cute/...`, `flashinfer_cutedsl_moe.py`). A
    #           strong match alone is enough to enter the lane even
    #           without a cute-dsl tag.
    #   KERNEL_OR_EXAMPLE: a superset used for the "does this PR
    #           author a kernel/example file?" check. Includes generic
    #           CUTLASS `examples/**/*.cu` paths which are CuTe-DSL in
    #           the Blackwell era but can't be distinguished from
    #           non-cute cuda-cpp examples by path alone.
    # fnmatch `**` matching requires both top-level and `**/` prefix
    # variants (BL-20260417-skip-globs-fnmatch-depth).
    STRONG_CUTE_GLOBS = _STRONG_CUTE_GLOBS
    KERNEL_OR_EXAMPLE_GLOBS = _KERNEL_OR_EXAMPLE_GLOBS
    # Restrict the path-signal checks to paths that survive fetch's
    # SKIP_GLOBS. Otherwise a PR whose only cute-ish path is
    # `test_cutedsl_foo.py` or a benchmark script falsely passes the
    # metadata-only gate via a path the fetch stage will drop
    # downstream (R27 tightening on top of R19).
    non_skip_paths = [
        p for p in changed_paths
        if not any(fnmatch(p, g) for g in _FETCH_SKIP_GLOBS)
    ]
    has_strong_cute = any(
        fnmatch(p, g) for p in non_skip_paths for g in STRONG_CUTE_GLOBS
    )
    has_kernel_example = any(
        fnmatch(p, g) for p in non_skip_paths for g in KERNEL_OR_EXAMPLE_GLOBS
    )
    language_tag_match = "cute-dsl" in langs or "cute-dsl" in tags

    # Initial gate: either the PR declares cute-dsl in metadata, or a
    # changed path carries a strong cute/cutedsl signal. A path-only
    # match under the WEAK globs (generic `examples/*.cu`) is NOT
    # enough to enter the lane without a cute-dsl tag — otherwise
    # untagged cuda-cpp CUTLASS-example PRs would be falsely pulled
    # in via the widened globs.
    if not (language_tag_match or has_strong_cute):
        return False, "not a cute-dsl PR"

    # Skip rule: docs/CHANGELOG/release_notes only
    skip_globs = []
    for crit in rules.get("skip_criteria", []):
        if isinstance(crit, dict) and "changed_paths_match_only" in crit:
            skip_globs.extend(crit["changed_paths_match_only"])
    if skip_globs and changed_paths:
        if all(any(fnmatch(p, g) for g in skip_globs) for p in changed_paths):
            return False, "documentation-only CuTe DSL PR"

    # Tightening (R19): reject PRs whose every changed path would be
    # dropped by scripts/fetch_pr_diff.py's SKIP_GLOBS (tests,
    # benchmarks, docs, ci). Such PRs end up with diff.patch-only
    # bundles and contaminate the captured core set. Concrete example:
    # pr-vllm-39644 is tagged `cute-dsl` but its only changed_path is
    # `tests/kernels/moe/test_cutedsl_moe.py`. The helper trusts the
    # source page's changed_paths; R24 backfilled every stale source
    # page from GitHub so the decision is now self-contained.
    if all_paths_would_be_skipped_by_fetch(changed_paths):
        return False, (
            "all changed_paths are tests/benchmarks/docs "
            "(empty bundle after fetch_pr_diff.SKIP_GLOBS)"
        )

    # Tightening (R27): metadata-only captures are rejected per policy.
    # If the PR entered the lane via `language_tag_match` but no path
    # matches any kernel/example glob (after fetch-skip filtering), it
    # is a "cute-dsl-affinity" PR rather than a cute-dsl authorship PR.
    # Concrete example: pr-sglang-21428 lists only
    # `python/sglang/srt/layers/attention/linear/kda_backend.py`, a
    # backend-dispatch file with no cute/cutedsl/CuTeDSL path marker
    # and no `.cu`/`.cuh`/`.hpp` example file. Policy doc's
    # `language_tag_only_without_kernel_path: true` skip rule covers
    # this case.
    if not has_kernel_example:
        return False, (
            "cute-dsl PR with no kernel/example path match after "
            "fetch-skip filtering (control-plane / affinity only; "
            "policy requires a kernel/example path)"
        )

    return True, None


def apply_triton_policy(policy, pr):
    """Return (captured: bool, skipped_reason: str|None)."""
    rules = policy.get("triton", {}) or {}
    langs = set(pr.get("languages") or [])
    if "triton" not in langs and "triton" not in set(pr.get("tags") or []):
        return False, "not a triton PR"

    archs = set(pr.get("architectures") or [])
    symptoms = set(pr.get("symptoms") or [])
    changed_paths = pr.get("changed_paths") or []
    desc = str(pr.get("description") or "").lower() + " " + str(pr.get("title") or "").lower()

    # Skip pure Hopper
    if archs and archs.issubset({"sm90", "sm90a"}):
        return False, "pure Hopper Triton (no SM100 relevance)"
    # R30: skip non-Blackwell-vendor Triton PRs. pytorch/pytorch tags
    # many Triton PRs with `architectures: [sm100]` because the Triton
    # backend has SM100 support generally, even when the PR itself is
    # scoped to a different GPU backend (ROCm / AMD / Intel GPU / XPU /
    # HIP / MPS / CPU). Concrete examples:
    #   pr-pytorch-170190: "[ROCm] Enable shared memory based pruning..."
    #   pr-pytorch-163388: "[Inductor][Intel GPU] Save threads_per_warp..."
    # The convention is `[<Vendor>]` prefix in the PR title. Match
    # case-insensitively so `[rocm]` / `[Intel GPU]` etc. all catch.
    import re as _re
    _NON_BLACKWELL_VENDOR_RE = _re.compile(
        r"\[\s*(rocm|amd|hip|intel\s*gpu|xpu|mps|cpu)\s*\]",
        _re.IGNORECASE,
    )
    title = str(pr.get("title") or "")
    if _NON_BLACKWELL_VENDOR_RE.search(title):
        return False, (
            f"Triton PR tagged for a non-Blackwell vendor backend in title "
            f"({title.split(']')[0]}]); SM100 metadata is incidental Triton "
            f"backend coverage, not Blackwell kernel authorship"
        )
    # Skip runtime-config-only
    if changed_paths:
        runtime_only_globs = ["**/config/**", "**/__init__.py"]
        if all(any(fnmatch(p, g) for g in runtime_only_globs) for p in changed_paths):
            return False, "runtime-config-only Triton PR"
    # R19: skip PRs whose every changed path is dropped by
    # scripts/fetch_pr_diff.py's SKIP_GLOBS (tests / benchmarks / docs /
    # ci). The sm100-integration sub-scope would otherwise match on
    # basename patterns like benchmark_*_triton.py even when every path
    # lives under benchmark/, emitting a diff.patch-only bundle.
    # Concrete case: pr-sglang-20305, where all 30+ paths are under
    # benchmark/kernels/.
    if all_paths_would_be_skipped_by_fetch(changed_paths):
        return False, (
            "all changed_paths are tests/benchmarks/docs "
            "(empty bundle after fetch_pr_diff.SKIP_GLOBS)"
        )

    # Sub-scope matching
    # memory-bound-kernel
    if symptoms & {"memory-bound", "low-sm-utilization"}:
        return True, None
    # sm100-integration
    if "sm100" in archs:
        # Directory-scoped patterns catch files living under a triton tree,
        # and basename-scoped patterns catch Triton backend-dispatch files
        # like vllm/v1/attention/backends/triton_attn.py or
        # vllm/v1/attention/ops/triton_reshape_and_cache_flash.py that sit
        # outside any triton/ directory but are unambiguously Triton code.
        integ_globs = [
            "**/triton_kernels/**",
            "**/triton/**",
            "**/triton_*.py",
            "**/*_triton.py",
            "**/*_triton_*.py",
        ]
        if any(fnmatch(p, g) for p in changed_paths for g in integ_globs):
            return True, None
    # backend-fallback (heuristic string match). Include spaced and
    # hyphenated variants because upstream titles use "Fall back to triton
    # MOE" (pr-sglang-21780) as commonly as the compound "fallback".
    fallback_keywords = (
        "fallback",
        "fall back",
        "fall-back",
        "falls back",
        "falling back",
        "sm100 path",
        "cutlass is unavailable",
    )
    for kw in fallback_keywords:
        if kw in desc:
            return True, None

    return False, "Triton PR outside the three in-policy sub-scopes"


def dump_sorted(d, sort_keys=False):
    """Stable dump for reproducibility."""
    return yaml.dump(d, allow_unicode=True, sort_keys=sort_keys, default_flow_style=False)


def compute_sha256_of_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_universe(policy, all_prs, apply_fn):
    """Return list of {id, captured, skipped_reason?} for all PRs that the lane
    considers candidates."""
    entries = []
    for pid in sorted(all_prs.keys()):
        pr = all_prs[pid]
        captured, reason = apply_fn(policy, pr)
        if captured:
            entries.append({"id": pid, "captured": True})
        else:
            entries.append({"id": pid, "captured": False, "skipped_reason": reason})
    return entries


def build_cute_universe(policy, all_prs):
    """CuTe lane universe: iterate the full PR corpus (no prefilter) and record
    every PR that the policy has an opinion on.

    A PR belongs to the CuTe universe iff at least one of the capture criteria
    matches (language tag, tag, or changed_paths glob). We record whether the
    policy captured it or explicitly skipped it (e.g., docs-only). PRs that
    match no CuTe entry condition at all are NOT included in the universe —
    recording every one of the 460 PRs would dilute the lane's meaning.
    """
    entries = []
    # R27: share the same STRONG path-signal list as apply_cute_dsl_policy
    # so the universe "is-a-candidate" test uses the identical criterion
    # as the capture decision. Previously the universe test pulled its
    # globs from the YAML, which drifted out of sync with the code when
    # the two-tier STRONG / KERNEL_OR_EXAMPLE split was introduced.
    cute_globs = list(_STRONG_CUTE_GLOBS)

    for pid in sorted(all_prs.keys()):
        pr = all_prs[pid]
        langs = set(pr.get("languages") or [])
        tags = set(pr.get("tags") or [])
        changed_paths = pr.get("changed_paths") or []
        path_match = path_matches_any(changed_paths, cute_globs) if cute_globs else False
        if not ("cute-dsl" in langs or "cute-dsl" in tags or path_match):
            continue  # not a CuTe candidate at all
        captured, reason = apply_cute_dsl_policy(policy, pr)
        if captured:
            entries.append({"id": pid, "captured": True})
        else:
            entries.append({"id": pid, "captured": False, "skipped_reason": reason})
    return entries


def filter_candidates(universe, predicate):
    """Return list of universe entries where predicate(pr) is True."""
    return [e for e in universe if predicate(e)]


def universe_text(entries, lane_name, captured_count, skipped_count):
    """Render the universe-file YAML text (same bytes write_universe_file
    would write to disk). Kept separate from disk I/O so compute_manifests()
    can return the bytes without touching the filesystem."""
    payload = {
        "lane": lane_name,
        "generated_by": "scripts/compute_core_prs.py",
        "total": len(entries),
        "captured": captured_count,
        "skipped": skipped_count,
        "prs": entries,
    }
    return (
        f"## Auto-generated by scripts/compute_core_prs.py — DO NOT EDIT BY HAND.\n"
        f"## Re-run the script after any corpus change.\n\n"
        + dump_sorted(payload)
    )


def write_universe_file(path, entries, lane_name, captured_count, skipped_count):
    path.write_text(
        universe_text(entries, lane_name, captured_count, skipped_count),
        encoding="utf-8",
    )


def compute_manifests():
    """Library entry point: return a dict of {filename: bytes} for the three
    generated manifests, computed purely from the in-repo source + policy
    files. No filesystem writes. Used by scripts/verify_core_prs.py to run
    the reproducibility check in fully read-only sandboxed environments.

    Callers get the same byte output the CLI writer would produce, so the
    verifier can compare against committed data/<name> bytes directly.
    """
    policy_path = DATA / "inclusion-policy.yaml"
    if not policy_path.is_file():
        raise RuntimeError("data/inclusion-policy.yaml not found")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}

    allowlist_path = DATA / "core-prs-allowlist.yaml"
    allowlist = {}
    if allowlist_path.is_file():
        allowlist = yaml.safe_load(allowlist_path.read_text(encoding="utf-8")) or {}

    all_prs = load_all_prs()
    cute_entries = build_cute_universe(policy, all_prs)
    triton_candidates = {
        pid: pr for pid, pr in all_prs.items()
        if "triton" in (pr.get("languages") or []) or "triton" in (pr.get("tags") or [])
    }
    triton_entries = build_universe(policy, triton_candidates, apply_triton_policy)
    cute_captured = {e["id"] for e in cute_entries if e["captured"]}
    triton_captured = {e["id"] for e in triton_entries if e["captured"]}

    graph_ids = graph_closure_from_wiki() & set(all_prs.keys())
    contest_ids = contest_referenced_prs() & set(all_prs.keys())
    core = set()
    source_of = {}

    def add(pid, src):
        if pid in all_prs and pid not in core:
            core.add(pid)
            source_of[pid] = src

    for pid in sorted(graph_ids):
        add(pid, "wiki-graph-closure")
    for pid in sorted(contest_ids):
        if pid not in core:
            add(pid, "contest-reference")
    for pid in sorted(cute_captured):
        if pid not in core:
            add(pid, "cute-dsl-tutorial")
    for pid in sorted(triton_captured):
        if pid not in core:
            add(pid, "triton-in-policy")

    additions = (allowlist.get("additions") or []) if isinstance(allowlist.get("additions"), list) else []
    for e in additions:
        if isinstance(e, dict) and "pr" in e:
            add(e["pr"], "allowlist")
    exclusions = set()
    raw_exclusions = allowlist.get("exclusions") or []
    if isinstance(raw_exclusions, list):
        for e in raw_exclusions:
            if isinstance(e, dict) and "pr" in e:
                exclusions.add(e["pr"])
    core -= exclusions
    for x in list(source_of):
        if x in exclusions:
            del source_of[x]

    prs_list = [{"id": pid, "source_of_inclusion": source_of[pid]} for pid in sorted(core)]
    checksum_body = yaml.dump(prs_list, allow_unicode=True, sort_keys=False, default_flow_style=False)
    checksum = compute_sha256_of_text(checksum_body)

    core_payload = {
        "generated_by": "scripts/compute_core_prs.py",
        "sources": ["wiki-graph-closure", "contest-reference", "cute-dsl-tutorial", "triton-in-policy", "allowlist"],
        "total_captured": len(prs_list),
        "checksum_sha256": checksum,
        "prs": prs_list,
    }
    core_text = (
        "## Auto-generated by scripts/compute_core_prs.py — DO NOT EDIT BY HAND.\n"
        "## Use data/core-prs-allowlist.yaml to add or exclude PRs; then re-run the script.\n\n"
        + yaml.dump(core_payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    )

    cute_text = universe_text(
        cute_entries, "cute-dsl",
        sum(1 for e in cute_entries if e["captured"]),
        sum(1 for e in cute_entries if not e["captured"]),
    )
    triton_text = universe_text(
        triton_entries, "triton",
        sum(1 for e in triton_entries if e["captured"]),
        sum(1 for e in triton_entries if not e["captured"]),
    )

    return {
        "core-prs.yaml": core_text.encode("utf-8"),
        "cute-dsl-universe.yaml": cute_text.encode("utf-8"),
        "triton-universe.yaml": triton_text.encode("utf-8"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir",
        default=str(DATA),
        help="Directory to write core-prs.yaml, cute-dsl-universe.yaml, and "
             "triton-universe.yaml into. Defaults to ./data.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Instead of writing to disk, print the three manifests to stdout "
             "separated by fenced '### FILE: <name>' markers. Useful in "
             "read-only environments where no writable directory is "
             "available; scripts/verify_core_prs.py also calls into "
             "compute_manifests() directly to avoid disk I/O.",
    )
    args = parser.parse_args()

    try:
        manifests = compute_manifests()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    if args.stdout:
        # Markers and payloads both go to the underlying binary buffer so
        # their ordering is deterministic when stdout is piped. Mixing
        # print() (text layer, buffered separately) with buffer.write()
        # (bytes layer) reorders output under pipes — each FILE marker
        # would land after its payload instead of before it.
        for name, payload in manifests.items():
            sys.stdout.buffer.write(f"### FILE: {name}\n".encode("utf-8"))
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        return

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in manifests.items():
        (out_dir / name).write_bytes(payload)

    # Summary derivable from bytes we just wrote.
    core_doc = yaml.safe_load(manifests["core-prs.yaml"].decode("utf-8"))
    cute_doc = yaml.safe_load(manifests["cute-dsl-universe.yaml"].decode("utf-8"))
    triton_doc = yaml.safe_load(manifests["triton-universe.yaml"].decode("utf-8"))
    print(f"core-prs.yaml: {core_doc.get('total_captured', 0)} PRs captured "
          f"(checksum {str(core_doc.get('checksum_sha256', ''))[:12]}...)")
    print(f"cute-dsl-universe.yaml: {cute_doc.get('total', 0)} candidates "
          f"({cute_doc.get('captured', 0)} captured)")
    print(f"triton-universe.yaml: {triton_doc.get('total', 0)} candidates "
          f"({triton_doc.get('captured', 0)} captured)")


if __name__ == "__main__":
    main()
