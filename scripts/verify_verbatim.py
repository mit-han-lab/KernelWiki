#!/usr/bin/env python3
"""Verify every asset_mode=verbatim / upstream-patch file against upstream.

Walks artifacts/*/PROVENANCE.yaml, finds each files[*] whose mode is 'verbatim'
or 'upstream-patch', fetches the upstream content via `gh api` (for verbatim)
or `gh pr diff` (for upstream-patch), and compares it with the local asset.

Verbatim files are byte-compared. GitHub-generated PR diffs are compared after
canonicalizing only the abbreviated object IDs on ``index`` lines. Git chooses
the abbreviation length dynamically as a repository grows, so those IDs can
lengthen even though the pinned PR patch is semantically byte-for-byte
unchanged. The merge-commit prefix is still verified before patch comparison.

Exit codes:
  0 — all verbatim/upstream-patch assets match their pinned upstream
  1 — at least one mismatch (reported to stderr)
  2 — invocation / environment error (missing gh, network, bad input)

Usage:
  scripts/verify_verbatim.py            # warn-only
  scripts/verify_verbatim.py --strict   # fail on any mismatch
  scripts/verify_verbatim.py --bundle artifacts/prs/cutlass/PR-2161
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
import yaml
import base64
import json


_PATCH_INDEX_RE = re.compile(
    rb"^(index )[0-9a-fA-F]+(\.\.)[0-9a-fA-F]+(.*)$", re.MULTILINE
)


def canonicalize_upstream_patch(data):
    """Remove Git's unstable abbreviation length from patch ``index`` lines.

    The remainder of each index line (including file mode) and every patch
    header/hunk byte remain part of the comparison oracle.
    """
    return _PATCH_INDEX_RE.sub(rb"\1<OLD>\2<NEW>\3", data)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


class EnvError(RuntimeError):
    """Environment / invocation failure (no gh, no network, auth missing, etc).

    Separated from RuntimeError so the script's documented exit-code contract
    can distinguish env failures (exit 2) from true upstream byte mismatches
    (exit 1).
    """


# Heuristic substrings that flag a network / environment problem rather than
# a real content mismatch. Matched case-insensitively on gh's stderr. The
# list is intentionally broad because a false positive (treating a real
# upstream problem as env) surfaces as exit 2 and invites investigation,
# while a false negative (treating a real env failure as a content mismatch)
# produces a deceptive exit code — the Round-5 blocker.
_ENV_ERROR_HINTS = (
    # Resolver / DNS
    "could not resolve host",
    "no such host",
    "temporary failure in name resolution",
    "unable to resolve",
    # TCP / transport
    "connection refused",
    "connection reset",
    "connection timed out",
    "network is unreachable",
    "no route to host",
    "dial tcp",
    "i/o timeout",
    "context deadline exceeded",
    "request timed out",
    # Generic transport strings used by gh's http client (covers the exact
    # "error connecting to api.github.com" message Codex reported)
    "error connecting",
    "failed to connect",
    "unable to connect",
    "couldn't connect",
    "cannot reach",
    "can't reach",
    # TLS / cert
    "x509:",
    "tls handshake",
    "certificate signed by unknown authority",
    "certificate has expired",
    "self signed certificate",
    # Proxy
    "proxy error",
    "proxyconnect",
    # Auth / account state (missing auth => env, not content)
    "authentication required",
    "you must authenticate",
    "you are not logged in",
    "not logged into",
    "bad credentials",
    "token expired",
    "sso enforcement",
    # Quota
    "rate limit",
    "api rate limit exceeded",
    "abuse detection",
    "secondary rate limit",
)


def _looks_like_env_error(stderr_text):
    low = stderr_text.lower()
    return any(h in low for h in _ENV_ERROR_HINTS)


def run_gh(args):
    """Run gh CLI. Returns stdout bytes on success; raises EnvError for env /
    network / auth failures, or RuntimeError for anything else gh rejects."""
    try:
        res = subprocess.run(
            ["gh"] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return res.stdout
    except FileNotFoundError:
        raise EnvError("gh CLI not found; install from https://cli.github.com/")
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode(errors="replace")
        if _looks_like_env_error(stderr_text):
            raise EnvError(f"gh {' '.join(args)} environment failure: {stderr_text.strip()[:200]}")
        raise RuntimeError(f"gh {' '.join(args)} failed: {stderr_text.strip()[:200]}")


def fetch_verbatim(upstream_repo, upstream_sha, upstream_path):
    """Fetch a single file's bytes from GitHub at the pinned SHA."""
    # gh api /repos/{owner}/{repo}/contents/{path}?ref={sha}
    endpoint = f"/repos/{upstream_repo}/contents/{upstream_path}?ref={upstream_sha}"
    out = run_gh(["api", endpoint])
    data = json.loads(out)
    if "content" not in data:
        raise RuntimeError(f"no content in response for {upstream_repo}:{upstream_path}@{upstream_sha}")
    return base64.b64decode(data["content"])


def fetch_upstream_patch(upstream_repo, pr_number, expected_sha=None):
    """Fetch the PR's diff at the declared upstream_sha.

    Strategy:
      1. GET /repos/{repo}/pulls/{N} to read merge_commit_sha.
      2. If expected_sha is provided, it must prefix-match
         merge_commit_sha (our stored SHAs are often 8-char shortcuts,
         so we accept prefix match). Mismatch -> hard error: the PR
         has been amended upstream and the patch we shipped no longer
         corresponds to the state at our pinned SHA. R30: removed the
         fallback to head.sha — a stale merge_sha pointing at head
         (after squash/rebase merge, or when the PR branch was kept
         alive) would have silently passed otherwise.
      3. Return the stable diff via `gh pr diff`; for merged PRs the
         diff is frozen at merge_commit_sha, so this byte-matches what
         we shipped.
    """
    if expected_sha:
        pr_json = run_gh(["api", f"/repos/{upstream_repo}/pulls/{pr_number}"])
        pr_data = json.loads(pr_json)
        merge_sha = pr_data.get("merge_commit_sha") or ""
        if not (merge_sha and merge_sha.startswith(expected_sha)):
            head_sha = (pr_data.get("head") or {}).get("sha") or ""
            raise RuntimeError(
                f"upstream_sha {expected_sha!r} does not prefix-match "
                f"merge_commit_sha={merge_sha[:12]}... "
                f"(head.sha={head_sha[:12]}... shown for reference; "
                f"not accepted) for {upstream_repo}#{pr_number}; "
                f"the PR was amended upstream"
            )
    out = run_gh(["pr", "diff", str(pr_number), "-R", upstream_repo])
    return out


PR_URL_RE = re.compile(r"github\.com/[^/]+/[^/]+/pull/(\d+)")
PR_ID_RE = re.compile(r"pr-[a-z0-9\-]+-(\d+)")
PR_FILENAME_RE = re.compile(r"PR[_-](\d+)", re.IGNORECASE)
PR_DIR_RE = re.compile(r"PR-(\d+)")


def _resolve_pr_number(bundle_root, prov, local_path):
    """Resolve the upstream PR number for an upstream-patch file.

    Resolution order (all lowered to a number string):
      1. bundle directory name matches PR-<N>
      2. PROVENANCE.yaml source_pr_id ends in -<N>
      3. PROVENANCE.yaml origin_url matches github.com/.../pull/<N>
      4. local_path filename contains PR-<N> or PR_<N>
    """
    m = PR_DIR_RE.fullmatch(bundle_root.name)
    if m:
        return m.group(1)
    src_id = (prov or {}).get("source_pr_id") or ""
    m = PR_ID_RE.fullmatch(src_id)
    if m:
        return m.group(1)
    origin_url = (prov or {}).get("origin_url") or ""
    m = PR_URL_RE.search(origin_url)
    if m:
        return m.group(1)
    m = PR_FILENAME_RE.search(local_path)
    if m:
        return m.group(1)
    return None


def iter_bundles(scope):
    if scope:
        yield Path(scope).resolve()
        return
    if not ARTIFACTS_DIR.is_dir():
        return
    for prov in ARTIFACTS_DIR.rglob("PROVENANCE.yaml"):
        yield prov.parent


def _rel_to_repo(p):
    """Format a path relative to REPO_ROOT when possible, otherwise return the
    absolute path. Lets --bundle point at a directory outside the repo
    (pipeline / symlink scenarios) without the error path crashing on
    Path.relative_to."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def verify_bundle(bundle_root):
    """Return (content_errors, env_errors) — two separate lists so the main
    loop can distinguish true upstream byte mismatches from environment or
    network failures when choosing its exit code.
    """
    prov_path = bundle_root / "PROVENANCE.yaml"
    if not prov_path.is_file():
        return [f"{_rel_to_repo(bundle_root)}: missing PROVENANCE.yaml"], []
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [f"{_rel_to_repo(prov_path)}: YAML parse error: {e}"], []

    upstream_repo = prov.get("upstream_repo")
    upstream_sha = prov.get("upstream_sha")
    errors = []
    env_errors = []

    for i, entry in enumerate(prov.get("files") or []):
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode")
        if mode not in ("verbatim", "upstream-patch"):
            continue
        if entry.get("size_cap_truncated"):
            continue
        lp = entry.get("local_path")
        if not lp:
            continue
        local_path = bundle_root / lp
        if not local_path.is_file():
            errors.append(f"{_rel_to_repo(bundle_root)}/{lp}: local file missing")
            continue
        local_bytes = local_path.read_bytes()

        # Per-file overrides beat bundle-level defaults so mixed-source bundles
        # (e.g. DeepGEMM verbatim upstream files + a sglang integration PR
        # patch) can cite the right upstream per file.
        file_upstream_repo = entry.get("upstream_repo") or upstream_repo
        file_upstream_sha = entry.get("upstream_sha") or upstream_sha

        try:
            if mode == "verbatim":
                upstream_path = entry.get("upstream_path")
                if not (file_upstream_repo and file_upstream_sha and upstream_path):
                    errors.append(
                        f"{_rel_to_repo(bundle_root)}/{lp}: verbatim mode requires "
                        f"upstream_repo + upstream_sha + upstream_path"
                    )
                    continue
                upstream_bytes = fetch_verbatim(file_upstream_repo, file_upstream_sha, upstream_path)
            else:  # upstream-patch
                pr_num = _resolve_pr_number(bundle_root, prov, lp)
                if not pr_num:
                    errors.append(
                        f"{_rel_to_repo(bundle_root)}/{lp}: upstream-patch mode "
                        f"could not resolve a PR number from source_pr_id, origin_url, "
                        f"patch filename, or bundle directory name"
                    )
                    continue
                if not file_upstream_sha:
                    errors.append(
                        f"{_rel_to_repo(bundle_root)}/{lp}: upstream-patch mode "
                        f"requires upstream_sha (bundle-level or per-file) to pin the "
                        f"verification to the exact upstream state"
                    )
                    continue
                upstream_bytes = fetch_upstream_patch(file_upstream_repo, pr_num, file_upstream_sha)
        except EnvError as e:
            env_errors.append(f"{_rel_to_repo(bundle_root)}/{lp}: environment failure: {e}")
            continue
        except RuntimeError as e:
            errors.append(f"{_rel_to_repo(bundle_root)}/{lp}: upstream fetch failed: {e}")
            continue

        comparable_local = local_bytes
        comparable_upstream = upstream_bytes
        comparison_label = mode
        if mode == "upstream-patch":
            comparable_local = canonicalize_upstream_patch(local_bytes)
            comparable_upstream = canonicalize_upstream_patch(upstream_bytes)
            comparison_label = "canonicalized upstream-patch"

        if comparable_upstream != comparable_local:
            local_sha = hashlib.sha256(local_bytes).hexdigest()[:12]
            upstream_sha12 = hashlib.sha256(upstream_bytes).hexdigest()[:12]
            errors.append(
                f"{_rel_to_repo(bundle_root)}/{lp}: {comparison_label} mismatch "
                f"(local {local_sha}..., upstream {upstream_sha12}...)"
            )
    return errors, env_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any content mismatch (default: warn-only exit 0)")
    parser.add_argument("--bundle", help="Verify a single bundle root instead of all")
    args = parser.parse_args()

    if not ARTIFACTS_DIR.is_dir():
        print("No artifacts/ directory; nothing to verify.")
        sys.exit(0)

    all_errors = []       # content / upstream byte mismatches
    all_env_errors = []   # network / auth / gh-missing / rate-limit failures
    bundles_checked = 0
    for bundle in iter_bundles(args.bundle):
        bundles_checked += 1
        errs, env_errs = verify_bundle(bundle)
        all_errors.extend(errs)
        all_env_errors.extend(env_errs)

    print(f"Verified {bundles_checked} bundle(s).")
    # Print env errors with a distinct marker so operators can tell them apart.
    if all_env_errors:
        for e in all_env_errors:
            print(f"  ENV: {e}", file=sys.stderr)
        print(f"\n{len(all_env_errors)} environment failure(s).", file=sys.stderr)
    if all_errors:
        for e in all_errors:
            print(f"  WARN: {e}", file=sys.stderr)
        print(f"\n{len(all_errors)} mismatch(es) found.", file=sys.stderr)

    # Exit-code contract (matches the file's module docstring):
    #   0 — all verbatim/upstream-patch assets match, or (without --strict) only
    #       content mismatches but no env failures
    #   1 — --strict and at least one true upstream byte mismatch
    #   2 — environment / invocation failure (gh missing, no network, etc)
    #       takes precedence over content errors: a run that could not complete
    #       is NOT the same as a run that completed and found a mismatch.
    if all_env_errors:
        sys.exit(2)
    if all_errors:
        sys.exit(1 if args.strict else 0)
    print("All verbatim/upstream-patch assets match upstream.")
    sys.exit(0)


if __name__ == "__main__":
    main()
