#!/usr/bin/env python3
"""Collect contest submission code into artifacts/contests/<contest>/<problem>/submissions/<rank-N-author>/.

Phase 3 tasks 9/10. For each `sources/contests/**/*.md` file, iterate its
`submissions[*]` entries and, where a public, author-republished code source
exists (GitHub repo, personal blog's published code, participant's shared
repo), fetch the code into the implicit submission bundle. Otherwise set
`submission_truth: unavailable` with a concrete `code_unavailable_reason`
citing exactly why the code is not locally retrievable.

Strategy:
  - Drive collection from a per-contest manifest, `data/contest-sources.yaml`,
    that maps each submission (contest / problem / rank / author) to an
    optional `origin_url` + kind ({github-file, inline-from-blog, discord-only, unavailable-public}).
  - For github-file: fetch via `gh api contents/...?ref=<sha>`.
  - For inline-from-blog: look up the originating blog in sources/blogs/ and
    copy the matching extracted code from artifacts/blogs/<slug>/code/.
  - For discord-only / unavailable-public: set submission_truth=unavailable
    and write the reason.

By default the manifest is empty, which means all submissions remain
`unavailable` with a structural reason. When an entry gains a concrete source,
add it to data/contest-sources.yaml and re-run.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONTESTS_SRC = REPO / "sources" / "contests"
CONTESTS_ART = REPO / "artifacts" / "contests"
MANIFEST = REPO / "data" / "contest-sources.yaml"


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gh(args):
    res = subprocess.run(["gh"] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return res.stdout


def rank_author_slug(rank, participant):
    participant_slug = re.sub(r"[^a-z0-9]+", "-", participant.lower()).strip("-") or "anon"
    return f"rank-{rank}-{participant_slug}"


def load_md(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^(---\s*\n)(.*?)(\n---\s*\n)(.*)", text, re.DOTALL)
    if not m:
        return None
    return m, yaml.safe_load(m.group(2)) or {}, text


def save_md(path, original_match, fm):
    new_fm = yaml.dump(fm, allow_unicode=True, sort_keys=False, default_flow_style=False).rstrip()
    path.write_text(original_match.group(1) + new_fm + original_match.group(3) + original_match.group(4), encoding="utf-8")


def load_manifest():
    if not MANIFEST.is_file():
        return {}
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}


def _resolve_canonical_file(entry, files_list):
    """Return the bundle-local filename that should be used as the
    submission's `code_path`.

    If the manifest entry declares `canonical_file`, that name MUST
    match one of the bundle's `local_path`s (returns None otherwise so
    the caller can flag a configuration error). Without `canonical_file`
    the final file in `files_list` is chosen because contest
    progression-style bundles (step 1 -> ... -> step N) record the
    winning kernel last; using `files_list[0]` pointed at the reference
    / baseline fragment instead, which was the Round-20 bug Codex
    flagged (R21 P2).
    """
    declared = entry.get("canonical_file") if isinstance(entry, dict) else None
    if declared:
        for f in files_list:
            if f.get("local_path") == declared:
                return declared
        return None
    if not files_list:
        return None
    return files_list[-1].get("local_path")


def collect_one(contest_page, sub_idx, sub, manifest, contest_page_id=None):
    """Returns (new_sub_dict, wrote_any_files: bool, error: str|None)."""
    contest_slug = contest_page.parent.name
    problem_slug = contest_page.stem
    rank = sub.get("rank")
    participant = sub.get("participant") or "anon"
    ra_slug = rank_author_slug(rank, participant)
    bundle_dir = CONTESTS_ART / contest_slug / problem_slug / "submissions" / ra_slug

    # Consult manifest: data/contest-sources.yaml structured as:
    # {contest: {problem: {rank-N-author-slug: {kind, origin_url, ...}}}}
    entry = ((manifest.get(contest_slug) or {}).get(problem_slug) or {}).get(ra_slug)
    if not entry:
        # No manifest entry — remain unavailable with structural reason.
        # R28: also delete any previously-collected bundle directory.
        # Without this, a manifest entry that's removed leaves stale
        # code under artifacts/contests/... which `query.py --has-code`
        # and `get_page.py --include-code` would still surface even
        # though the contest page declares the submission unavailable.
        if bundle_dir.is_dir():
            shutil.rmtree(bundle_dir)
            # Clean up the now-empty `submissions/` parent if we were
            # the last entry (keeps the tree tidy; failure is non-fatal).
            parent = bundle_dir.parent
            try:
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
        new_sub = dict(sub)
        new_sub["submission_truth"] = "unavailable"
        new_sub["code_unavailable_reason"] = (
            "No public author-republished source declared in data/contest-sources.yaml; "
            "add an entry if a public URL becomes available."
        )
        new_sub.pop("code_path", None)
        return new_sub, False, None

    kind = entry.get("kind")
    origin_url = entry.get("origin_url", "")

    new_sub = dict(sub)
    files_list = []

    # For unavailable kinds, never create the bundle directory.
    # R28: use rmtree for full recursion; the previous iter+unlink loop
    # would leave stale subdirectories behind when a bundle that had
    # nested key-files was reclassified to discord-only / unavailable.
    if kind in ("discord-only", "unavailable-public"):
        if bundle_dir.is_dir():
            shutil.rmtree(bundle_dir)
        new_sub["submission_truth"] = "unavailable"
        new_sub["code_unavailable_reason"] = entry.get("reason", f"kind={kind}; no public code")
        new_sub.pop("code_path", None)
        return new_sub, False, None

    # R25: atomic-swap bundle replacement. All writes go into a sibling
    # `.new` staging dir; the prior bundle stays intact until the new
    # one is complete. On any error (bad canonical_file, missing blog
    # bundle, gh failure, mid-copy OS error) the staging dir is removed
    # and the prior bundle is preserved, so `code_path` in the contest
    # page never points at a half-written or missing submission.

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    bundle_work = bundle_dir.parent / f".{bundle_dir.name}.new"
    if bundle_work.exists():
        shutil.rmtree(bundle_work)
    bundle_work.mkdir(parents=True)
    # All writes target `bundle_work`; only the final atomic swap
    # touches `bundle_dir`. Keep the local name `bundle_dir` pointed at
    # the staging dir so the unchanged write-site code still works.
    bundle_dir_final = bundle_dir
    bundle_dir = bundle_work

    # Flag flipped to True immediately before a success return; the
    # finally clause below uses it to decide whether to atomically swap
    # the staging dir into place or just clean it up.
    _swap_on_exit = [False]
    try:
        if kind == "github-file":
            repo = entry["upstream_repo"]
            sha = entry.get("upstream_sha")
            if not sha or str(sha).strip().upper() in ("HEAD", "MAIN", "MASTER", ""):
                # Refuse to fetch at a moving reference. A github-file entry
                # without a pinned commit SHA would produce a non-deterministic
                # capture (the content of the default branch changes over
                # time) that scripts/verify_verbatim.py can't SHA-pin.
                return new_sub, False, (
                    f"manifest kind=github-file for {repo} needs a pinned "
                    f"upstream_sha (40-char commit SHA or short prefix); "
                    f"refusing to fetch at '{sha or 'missing'}' which would "
                    f"produce an unpinned, unverifiable bundle"
                )
            files_spec = entry.get("files", [])
            for fs in files_spec:
                src_path = fs["upstream_path"]
                out_name = fs.get("local_name") or src_path.split("/")[-1]
                out = bundle_dir / out_name
                data = run_gh(["api", f"/repos/{repo}/contents/{src_path}?ref={sha}"])
                doc = json.loads(data)
                if doc.get("type") != "file":
                    return new_sub, False, f"non-file response for {repo}:{src_path}"
                out.write_bytes(base64.b64decode(doc["content"]))
                files_list.append({
                    "local_path": out_name,
                    "role": "upstream-file",
                    "mode": "verbatim",
                    "upstream_path": src_path,
                    "sha256": sha256_of(out),
                })
            truth = "official-submission" if entry.get("official") else "author-published-posthoc"
            # Write PROVENANCE.yaml
            prov = {
                "origin_url": origin_url,
                "upstream_repo": repo,
                "upstream_sha": sha,
                "license": entry.get("license", "inherits-from-upstream"),
                "retrieved_at": "2026-04-17",
                "asset_mode": "verbatim",
                "size_cap_truncated": False,
                "generated_by": "scripts/collect_contest_code.py",
                "source_contest_id": contest_page_id or f"contest-{contest_slug}-{problem_slug}",
                "files": files_list,
            }
            (bundle_dir / "PROVENANCE.yaml").write_text(
                yaml.dump(prov, sort_keys=False, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            new_sub["submission_truth"] = truth
            canonical = _resolve_canonical_file(entry, files_list)
            if canonical is None:
                return new_sub, False, (
                    f"manifest canonical_file='{entry.get('canonical_file')}' does not "
                    f"match any local_path in the fetched bundle; fix the manifest"
                )
            new_sub["code_path"] = f"artifacts/contests/{contest_slug}/{problem_slug}/submissions/{ra_slug}/{canonical}"
            new_sub.pop("code_unavailable_reason", None)
            _swap_on_exit[0] = True
            return new_sub, True, None

        elif kind == "inline-from-blog":
            blog_slug = entry["blog_slug"]
            blog_bundle = REPO / "artifacts" / "blogs" / blog_slug
            blog_code_dir = blog_bundle / "code"
            if not blog_code_dir.is_dir():
                return new_sub, False, f"blog code bundle {blog_slug} not extracted yet"
            # Load the blog's MANIFEST.yaml so we can copy the real heading_path
            # for each file into the contest bundle's PROVENANCE.yaml (instead of
            # writing a placeholder inheritance string).
            blog_manifest_path = blog_bundle / "MANIFEST.yaml"
            blog_manifest_files = {}
            if blog_manifest_path.is_file():
                try:
                    bm = yaml.safe_load(blog_manifest_path.read_text(encoding="utf-8")) or {}
                    for e in (bm.get("files") or []):
                        lp = e.get("local_path")
                        if lp and lp.startswith("code/"):
                            blog_manifest_files[lp[len("code/"):]] = e
                except yaml.YAMLError:
                    pass
            # Copy each named file into the submission bundle
            want = set(entry.get("files") or [])
            for f in sorted(blog_code_dir.iterdir()):
                if f.name == "PROVENANCE.yaml" or not f.is_file():
                    continue
                if want and f.name not in want:
                    continue
                dst = bundle_dir / f.name
                shutil.copy(f, dst)
                heading = blog_manifest_files.get(f.name, {}).get("heading_path") or f"(heading not recorded in {blog_slug}/MANIFEST.yaml)"
                files_list.append({
                    "local_path": f.name,
                    "role": "extracted-block",
                    "mode": "extracted",
                    "upstream_path": f"sources/blogs/{blog_slug}.md",
                    "heading_path": heading,
                    "sha256": sha256_of(dst),
                })
            if not files_list:
                return new_sub, False, "no matching files in blog bundle"
            prov = {
                "origin_url": origin_url,
                "upstream_repo": f"blog/{blog_slug}",
                "upstream_sha": "none",
                "license": "inherits-from-source-blog",
                "retrieved_at": "2026-04-17",
                "asset_mode": "extracted",
                "size_cap_truncated": False,
                "generated_by": "scripts/collect_contest_code.py",
                "source_contest_id": contest_page_id or f"contest-{contest_slug}-{problem_slug}",
                "files": files_list,
            }
            (bundle_dir / "PROVENANCE.yaml").write_text(
                yaml.dump(prov, sort_keys=False, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            new_sub["submission_truth"] = "reconstructed-from-blog"
            canonical = _resolve_canonical_file(entry, files_list)
            if canonical is None:
                return new_sub, False, (
                    f"manifest canonical_file='{entry.get('canonical_file')}' does not "
                    f"match any local_path in the reconstructed bundle for {blog_slug}"
                )
            new_sub["code_path"] = f"artifacts/contests/{contest_slug}/{problem_slug}/submissions/{ra_slug}/{canonical}"
            new_sub.pop("code_unavailable_reason", None)
            _swap_on_exit[0] = True
            return new_sub, True, None

        else:
            return new_sub, False, f"unknown kind '{kind}'"
    except subprocess.CalledProcessError as e:
        return new_sub, False, f"gh failed: {e.stderr.decode(errors='replace').strip()[:120]}"
    except Exception as e:
        return new_sub, False, f"{type(e).__name__}: {e}"
    finally:
        # Atomic swap on success; cleanup on any failure path. Running
        # in `finally` guarantees this fires even for the `return` paths
        # above — without it, a partial capture could replace the old
        # bundle before the caller sees the error.
        #
        # R29: if the swap itself fails (destination locked, cross-
        # filesystem rename, etc.), raise so `collect_one()`'s caller
        # treats the submission as failed. Previously the `finally`
        # silently swallowed OSError, leaving `collect_one` to return
        # (new_sub, True, None) from the earlier `return` while the
        # new bundle never actually installed — the contest page was
        # then rewritten with a `code_path` pointing at a stale /
        # missing directory. Raising from `finally` overrides the
        # pending return value, which is the Python semantic we want.
        if _swap_on_exit[0]:
            bundle_prev = None
            if bundle_dir_final.exists():
                bundle_prev = bundle_dir_final.parent / f".{bundle_dir_final.name}.prev"
                if bundle_prev.exists():
                    shutil.rmtree(bundle_prev, ignore_errors=True)
                try:
                    os.rename(bundle_dir_final, bundle_prev)
                except OSError:
                    bundle_prev = None
            try:
                os.rename(bundle_work, bundle_dir_final)
            except OSError as e:
                if bundle_prev is not None and bundle_prev.exists():
                    try:
                        os.rename(bundle_prev, bundle_dir_final)
                    except OSError:
                        pass
                # Remove the orphan staging dir before re-raising, so a
                # failing swap doesn't leave `.<name>.new` lying around
                # between runs.
                if bundle_work.exists():
                    shutil.rmtree(bundle_work, ignore_errors=True)
                # Propagate: the main loop wraps collect_one and
                # converts this into a soft failure for the fails list.
                raise RuntimeError(
                    f"atomic bundle swap failed for {bundle_dir_final}: {e}; "
                    f"prior bundle restored if possible"
                )
            if bundle_prev is not None and bundle_prev.exists():
                shutil.rmtree(bundle_prev, ignore_errors=True)
        else:
            # Failure: remove the staging dir so the prior bundle (if
            # any) remains the committed truth.
            if bundle_work.exists():
                shutil.rmtree(bundle_work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    wrote = 0
    unavailable = 0
    fails = []

    for contest_md in sorted(CONTESTS_SRC.rglob("*.md")):
        loaded = load_md(contest_md)
        if not loaded:
            continue
        mmatch, fm, _ = loaded
        subs = fm.get("submissions") or []
        if not subs:
            continue
        new_subs = []
        changed = False
        for i, sub in enumerate(subs):
            if not isinstance(sub, dict):
                new_subs.append(sub)
                continue
            if args.dry_run:
                new_subs.append(sub)
                continue
            # R28: pass the contest page's real frontmatter id so the
            # emitted PROVENANCE `source_contest_id` links back to the
            # right source page. Path-derived ids (contest-<dir>-<stem>)
            # don't match curator-assigned ids like `contest-gpumode-p1`.
            page_id = fm.get("id") if isinstance(fm, dict) else None
            try:
                new_sub, ok, err = collect_one(contest_md, i, sub, manifest, contest_page_id=page_id)
            except RuntimeError as swap_err:
                # R29: collect_one's finally raises on atomic-swap
                # failure. Treat as a soft failure so the run exits
                # non-zero but doesn't crash the whole collection.
                # Critically, don't treat as ok=True: preserve the
                # ORIGINAL `sub` dict so main() doesn't rewrite the
                # contest page's submission_truth / code_path fields
                # to point at a bundle that never installed.
                new_sub, ok, err = dict(sub), False, f"swap failed: {swap_err}"
            if err:
                fails.append(f"{contest_md.relative_to(REPO)}[{i}]: {err}")
                # R33: preserve the ORIGINAL submission dict on failure
                # so the contest page doesn't drift ahead of the
                # actually-installed bundles. collect_one() may have
                # mutated fields like `submission_truth` before
                # raising, and appending that partial state would
                # persist metadata that no longer matches
                # artifacts/contests/**.
                new_sub = dict(sub)
            if ok:
                wrote += 1
            elif new_sub.get("submission_truth") == "unavailable":
                unavailable += 1
            if new_sub != sub:
                changed = True
            new_subs.append(new_sub)
        if changed:
            fm["submissions"] = new_subs
            save_md(contest_md, mmatch, fm)

    print(f"Collected code for {wrote} submissions; marked {unavailable} as unavailable.")
    if fails:
        # R21: a soft failure used to fall through to exit 0, which let
        # sources/contests/** drift silently from data/contest-sources.yaml
        # in CI. A manifest typo, missing blog bundle, or gh fetch error
        # now terminates the run non-zero so regeneration workflows fail
        # loudly instead of committing a half-collected state.
        print(f"\n{len(fails)} soft failure(s):", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
