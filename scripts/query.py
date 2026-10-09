#!/usr/bin/env python3
"""Unified query tool for the Blackwell kernel wiki.

Finds keyword candidates with aliases and metadata filters. Results are paginated
in path order; the calling LLM judges relevance from the evidence, not a score.

Usage:
    query.py "gate-up GEMM" --match all --type kernel
    query.py --tag nvfp4 --type kernel
    query.py --repo cutlass --limit 20
    query.py --language cute-dsl
    query.py --symptom memory-bound

Returns matching pages with titles, paths, metadata and bounded match excerpts.
"""

import argparse
import copy
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402
from pr_policy import architecture_matches_filter  # noqa: E402


_ALIAS_CACHE = None


def load_alias_expansions():
    """Return a dict mapping lowercased alias → canonical term, loaded from data/aliases.yaml.

    Each canonical is also mapped to itself, so lookups always return something
    sensible when the user already types the canonical form.
    """
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    out = {}
    aliases_path = WIKI_ROOT / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in (variants or []):
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE = out
    return out


def expand_keyword(kw):
    """Return a list of search variants for a keyword: the original plus any
    canonical term the alias map points to."""
    aliases = load_alias_expansions()
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


def load_frontmatter(path):
    """Parse YAML frontmatter from a markdown file. Returns (fm_dict, body_str) or (None, None)."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            return None, None
        return fm, m.group(2)
    except yaml.YAMLError:
        return None, None


def load_all_pages():
    """Load frontmatter + body summary for every sources/*.md and wiki/*.md file."""
    pages = []
    for subdir in ["sources", "wiki"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            fm, body = load_frontmatter(md)
            if fm is None:
                continue
            pages.append({
                "path": str(md.relative_to(WIKI_ROOT)),
                "fm": fm,
                "body": body or "",
            })
    return pages


def detect_page_type(fm, path):
    """Return a page-type label for filtering: source-pr/source-blog/..., wiki-hardware/..."""
    if "type" in fm:
        return f"wiki-{fm['type']}"
    parts = path.split("/")
    if parts[0] == "sources" and len(parts) > 1:
        return f"source-{parts[1].rstrip('s')}"  # prs → source-pr
    return "unknown"


def keyword_matches(fm, body, keywords):
    """Report matching variants by field, preserving the original OR/alias recall."""
    metadata = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                          "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    )
    variants = list(dict.fromkeys(
        variant.lower() for kw in keywords for variant in expand_keyword(kw) if variant
    ))
    matches = {}
    for name, text in (("title", str(fm.get("title", ""))),
                       ("metadata", metadata), ("body", body)):
        matched = [variant for variant in variants if variant in text.lower()]
        if matched:
            matches[name] = matched
    return matches


def select_candidates(pages, keywords, match_mode="any"):
    """Select keyword matches without assigning relevance scores or priorities."""
    keyword_groups = [
        {variant.lower() for variant in expand_keyword(keyword)} for keyword in keywords
    ] if match_mode == "all" else []
    candidates = []
    for page in pages:
        matches = keyword_matches(page["fm"], page["body"], keywords)
        if not keywords or matches:
            matched_terms = {term for terms in matches.values() for term in terms}
            if not all(group & matched_terms for group in keyword_groups):
                continue
            candidates.append(dict(page, _matches=matches))
    return sorted(candidates, key=lambda page: page["path"])


def match_excerpt(body, variants, width=240):
    """Preview the first body match in context, not a generated summary."""
    text = " ".join(body.split())
    positions = [text.lower().find(" ".join(term.split())) for term in variants]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return ""
    start = max(0, min(positions) - width // 3)
    end = min(len(text), start + width)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def filter_pages(pages, args):
    """Apply CLI filters to narrow the page set."""
    out = []
    for p in pages:
        fm = p["fm"]
        path = p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if args.type and not ptype.endswith(args.type):
            # --type kernel matches wiki-kernel, --type pr matches source-pr
            if ptype != args.type:
                continue

        if args.tag:
            all_tags = set()
            for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(args.tag)}
            if not any(t.lower() in tag_variants for t in all_tags):
                continue

        if args.repo:
            repo = str(fm.get("repo", "")).lower()
            if args.repo.lower() not in repo:
                continue

        if args.language:
            langs = set(fm.get("languages") or [])
            tags = set(fm.get("tags") or [])
            if args.language not in langs and args.language not in tags:
                continue

        if args.architecture:
            if not architecture_matches_filter(
                fm.get("architectures") or [],
                str(fm.get("architecture_disposition", "")),
                args.architecture,
            ):
                continue

        if args.symptom:
            symptoms = set(fm.get("symptoms") or [])
            if args.symptom not in symptoms:
                continue

        if args.confidence:
            if str(fm.get("confidence", "")) != args.confidence:
                continue

        if args.has_code:
            # NOTE: `.patch` is deliberately excluded. --has-code is documented
            # as "at least one source file", so a PR bundle that only shipped
            # diff.patch (no captured key-files/) should NOT surface here —
            # a raw diff is not browsable source. See: get_page.py --include-code
            # still prints diff.patch for bundles that have one.
            #
            # `.txt` IS counted: scripts/extract_blog_code.py writes unlabeled
            # fenced blocks under artifacts/blogs/<slug>/code/NN-<name>.txt via
            # the EXT_MAP.get(lang, "txt") fallback. These are real code
            # snippets whose fence just lacked a language tag (R20 widening).
            #
            # `.sh`, `.yaml`, `.json` are also counted (R33) to match
            # extract_blog_code.py's EXT_MAP mappings for bash / shell /
            # yaml / json fences. Keep this set in sync with
            # validate.py's ASSET_SOURCE_EXTS and get_page.py's
            # --include-code exts; the three together form the Phase-3
            # asset-source contract.
            exts = {".cu", ".cuh", ".ptx", ".py", ".cpp", ".h", ".hpp", ".inl",
                    ".pyx", ".cxx", ".cc", ".txt",
                    ".sh", ".yaml", ".json"}
            candidate_dirs = []

            # Primary: explicit page-level artifact_dir.
            ad = fm.get("artifact_dir")
            if ad:
                candidate_dirs.append(WIKI_ROOT / ad)

            # Fallback 1: conventional bundle locations per page type.
            # A source-blog's extracted code lives at artifacts/blogs/<slug>/code/;
            # a source-contest's reconstructed bundles live under
            # artifacts/contests/<contest>/<problem>/submissions/**.
            # This lets --has-code still work for pages whose artifact_dir
            # hasn't been backfilled yet (Codex R8 finding #1).
            slug = str(fm.get("id", "")).replace("blog-", "")
            if p["_ptype"] == "source-blog":
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "blogs" / Path(path).stem / "code")
            elif p["_ptype"] == "source-contest":
                contest_dir_name = Path(path).parent.name
                problem_stem = Path(path).stem
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "contests" / contest_dir_name / problem_stem)

            # Fallback 2: for source-pr pages, infer the conventional
            # artifacts/prs/<repo-short>/PR-<N>/ location from frontmatter
            # repo+pr fields.
            if p["_ptype"] == "source-pr" and fm.get("repo") and fm.get("pr"):
                repo_short = str(fm["repo"]).split("/")[-1].lower()
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "prs" / repo_short / f"PR-{fm['pr']}")

            has_any = False
            for cand in candidate_dirs:
                if not cand.is_dir():
                    continue
                for f in cand.rglob("*"):
                    if f.is_file() and f.suffix.lower() in exts:
                        has_any = True
                        break
                if has_any:
                    break

            if not has_any:
                continue

        out.append(p)
    return out


def architecture_filter_exclusion_counts(pages, requested):
    """Counts disclosed when an exact query omits broader evidence classes."""
    normalized = requested.lower().replace("_", "")
    if normalized in {"blackwell", "hopper", "ampere", "ada", "unknown"}:
        return None
    family_only = sum(p["fm"].get("architecture_disposition") == "family" for p in pages)
    unknown = sum(
        p["fm"].get("architecture_disposition") == "unknown"
        and not (p["fm"].get("architectures") or [])
        for p in pages
    )
    return family_only, unknown


def format_result(page, compact=False):
    """Format a single page as a readable line."""
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")
    matches = page.get("_matches", {})
    matched = "; ".join(f"{field}: {', '.join(terms)}" for field, terms in matches.items())
    excerpt = match_excerpt(page["body"], matches.get("body", []))

    if compact:
        result = f"  [{ptype}] {pid}: {title}  ({path})"
        if matched:
            result += f" | matches: {matched}"
        if excerpt:
            result += f" | first body match: {excerpt}"
        return result

    # Rich: include key metadata
    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    if matched:
        lines.append(f"- **matches**: {matched}")
    if excerpt:
        lines.append(f"- **first body match (excerpt)**: {excerpt}")
    if "architectures" in fm:
        lines.append(f"- **architectures**: {fm['architectures']}")
    for k in ("confidence", "reproducibility"):
        if k in fm:
            lines.append(f"- **{k}**: {fm[k]}")
    for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
        v = fm.get(k)
        if v:
            lines.append(f"- **{k}**: {v}")
    if "performance_claims" in fm and isinstance(fm["performance_claims"], list):
        for claim in fm["performance_claims"][:2]:
            lines.append(f"- **perf**: {claim.get('value')} {claim.get('metric')} on {claim.get('gpu')} ({claim.get('dtype')}, {claim.get('shape')})")
    if "sources" in fm:
        lines.append(f"- **sources**: {fm['sources'][:5]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query the Blackwell kernel wiki")
    parser.add_argument("query", nargs="*", help="Whitespace-separated keywords; no relevance scoring")
    parser.add_argument("--match", choices=("any", "all"), default="any",
                        help="Match any keyword (default), or every keyword; aliases are alternatives per keyword, not phrase matching")
    parser.add_argument("--type", help="Filter by page type (kernel, technique, hardware, pattern, language, migration, pr, blog, doc, contest)")
    parser.add_argument("--tag", help="Filter by tag (must appear in tags/techniques/hardware_features/kernel_types/languages)")
    parser.add_argument("--repo", help="Filter by source repo (partial match, e.g. 'cutlass')")
    parser.add_argument("--language", help="Filter by language/DSL (cute-dsl, cuda-cpp, ptx, triton, etc.)")
    parser.add_argument("--architecture", help="Filter by exact architecture, blackwell family hierarchy, or unknown")
    parser.add_argument("--symptom", help="Filter by pattern symptom (memory-bound, register-pressure, etc.)")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--has-code", action="store_true", help="Only return pages whose artifact_dir contains at least one source file")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many matching candidates (default 0)")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths, one per line")
    args = parser.parse_args()
    if args.limit < 1 or args.offset < 0:
        parser.error("--limit must be positive and --offset must be nonnegative")

    all_pages = load_all_pages()
    architecture_stats = None
    if args.architecture:
        without_architecture = copy.copy(args)
        without_architecture.architecture = None
        eligible_pages = filter_pages(all_pages, without_architecture)
        pages = filter_pages(eligible_pages, args)
        architecture_stats = architecture_filter_exclusion_counts(
            eligible_pages, args.architecture
        )
    else:
        pages = filter_pages(all_pages, args)

    if architecture_stats is not None:
        family_only, unknown = architecture_stats
        print(
            f"Exact architecture filter excluded {family_only} family-only "
            f"page(s) and {unknown} architecture-unknown page(s); use "
            "--architecture blackwell or --architecture unknown for those broader sets.",
            file=sys.stderr,
        )

    # Match keywords without scoring. Flatten multi-word quoted args into tokens so
    # `query.py "gate-up GEMM"` behaves like `query.py gate-up GEMM`.
    keywords = []
    for q in args.query:
        for tok in re.split(r"\s+", q.strip()):
            if tok:
                keywords.append(tok)
    candidates = select_candidates(pages, keywords, match_mode=args.match)
    total = len(candidates)
    pages = candidates[args.offset:args.offset + args.limit]
    summary = f"{len(pages)} candidate(s) shown; {total} total matches; path order, not relevance order."
    if pages:
        summary += f" Showing {args.offset + 1}–{args.offset + len(pages)}."
    if args.offset + len(pages) < total:
        summary += f" More candidates: repeat with the same query/filters and --offset {args.offset + len(pages)}."
    if total:
        type_counts = Counter(detect_page_type(p["fm"], p["path"]) for p in candidates)
        summary += "\nTypes across all matches: " + ", ".join(
            f"{ptype}={type_counts[ptype]}" for ptype in sorted(type_counts)
        ) + ". Use --type to narrow."

    if args.paths_only:
        print(summary, file=sys.stderr)
        for p in pages:
            print(p["path"])
        return

    if not pages:
        if total:
            print(f"No candidates at offset {args.offset}; {total} total matches. Use a smaller --offset.")
        else:
            print("No matching pages.")
        return

    print(f"# {summary}")
    print()
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    main()
