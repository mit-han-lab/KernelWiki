#!/usr/bin/env python3
"""Unified query tool for the Blackwell kernel wiki.

Run an explicitly chosen strategy against reusable keyword/alias/filter primitives.
The calling LLM assesses returned candidates; this script makes no model calls.
Master's scoring is one selectable strategy, with no implicit candidate-count cap.

Usage:
    query.py "gate-up GEMM" --strategy master --run-dir ./tmp/search --limit 20
    query.py --strategy ./tmp/strategy.py --context ./tmp/context.json --run-dir ./tmp/search

Relative paths are resolved from the caller's working directory, not the wiki.

Returns matching pages with titles, paths, metadata and bounded match excerpts.
"""

import argparse
import contextlib
import copy
import io
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402
from pr_policy import architecture_matches_filter  # noqa: E402
from _retrieval_log import RunLog, digest  # noqa: E402


_ALIAS_CACHE = None
FILTER_NAMES = ("type", "tag", "repo", "language", "architecture", "symptom",
                "confidence", "has_code")
BUILTIN_STRATEGIES = {
    name: Path(__file__).resolve().parent / "strategies" / f"{name}.py"
    for name in ("master",)
}


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


def score_keyword_match(fm, body, keywords):
    """Master's score, used only when a strategy explicitly selects it."""
    title = str(fm.get("title", "")).lower()
    metadata = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                          "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    ).lower()
    body = body.lower()
    score = 0
    for keyword in keywords:
        variant_scores = []
        for variant in expand_keyword(keyword):
            term = variant.lower()
            variant_scores.append(
                10 * (term in title) + 5 * (term in metadata) + min(body.count(term), 3)
            )
        score += max(variant_scores)
    return score


def select_candidates(pages, keywords):
    """Select OR keyword matches without assigning relevance scores or priorities."""
    candidates = []
    for page in pages:
        matches = keyword_matches(page["fm"], page["body"], keywords)
        if not keywords or matches:
            candidates.append(dict(page, _matches=matches))
    return sorted(candidates, key=lambda page: page["path"])


def query_keywords(text):
    return text.split()


class StrategyIndex:
    """Untruncated retrieval primitives; a strategy owns selection and ordering."""

    def __init__(self, pages):
        self._pages = sorted(pages, key=lambda page: page["path"])
        self._by_id = {p["fm"].get("id", p["path"]): p for p in self._pages}
        self._by_path = {p["path"]: p for p in self._pages}
        self.matches = {}
        self.searches = []
        self.reads = []

    def _page(self, identifier):
        if not isinstance(identifier, str):
            raise ValueError("strategy page references must be IDs or relative paths")
        page = self._by_id.get(identifier) or self._by_path.get(identifier)
        if page is None:
            raise ValueError(f"unknown page or excluded by CLI filters: {identifier}")
        return page

    def search(self, query="", **filters):
        """Return ALL OR/alias matches in path order, without a relevance cutoff."""
        if not isinstance(query, str):
            raise ValueError("search query must be a string")
        if set(filters) - set(FILTER_NAMES):
            raise ValueError(f"unknown filters: {sorted(set(filters) - set(FILTER_NAMES))}")
        for key, value in filters.items():
            expected = bool if key == "has_code" else str
            if value is not None and not isinstance(value, expected):
                raise ValueError(f"filter {key} must be {expected.__name__}")
        args = SimpleNamespace(**{name: filters.get(name) for name in FILTER_NAMES})
        pages = select_candidates(filter_pages(self._pages, args), query_keywords(query))
        ids = []
        for page in pages:
            identifier = page["fm"].get("id", page["path"])
            ids.append(identifier)
            matches = self.matches.setdefault(identifier, {})
            for field, terms in page["_matches"].items():
                matches[field] = list(dict.fromkeys(matches.get(field, []) + terms))
        self.searches.append({"query": query, "filters": filters, "rule": "keyword",
                              "matched_ids": ids.copy(), "match_count": len(ids)})
        return ids

    def master(self, query="", **filters):
        """All matches in master's score/path order, without its default limit."""
        ids = self.search(query, **filters)
        if query_keywords(query):
            ids.sort(key=lambda identifier: (
                -self.keyword_score(identifier, query), self._page(identifier)["path"],
            ))
        self.searches[-1].update(rule="master", ordered_ids=ids.copy())
        return ids

    def metadata(self, identifier):
        """Copy frontmatter; edits to the copy cannot change the corpus."""
        return copy.deepcopy(self._page(identifier)["fm"])

    def read(self, identifier):
        """Read body for strategy logic, logged separately from LLM page reads."""
        page = self._page(identifier)
        self.reads.append({"id": page["fm"].get("id", page["path"]),
                           "path": page["path"], "body_sha256": digest(page["body"]),
                           "body_chars": len(page["body"])})
        return page["body"]

    def keyword_score(self, identifier, query):
        """Helper for policies that explicitly select master's heuristic."""
        page = self._page(identifier)
        return score_keyword_match(page["fm"], page["body"], query_keywords(query))

    def candidates(self, identifiers):
        if not isinstance(identifiers, (list, tuple)):
            raise ValueError("retrieve(index, context) must return a list of page IDs or paths")
        seen, pages = set(), []
        for identifier in identifiers:
            page = self._page(identifier)
            canonical = page["fm"].get("id", page["path"])
            if canonical in seen:
                continue
            seen.add(canonical)
            pages.append(dict(page, _matches=self.matches.get(canonical, {})))
        return pages


class StrategyOutput(io.TextIOBase):
    """Keep accidental strategy prints out of the candidate output budget."""
    def __init__(self):
        self.text = ""
        self.total_chars = 0

    def write(self, text):
        self.total_chars += len(text)
        self.text += text[:max(0, 4096 - len(self.text))]
        return len(text)


def execute_strategy(source, path, index, context):
    # Standard-library helpers such as dataclasses resolve annotations through
    # sys.modules. Give generated code normal module semantics during execution.
    module_name = "kernelwiki_strategy"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    missing = object()
    previous = sys.modules.get(module_name, missing)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        retrieve = getattr(module, "retrieve", None)
        if not callable(retrieve):
            raise ValueError("strategy must define retrieve(index, context)")
        return index.candidates(retrieve(index, copy.deepcopy(context)))
    finally:
        if previous is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def resolve_strategy(selection):
    """Named policies and generated files use the same execution/logging path."""
    if selection in BUILTIN_STRATEGIES:
        return BUILTIN_STRATEGIES[selection]
    path = Path(selection).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"strategy not found: {selection}; choose master or a Python file"
        )
    return path


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
    excerpt_label = "first body match"
    if not excerpt:
        excerpt = " ".join(page["body"].split())[:240]
        excerpt_label = "body preview"

    if compact:
        result = f"  [{ptype}] {pid}: {title}  ({path})"
        if fm.get("architectures"):
            result += f" | architectures: {', '.join(fm['architectures'])}"
        if matched:
            result += f" | matches: {matched}"
        if excerpt:
            result += f" | {excerpt_label}: {excerpt}"
        return result

    # Rich: include key metadata
    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    if matched:
        lines.append(f"- **matches**: {matched}")
    if excerpt:
        lines.append(f"- **{excerpt_label} (excerpt)**: {excerpt}")
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


def render_candidates(pages, *, limit, max_chars, compact, paths_only,
                      order, query_id=None):
    """Emit a prefix of strategy order; disclose every omission in the receipt."""
    def header(count):
        text = (f"{count} candidate(s); {len(pages)} proposed; "
                f"{len(pages) - count} omitted by output limits. Order: {order}.\n")
        if query_id:
            text += f"query_id={query_id}\n"
        text += ("Assess relevance for the current task, then read selected pages. "
                 "If evidence is missing, revise the strategy or output budget.\n")
        return text

    chosen, cards = [], []
    considered = pages if limit is None else pages[:limit]
    for page in considered:
        card = page["path"] + "\n" if paths_only else format_result(page, compact=compact) + "\n\n"
        if len(header(len(chosen) + 1)) + sum(map(len, cards)) + len(card) > max_chars:
            break
        chosen.append(page)
        cards.append(card)
    summary = header(len(chosen))
    if len(summary) > max_chars:
        raise ValueError("--max-chars is too small for the result summary")
    return summary, "".join(cards), chosen


def main():
    parser = argparse.ArgumentParser(description="Query the Blackwell kernel wiki")
    parser.add_argument("query", nargs="*", help="Whitespace-separated keywords for OR candidate retrieval")
    parser.add_argument("--type", help="Filter by page type (kernel, technique, hardware, pattern, language, migration, pr, blog, doc, contest)")
    parser.add_argument("--tag", help="Filter by tag (must appear in tags/techniques/hardware_features/kernel_types/languages)")
    parser.add_argument("--repo", help="Filter by source repo (partial match, e.g. 'cutlass')")
    parser.add_argument("--language", help="Filter by language/DSL (cute-dsl, cuda-cpp, ptx, triton, etc.)")
    parser.add_argument("--architecture", help="Filter by exact architecture, blackwell family hierarchy, or unknown")
    parser.add_argument("--symptom", help="Filter by pattern symptom (memory-bound, register-pressure, etc.)")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--has-code", action="store_true", help="Only return pages whose artifact_dir contains at least one source file")
    parser.add_argument("--strategy", required=True, metavar="master|PATH",
                        help="Explicit policy: master's scoring or a Python file authored by the calling LLM")
    parser.add_argument("--context", type=Path, help="JSON object with task context for the strategy")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Writable experiment directory, e.g. ./tmp/kernelwiki-search in the optimization workspace")
    parser.add_argument("--limit", type=int,
                        help="Optional maximum emitted candidates; no default count cap for any strategy")
    parser.add_argument("--max-chars", type=int, default=16384,
                        help="Maximum candidate output characters including summary (default 16384; not a token count)")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths, one per line")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_chars < 512:
        parser.error("--max-chars must be at least 512")
    context = {}
    try:
        if args.context:
            context = json.loads(args.context.read_text(encoding="utf-8"))
            if not isinstance(context, dict):
                raise ValueError("--context must contain a JSON object")
        if args.query:
            context["query"] = " ".join(args.query)
        else:
            context.setdefault("query", "")
        context["budget"] = {"max_chars": args.max_chars, "limit": args.limit}
        strategy_path = resolve_strategy(args.strategy)
        source = strategy_path.read_bytes()
        log = RunLog(args.run_dir)
        query_id, _ = log.start_query(
            WIKI_ROOT, context, vars(args), source, strategy_path,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

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

    index = StrategyIndex(pages)
    diagnostics = StrategyOutput()
    try:
        with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
            proposed = execute_strategy(source, strategy_path, index, context)
        order = f"strategy {args.strategy}" if args.strategy in BUILTIN_STRATEGIES else "custom strategy"
        summary, output, emitted = render_candidates(
            proposed, limit=args.limit, max_chars=args.max_chars, compact=args.compact,
            paths_only=args.paths_only, order=order, query_id=query_id,
        )
    except (Exception, SystemExit) as error:
        if log:
            log.finish_query(query_id, {"status": "error", "error": str(error),
                             "error_type": type(error).__name__, "searches": index.searches,
                             "strategy_reads": index.reads}, "", diagnostics.text)
        parser.error(f"retrieval failed: {error}")

    if log:
        corpus = [{"path": p["path"], "fm": p["fm"], "body": p["body"]}
                  for p in sorted(all_pages, key=lambda p: p["path"])]
        emitted_ids = [p["fm"].get("id", p["path"]) for p in emitted]
        proposed_ids = [p["fm"].get("id", p["path"]) for p in proposed]
        matched_ids = set(identifier for search in index.searches for identifier in search["matched_ids"])
        log.finish_query(query_id, {
            "status": "ok", "mode": "strategy", "strategy": args.strategy,
            "corpus_pages": len(all_pages),
            "corpus_sha256": digest(json.dumps(corpus, sort_keys=True, default=str)),
            "searches": index.searches, "strategy_reads": index.reads,
            "proposed_ids": proposed_ids, "emitted_ids": emitted_ids,
            "omitted_by_output_limits": proposed_ids[len(emitted):],
            "matched_but_not_proposed": sorted(matched_ids - set(proposed_ids)),
            "budget": context["budget"], "output_chars": len(summary) + len(output),
            "strategy_diagnostic_chars": diagnostics.total_chars,
            "candidate_matches": {identifier: index.matches.get(identifier, {}) for identifier in emitted_ids},
        }, summary + output, diagnostics.text)
    if args.paths_only:
        print(summary, end="", file=sys.stderr)
        print(output, end="")
    else:
        print(summary + output, end="")


if __name__ == "__main__":
    main()
