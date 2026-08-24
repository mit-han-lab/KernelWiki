#!/usr/bin/env python3
"""Generate query index pages from YAML frontmatter in sources/ and wiki/."""

import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

from pr_policy import ARCHITECTURE_FAMILY_PREFIXES

REPO_ROOT = Path(__file__).parent.parent
SOURCES_DIR = REPO_ROOT / "sources"
WIKI_DIR = REPO_ROOT / "wiki"
QUERIES_DIR = REPO_ROOT / "queries"


def extract_frontmatter(filepath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def qlink(path):
    """Make a link relative from the queries/ directory."""
    return f"../{path}"


def load_valid_languages():
    """Load valid language tags from data/tags.yaml."""
    tags_path = REPO_ROOT / "data" / "tags.yaml"
    if tags_path.exists():
        with open(tags_path, encoding='utf-8') as f:
            return set(yaml.safe_load(f).get("languages", []))
    return set()


def collect_all_pages():
    pages = []
    errors = []
    for search_dir in [SOURCES_DIR, WIKI_DIR]:
        if not search_dir.exists():
            continue
        for md_file in sorted(search_dir.rglob("*.md")):
            fm = extract_frontmatter(md_file)
            rel = str(md_file.relative_to(REPO_ROOT))
            if fm is None:
                errors.append(f"{rel}: missing or unparseable frontmatter")
            elif not isinstance(fm, dict):
                errors.append(f"{rel}: frontmatter is {type(fm).__name__}, not a mapping")
            else:
                fm["_path"] = md_file.relative_to(REPO_ROOT).as_posix()
                fm["_dir"] = search_dir.relative_to(REPO_ROOT).as_posix()
                pages.append(fm)
    if errors:
        print(f"ERROR: {len(errors)} files have missing/unparseable frontmatter:")
        for e in errors:
            print(f"  {e}")
        print("Run python3 scripts/validate.py first to fix these files.")
        sys.exit(1)
    return pages


def generate_by_problem(pages):
    # Build id→page map for resolving technique IDs to links
    id_map = {p["id"]: p for p in pages if "id" in p}

    lines = [
        "# Query: By Problem / Symptom",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Symptom | Pattern Page | Candidate Techniques | Sources |",
        "|---------|-------------|---------------------|---------|",
    ]
    for p in sorted(
        (p for p in pages if p.get("type") == "pattern"),
        key=lambda x: x.get("title", ""),
    ):
        symptoms = ", ".join(p.get("symptoms", []))
        title = p.get("title", "Untitled")
        path = qlink(p["_path"])
        # Resolve technique IDs to clickable links
        tech_links = []
        for tid in p.get("candidate_techniques", []):
            if tid in id_map:
                t = id_map[tid]
                tech_links.append(f"[{t.get('title', tid)}]({qlink(t['_path'])})")
            else:
                tech_links.append(tid)
        techniques = ", ".join(tech_links)
        src_count = len(p.get("sources", []))
        lines.append(f"| {symptoms} | [{title}]({path}) | {techniques} | {src_count} sources |")
    return "\n".join(lines) + "\n"


def generate_by_technique(pages):
    lines = [
        "# Query: By Technique",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Technique | Tags | Architectures | Confidence | Reproducibility | Sources |",
        "|-----------|------|--------------|------------|-----------------|---------|",
    ]
    for t in sorted(
        (p for p in pages if p.get("type") == "technique"),
        key=lambda x: x.get("title", ""),
    ):
        title = t.get("title", "Untitled")
        path = qlink(t["_path"])
        tags = ", ".join(t.get("tags", [])[:4])
        archs = ", ".join(t.get("architectures", []))
        conf = t.get("confidence", "")
        repro = t.get("reproducibility", "")
        src_count = len(t.get("sources", []))
        lines.append(f"| [{title}]({path}) | {tags} | {archs} | {conf} | {repro} | {src_count} |")
    return "\n".join(lines) + "\n"


def generate_by_hardware_feature(pages):
    lines = [
        "# Query: By Hardware Feature",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]
    tags_path = REPO_ROOT / "data" / "tags.yaml"
    hw_tag_set = set()
    if tags_path.exists():
        with open(tags_path, encoding='utf-8') as f:
            hw_tag_set = set(yaml.safe_load(f).get("hardware_features", []))

    # Merge explicit hardware_features AND supplemental hw tags (deduplicated)
    feature_pages = defaultdict(list)
    for p in pages:
        indexed = set()
        for feat in p.get("hardware_features", []):
            feature_pages[feat].append(p)
            indexed.add(feat)
        for tag in p.get("tags", []):
            if tag in hw_tag_set and tag not in indexed:
                feature_pages[tag].append(p)
                indexed.add(tag)

    # Add dedicated hardware pages under their canonical tags (from tags field)
    hw_pages = [p for p in pages if p.get("type") == "hardware"]
    for hp in hw_pages:
        for tag in hp.get("tags", []):
            if tag in hw_tag_set and hp not in feature_pages.get(tag, []):
                feature_pages[tag].append(hp)

    lines.append("| Feature | Related Pages |")
    lines.append("|---------|--------------|")
    for feat in sorted(feature_pages.keys()):
        page_links = []
        seen = set()
        for p in feature_pages[feat]:
            pid = p.get("id", p.get("_path", ""))
            if pid not in seen:
                seen.add(pid)
                title = p.get("title", pid)
                path = qlink(p["_path"])
                page_links.append(f"[{title}]({path})")
        lines.append(f"| `{feat}` | {', '.join(page_links)} |")
    return "\n".join(lines) + "\n"


def generate_by_repo(pages):
    lines = [
        "# Query: By Repository",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]
    repos = defaultdict(list)
    for p in pages:
        if "repo" in p and "pr" in p:
            repos[p["repo"]].append(p)

    for repo in sorted(repos.keys()):
        prs = sorted(repos[repo], key=lambda x: str(x.get("date", "")), reverse=True)
        # Real HTML anchor so index.md fragment links work in any renderer
        slug = repo.lower().replace("/", "")
        lines.append(f'<a id="{slug}"></a>')
        lines.append(f"## {repo}")
        lines.append(f"{len(prs)} PRs")
        lines.append("")
        lines.append("| PR | Title | Date | Techniques | Tags |")
        lines.append("|-----|-------|------|------------|------|")
        for pr in prs:
            pr_num = pr.get("pr", "?")
            title = pr.get("title", "Untitled")
            date = pr.get("date", "")
            path = qlink(pr["_path"])
            techniques = ", ".join(list(dict.fromkeys(pr.get("techniques", [])))[:3])
            tags = ", ".join(list(dict.fromkeys(pr.get("tags", [])))[:3])
            lines.append(f"| [#{pr_num}]({path}) | {title} | {date} | {techniques} | {tags} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def generate_by_kernel_type(pages):
    lines = [
        "# Query: By Kernel Type",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Kernel Type | Pages |",
        "|-------------|-------|",
    ]
    # Load valid kernel type tags to filter tags field
    tags_path = REPO_ROOT / "data" / "tags.yaml"
    kt_tag_set = set()
    if tags_path.exists():
        with open(tags_path, encoding='utf-8') as f:
            kt_tag_set = set(yaml.safe_load(f).get("kernel_types", []))

    # Explicit kernel_types is authoritative; tags only when field is absent/empty
    kernel_bearing_types = {"kernel", None}
    type_pages = defaultdict(list)
    for p in pages:
        explicit = p.get("kernel_types", [])
        if explicit:
            for kt in explicit:
                type_pages[kt].append(p)
        else:
            page_type = p.get("type")
            if page_type in kernel_bearing_types:
                for tag in p.get("tags", []):
                    if tag in kt_tag_set:
                        type_pages[tag].append(p)

    for kt in sorted(type_pages.keys()):
        page_links = []
        seen = set()
        for p in type_pages[kt]:
            pid = p.get("id", p.get("_path", ""))
            if pid not in seen:
                seen.add(pid)
                title = p.get("title", pid)
                path = qlink(p["_path"])
                page_links.append(f"[{title}]({path})")
        lines.append(f"| `{kt}` | {', '.join(page_links)} |")
    return "\n".join(lines) + "\n"


def generate_by_language(pages):
    lines = [
        "# Query: By Language / DSL",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
    ]
    valid_langs = load_valid_languages()

    # Map: canonical language tag -> {guide: page, consumers: [pages]}
    lang_data = {}
    for lang in sorted(valid_langs):
        lang_data[lang] = {"guide": None, "consumers": []}

    # Find dedicated language guide pages
    # Match by: first tag that's a valid language, OR id-derived tag as fallback
    for p in pages:
        if p.get("type") == "language":
            # Primary language: first tag that's in the valid languages set
            primary_lang = None
            for tag in p.get("tags", []):
                if tag in valid_langs:
                    primary_lang = tag
                    break
            # Fallback: derive from id (lang-cuda-cpp -> cuda-cpp)
            if not primary_lang:
                lang_id = p.get("id", "")
                if lang_id.startswith("lang-"):
                    candidate = lang_id[5:]  # strip "lang-"
                    if candidate in valid_langs:
                        primary_lang = candidate
            if primary_lang and primary_lang in lang_data:
                lang_data[primary_lang]["guide"] = p

    # Explicit languages is authoritative; tags only when field is absent/empty
    for p in pages:
        if p.get("type") != "language":
            explicit = p.get("languages", [])
            if explicit:
                for lang in explicit:
                    if lang in lang_data:
                        lang_data[lang]["consumers"].append(p)
            else:
                for tag in p.get("tags", []):
                    if tag in valid_langs:
                        lang_data[tag]["consumers"].append(p)

    lines.append("| Language | Guide | Related Pages |")
    lines.append("|----------|-------|--------------|")

    for lang in sorted(lang_data.keys()):
        data = lang_data[lang]
        if not data["guide"] and not data["consumers"]:
            continue
        guide_link = ""
        if data["guide"]:
            g = data["guide"]
            guide_link = f"[{g.get('title', lang)}]({qlink(g['_path'])})"
        related = []
        seen = set()
        for p in data["consumers"]:
            pid = p.get("id", p.get("_path", ""))
            if pid not in seen:
                seen.add(pid)
                related.append(f"[{p.get('title', pid)}]({qlink(p['_path'])})")
        lines.append(f"| `{lang}` | {guide_link} | {', '.join(related)} |")

    return "\n".join(lines) + "\n"


def architecture_index_sets(pages):
    """Return the canonical exact, family-only, and unknown index memberships."""
    exact = defaultdict(list)
    family_only = defaultdict(list)
    unknown = []
    for page in pages:
        architectures = page.get("architectures") or []
        disposition = page.get("architecture_disposition")
        family_architectures = [
            architecture for architecture in architectures
            if architecture in ARCHITECTURE_FAMILY_PREFIXES
        ]
        # Hand-authored wiki/source pages do not carry the generated-PR
        # disposition field.  Their declared family value must still be in the
        # same index lane returned by the architecture query.
        if disposition == "family" or family_architectures:
            for architecture in family_architectures:
                family_only[architecture].append(page)
        if not architectures and disposition == "unknown":
            unknown.append(page)
        for architecture in architectures:
            if architecture not in ARCHITECTURE_FAMILY_PREFIXES:
                exact[architecture].append(page)
    return exact, family_only, unknown


def _architecture_page_table(lines, pages):
    lines.extend(["| Page | Path |", "|------|------|"])
    for page in sorted(pages, key=lambda value: value.get("_path", "")):
        title = page.get("title", page.get("id", "Untitled"))
        path = page["_path"]
        lines.append(f"| [{title}]({qlink(path)}) | `{path}` |")
    if not pages:
        lines.append("| _None_ | |")
    lines.append("")


def generate_by_architecture(pages):
    """Generate an architecture index with visible family-only and unknown lanes."""
    exact, family_only, unknown = architecture_index_sets(pages)
    lines = [
        "# Query: By Architecture",
        "",
        "> Auto-generated. Do not edit manually. Exact sections contain only pages "
        "carrying that exact value; family-only pages and validated source-PR unknowns "
        "are intentionally separate. Non-PR sources with an empty architecture list have "
        "no validated unknown disposition and are outside the unknown lane.",
        "",
    ]
    for family in ARCHITECTURE_FAMILY_PREFIXES:
        lines.extend([
            f"## {family.title()} family-only",
            "",
            f"Pages with explicit generic {family.title()} evidence but no supported exact SM in that family.",
            "",
        ])
        _architecture_page_table(lines, family_only.get(family, []))
    lines.extend([
        "## Architecture unknown",
        "",
        "Source-PR pages whose validated evidence review found no family-level or exact architecture signal.",
        "",
    ])
    _architecture_page_table(lines, unknown)
    lines.extend(["## Exact architectures", ""])
    for architecture in sorted(exact):
        lines.extend([f"### `{architecture}`", ""])
        _architecture_page_table(lines, exact[architecture])
    return "\n".join(lines) + "\n"


def main():
    QUERIES_DIR.mkdir(exist_ok=True)
    pages = collect_all_pages()
    print(f"Collected {len(pages)} pages from sources/ and wiki/")

    generators = {
        "by-architecture.md": generate_by_architecture,
        "by-problem.md": generate_by_problem,
        "by-technique.md": generate_by_technique,
        "by-hardware-feature.md": generate_by_hardware_feature,
        "by-repo.md": generate_by_repo,
        "by-kernel-type.md": generate_by_kernel_type,
        "by-language.md": generate_by_language,
    }

    for filename, gen_func in generators.items():
        content = gen_func(pages)
        outpath = QUERIES_DIR / filename
        outpath.write_text(content, encoding="utf-8")
        print(f"  Generated {outpath.relative_to(REPO_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
