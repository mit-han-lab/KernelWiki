#!/usr/bin/env python3
"""Extract fenced code blocks from sources/blogs/*.md into standalone files
under artifacts/blogs/<slug>/code/, with a MANIFEST.yaml that maps each
extracted file to its originating heading path and SHA-256 (AC-4).

Modes:
  extract   — write fresh files (default; overwrites existing if drift detected)
  --check <slug>   — idempotent check: re-parse markdown + verify checksums
  --check-all      — run --check across every blog slug

Extension mapping (fence language -> suffix):
  cuda, cu, c++, cpp, cxx -> .cu
  cuh                     -> .cuh
  ptx                     -> .ptx
  python, py, triton      -> .py (triton kernels are python code)
  cute                    -> .py   (CuTe DSL is python)
  yaml                    -> .yaml
  (unknown)               -> .txt

Output format: every extracted file carries a header comment documenting
provenance-in-source-blog; the PROVENANCE.yaml bundle metadata lives at
artifacts/blogs/<slug>/PROVENANCE.yaml (asset_mode: extracted).
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOGS_DIR = REPO_ROOT / "sources" / "blogs"
OUT_DIR = REPO_ROOT / "artifacts" / "blogs"

EXT_MAP = {
    "cuda": "cu", "cu": "cu", "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "c": "cpp",
    "cuh": "cuh",
    "ptx": "ptx", "asm": "ptx",
    "python": "py", "py": "py", "triton": "py", "cute": "py",
    "yaml": "yaml",
    "bash": "sh", "shell": "sh", "sh": "sh",
    "json": "json",
}

SLUGIFY_RE = re.compile(r"[^a-z0-9]+")


def slugify(s):
    s = s.lower().strip()
    s = SLUGIFY_RE.sub("-", s)
    return s.strip("-")[:60] or "section"


def is_extractable_block(lang, body):
    """Return True iff a fenced block should be extracted.

    - Empty body: reject.
    - Known language (in EXT_MAP): accept unconditionally — the author
      declared it as code.
    - Unlabeled / unknown language: apply a cheap code-likeness heuristic
      so prose fences (bullet lists, paragraphs, attempt write-ups) are
      NOT misclassified as code. Blogs like `amandeep-nvfp4-attempts`
      use unlabeled fences for formatted notes; without this check the
      extractor would ship `01-attempts-1-3-getting-the-basics-right.txt`
      as a `.txt` "code" file and pollute `query.py --has-code`
      (Codex R26 P2).
    """
    if not body.strip():
        return False
    if lang and lang in EXT_MAP:
        return True
    return _looks_like_code_fence(body)


_BULLET_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S")
_COMMENT_LINE_RE = re.compile(r"^\s*(?://|#|/\*|\*/|\*|<!--)")
# R35: real-code structure markers. Formulas (`x = f(y)`), configs
# (`DP=8, EP=8, TP=1`), shell commands (`vllm serve ...`), and output
# logs (`Initial: grid=128`) all lack these. Source code carries at
# least one of:
#   - `;` or `{` / `}` (C-family statement terminators or blocks)
#   - a language keyword: control flow (for/while/if/else/return),
#     definition (def/class/struct/template/namespace/typedef/using),
#     or GPU / low-level markers (__global__/__device__/__host__,
#     extern/static/inline/void).
_CODE_STRUCTURE_RE = re.compile(
    r"[;{}]"
    r"|\b(?:def|class|struct|template|namespace|typedef|using|"
    r"for|while|if|else|return|extern|static|inline|void|"
    r"__global__|__device__|__host__|__forceinline__|__shared__|"
    r"asm\s+volatile)\b"
)


def _looks_like_code_fence(body):
    """Heuristic: an unlabeled fence is treated as code only when it
    (a) is not dominantly bullet / numbered prose AND (b) carries a
    positive code signal (non-comment line) AND (c) carries a real
    code-structure signal.

    Bullet check (R26): if >50% of non-blank lines are bullet or
    numbered list items, reject.
    Non-comment check (R32): at least one non-blank line that does NOT
    start with a comment marker (`//`, `#`, `/*`, `*/`, `*`, `<!--`).
    Code-structure check (R35): at least one non-comment line must
    contain a statement terminator / block brace (`;`, `{`, `}`) or a
    language keyword (for / while / if / return / def / class / etc.).
    This rejects formulas (`x_hat = s * deq(q)`), configs
    (`DP=8, EP=8, TP=1`), shell commands (`vllm serve ...`), and
    output logs (`Initial: grid=128`) which would otherwise pass the
    first two checks via a single non-comment line without being real
    source code.

    Known-language fences (`lang in EXT_MAP`) still bypass the
    heuristic entirely — those are explicit curator declarations.
    """
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    bullet_lines = sum(1 for ln in lines if _BULLET_RE.match(ln))
    bullet_ratio = bullet_lines / len(lines)
    if bullet_ratio > 0.5:
        return False
    non_comment = [ln for ln in lines if not _COMMENT_LINE_RE.match(ln)]
    if not non_comment:
        return False
    if not any(_CODE_STRUCTURE_RE.search(ln) for ln in non_comment):
        return False
    return True


_EXTRACT_SKIP_RE = re.compile(r"<!--\s*extract-skip\b.*?-->", re.IGNORECASE)


def parse_markdown(md_path):
    """Yield (heading_path, fence_lang, fence_body) for every fenced code block.

    heading_path is "## A > ### B" — the nearest ancestor chain of markdown
    headings above the fence.

    R30: an HTML comment matching `<!-- extract-skip ... -->` on the
    immediately-preceding non-blank line suppresses the fence. Curators
    use this marker when a code block is synthesized pseudo-code,
    placeholder sketches, or otherwise not verbatim upstream. Without
    the marker, such blocks would be published under `artifacts/blogs/**`
    with `mode: extracted`, falsely asserting a provenance guarantee
    the content doesn't actually meet.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return

    # Strip frontmatter
    m = re.match(r"^---\s*\r?\n.*?\r?\n---\s*\r?\n(.*)", text, re.DOTALL)
    body = m.group(1) if m else text

    lines = body.splitlines()
    heading_stack = []  # list of (level, text)
    i = 0
    while i < len(lines):
        line = lines[i]
        # heading detection
        mh = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if mh:
            level = len(mh.group(1))
            htext = mh.group(2).rstrip("#").strip()
            # Pop stack until we find a strictly-less-than level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, htext))
            i += 1
            continue
        # fence detection (```lang or ~~~lang)
        mf = re.match(r"^```(\S*)\s*$", line)
        if mf:
            fence_open_idx = i
            lang = mf.group(1).strip().lower()
            buf = []
            i += 1
            while i < len(lines) and not re.match(r"^```\s*$", lines[i]):
                buf.append(lines[i])
                i += 1
            # closing fence
            if i < len(lines):
                i += 1
            # Check for an extract-skip directive on the immediately
            # preceding non-blank line (blank lines between directive
            # and fence are tolerated).
            k = fence_open_idx - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            if k >= 0 and _EXTRACT_SKIP_RE.search(lines[k]):
                continue
            hp = " > ".join(f"{'#'*lvl} {t}" for lvl, t in heading_stack) if heading_stack else "(root)"
            yield hp, lang, "\n".join(buf) + "\n"
            continue
        i += 1


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _derive_filename(idx, heading_path, lang, seen_names):
    """Compute the NN-<slug>.<ext> filename for a code block. Matches the
    naming rule used during extraction; shared with --check so fresh
    regeneration is byte-identical."""
    leaf = heading_path.split(" > ")[-1] if heading_path != "(root)" else "root"
    leaf = re.sub(r"^#+\s*", "", leaf)
    stub = slugify(leaf)
    ext = EXT_MAP.get(lang, "txt")
    fn = f"{idx:02d}-{stub}.{ext}"
    orig = fn
    k = 2
    while fn in seen_names:
        fn = orig.replace(f".{ext}", f"-{k}.{ext}")
        k += 1
    seen_names.add(fn)
    return fn, ext


def _file_header(slug, heading_path, lang, ext):
    """Build the per-file provenance header injected by the extractor.
    C-family extensions get `//` comments; JSON gets no header (it has
    no comment syntax); everything else gets `#`.

    The header cites the bundle's actual PROVENANCE.yaml path
    (`artifacts/blogs/<slug>/code/PROVENANCE.yaml`); the `code/` suffix
    matters because that is where the asset-bundle root lives and where
    the provenance metadata is written. Without the suffix readers
    following the trail would land on a non-existent file.

    R34: JSON doesn't allow any comment syntax (not `//`, not `#`, not
    `/* */`) so a `#`-prefixed header would make the extracted `.json`
    file unparseable. Emit JSON with no provenance prelude — the
    bundle-level PROVENANCE.yaml still records the block's heading_path
    and origin, so readers following the backlink still get the full
    metadata, just not inlined into the JSON file itself.
    """
    if ext in ("cu", "cuh", "cpp", "ptx", "h", "hpp"):
        return (
            f"// Extracted from sources/blogs/{slug}.md by scripts/extract_blog_code.py\n"
            f"// Heading: {heading_path}\n"
            f"// Original fence language: {lang}\n"
            f"// See artifacts/blogs/{slug}/code/PROVENANCE.yaml for origin + license metadata.\n\n"
        )
    if ext == "json":
        return ""
    return (
        f"# Extracted from sources/blogs/{slug}.md by scripts/extract_blog_code.py\n"
        f"# Heading: {heading_path}\n"
        f"# Original fence language: {lang}\n"
        f"# See artifacts/blogs/{slug}/code/PROVENANCE.yaml for origin + license metadata.\n\n"
    )


def _expected_block_content(slug, heading_path, lang, body, ext):
    """Return the exact bytes the emitter would write for a given block.
    --check uses this to byte-compare against the on-disk extracted file,
    so a same-count same-filename body edit in the source markdown is
    detected."""
    return _file_header(slug, heading_path, lang, ext) + body


def extract_one_blog(blog_md, force=False):
    """Extract code from a single blog. Returns (num_files_written, code_present_bool)."""
    slug = blog_md.stem
    bundle = OUT_DIR / slug
    code_dir = bundle / "code"
    manifest_path = bundle / "MANIFEST.yaml"
    prov_path = bundle / "PROVENANCE.yaml"

    blocks = list(parse_markdown(blog_md))

    # Read blog frontmatter for provenance metadata
    try:
        md_text = blog_md.read_text(encoding="utf-8")
    except OSError:
        return 0, False
    mf = re.match(r"^---\s*\r?\n(.*?)\r?\n---", md_text, re.DOTALL)
    fm = yaml.safe_load(mf.group(1)) if mf else {}
    origin_url = fm.get("url", "") if isinstance(fm, dict) else ""
    author = fm.get("author", "") if isinstance(fm, dict) else ""
    retrieved = fm.get("retrieved_at", "") if isinstance(fm, dict) else ""

    # Filter out trivial blocks (empty body). Unknown / unlabeled
    # fences pass through — _derive_filename's EXT_MAP.get(lang, "txt")
    # handles the extension fallback.
    code_blocks = [(hp, lang, body) for (hp, lang, body) in blocks
                   if is_extractable_block(lang, body)]

    if not code_blocks:
        # Blog has no supported fenced code. If a previous extraction left a
        # code/ subtree on disk, remove it entirely — otherwise validate.py,
        # query.py --has-code, and get_page.py --include-code will continue
        # to treat the blog as if it had extractable code.

        if code_dir.is_dir():
            shutil.rmtree(code_dir)
        bundle.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(yaml.dump({
            "slug": slug,
            "origin_url": origin_url,
            "code_present": False,
            "generated_by": "scripts/extract_blog_code.py",
        }, sort_keys=False), encoding="utf-8")
        return 0, False

    # Fresh extract: clear existing code dir to avoid orphan files
    if code_dir.is_dir():
        for f in code_dir.iterdir():
            if f.is_file():
                f.unlink()

    code_dir.mkdir(parents=True, exist_ok=True)

    manifest_files = []
    files_entries = []  # for PROVENANCE.yaml files[*]
    seen_names = set()
    for idx, (heading_path, lang, body) in enumerate(code_blocks, start=1):
        fn, ext = _derive_filename(idx, heading_path, lang, seen_names)
        content = _expected_block_content(slug, heading_path, lang, body, ext)
        (code_dir / fn).write_text(content, encoding="utf-8")
        sha = sha256_bytes(content.encode("utf-8"))
        # MANIFEST.yaml sits at the parent (bundle/) level, so its paths are
        # bundle-relative ("code/<fn>"). PROVENANCE.yaml lives at code/ (which
        # IS the asset-bundle root), so its local_path entries are just "<fn>".
        manifest_files.append({
            "local_path": f"code/{fn}",
            "heading_path": heading_path,
            "fence_lang": lang,
            "sha256": sha,
        })
        files_entries.append({
            "local_path": fn,
            "role": "extracted-block",
            "mode": "extracted",
            "upstream_path": "inline-in-blog-markdown",
            "heading_path": heading_path,
            "sha256": sha,
        })

    # Write MANIFEST.yaml (bundle-relative)
    manifest_path.write_text(yaml.dump({
        "slug": slug,
        "origin_url": origin_url,
        "code_present": True,
        "total_blocks": len(manifest_files),
        "generated_by": "scripts/extract_blog_code.py",
        "files": manifest_files,
    }, sort_keys=False), encoding="utf-8")

    # Write PROVENANCE.yaml for the code/ bundle root
    prov_code_path = code_dir / "PROVENANCE.yaml"
    prov_code_path.write_text(yaml.dump({
        "origin_url": origin_url,
        "upstream_repo": "blog",
        "upstream_sha": "none",
        "license": "inherits-from-source-blog",
        "retrieved_at": retrieved or "",
        "asset_mode": "extracted",
        "generated_by": "scripts/extract_blog_code.py",
        "size_cap_truncated": False,
        "files": files_entries,
    }, sort_keys=False), encoding="utf-8")

    return len(manifest_files), True


def check_one_blog(slug):
    bundle = OUT_DIR / slug
    manifest_path = bundle / "MANIFEST.yaml"
    code_dir = bundle / "code"
    blog_md = BLOGS_DIR / f"{slug}.md"

    # Pre-scan the source markdown for extractable fenced blocks so we can
    # distinguish "legitimately no extraction needed" (pure-text blog) from
    # "source has code but bundle never generated" (drift that must fail).
    source_has_code = False
    if blog_md.is_file():
        for (_hp, lang, body) in parse_markdown(blog_md):
            if is_extractable_block(lang, body):
                source_has_code = True
                break

    if not manifest_path.is_file():
        if not blog_md.is_file():
            return [f"{slug}: neither source markdown nor extraction bundle exists"]
        if source_has_code:
            return [
                f"{slug}: source blog has fenced code but "
                f"artifacts/blogs/{slug}/ bundle is missing — "
                f"run scripts/extract_blog_code.py to generate it"
            ]
        # Source markdown exists with no extractable code; absence of a
        # bundle is the expected state and is not a drift.
        return []
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [f"{slug}: MANIFEST.yaml parse error: {e}"]

    errors = []
    if not manifest.get("code_present"):
        # Manifest says no code; but if the source markdown has grown new
        # fenced code blocks we must flag that drift instead of returning OK.
        if source_has_code:
            errors.append(
                f"{slug}: manifest code_present=false but source markdown "
                f"has fenced code — re-run scripts/extract_blog_code.py"
            )
        return errors

    if not blog_md.is_file():
        errors.append(f"{slug}: source blog not found at {blog_md.relative_to(REPO_ROOT)}")
        return errors

    fresh_blocks = [(hp, lang, body) for (hp, lang, body) in parse_markdown(blog_md)
                    if is_extractable_block(lang, body)]

    manifest_count = len(manifest.get("files", []))
    if len(fresh_blocks) != manifest_count:
        errors.append(
            f"{slug}: markdown-vs-manifest drift "
            f"(markdown has {len(fresh_blocks)} code blocks, manifest has {manifest_count})"
        )

    # Regenerate expected bytes for each fresh block and byte-compare against
    # the on-disk extracted file. This detects the "same fence count, different
    # body" case that a pure count + manifest-SHA check misses.
    seen_names = set()
    expected_by_fn = {}           # filename -> expected bytes
    expected_sha_by_fn = {}       # filename -> expected sha256
    for idx, (hp, lang, body) in enumerate(fresh_blocks, start=1):
        fn, ext = _derive_filename(idx, hp, lang, seen_names)
        content = _expected_block_content(slug, hp, lang, body, ext)
        expected_by_fn[fn] = content
        expected_sha_by_fn[fn] = sha256_bytes(content.encode("utf-8"))

    # Flag manifest entries whose filename is not present in the freshly
    # derived set, or whose sha256 field disagrees with the fresh extraction.
    for entry in manifest.get("files") or []:
        lp = entry.get("local_path")
        if not lp:
            continue
        # MANIFEST uses bundle-relative paths like "code/<fn>"; strip that prefix
        fn = lp.split("/", 1)[1] if lp.startswith("code/") else lp
        if fn not in expected_by_fn:
            errors.append(
                f"{slug}/{lp}: manifest entry refers to a file no longer produced "
                f"from the source markdown (renamed or dropped block)"
            )
            continue
        declared_sha = entry.get("sha256")
        if declared_sha and declared_sha != expected_sha_by_fn[fn]:
            errors.append(
                f"{slug}/{lp}: manifest sha256 disagrees with fresh extraction "
                f"(markdown body changed; re-run scripts/extract_blog_code.py)"
            )

    # Also flag any fresh filenames that the manifest does not list.
    for fn in expected_by_fn:
        if not any(
            (e.get("local_path", "").split("/", 1)[1] if e.get("local_path", "").startswith("code/") else e.get("local_path")) == fn
            for e in (manifest.get("files") or [])
        ):
            errors.append(
                f"{slug}/code/{fn}: fresh extraction produces this file but the "
                f"manifest does not list it (new block in markdown)"
            )

    # Finally, byte-compare every manifest-listed on-disk file against the
    # freshly-regenerated expected bytes. Old behaviour only checked
    # on-disk-vs-manifest-SHA, which silently passed when the body changed.
    for entry in manifest.get("files") or []:
        lp = entry.get("local_path")
        declared_sha = entry.get("sha256")
        if not lp or not declared_sha:
            continue
        fn = lp.split("/", 1)[1] if lp.startswith("code/") else lp
        fp = bundle / lp
        if not fp.is_file():
            errors.append(f"{slug}/{lp}: file missing")
            continue
        on_disk = fp.read_bytes()
        actual = sha256_bytes(on_disk)
        # On-disk vs manifest (hand-edit detection)
        if actual != declared_sha:
            errors.append(f"{slug}/{lp}: SHA-256 mismatch (hand-edit detected)")
        # Byte-compare on-disk content against freshly-regenerated expected
        # bytes (source-markdown drift detection). A same-count-different-body
        # body change would sail past the manifest-SHA check above; this step
        # catches it.
        if fn in expected_by_fn:
            expected_bytes = expected_by_fn[fn].encode("utf-8")
            if on_disk != expected_bytes:
                errors.append(
                    f"{slug}/{lp}: on-disk bytes differ from fresh extraction "
                    f"of the current markdown (source body changed; "
                    f"re-run scripts/extract_blog_code.py)"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", metavar="SLUG", help="Idempotent check for one blog")
    parser.add_argument("--check-all", action="store_true", help="Check every extracted blog")
    parser.add_argument("--only", metavar="SLUG", help="Extract only one blog (debugging)")
    args = parser.parse_args()

    BLOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.check or args.check_all:
        all_errors = []
        if args.check_all:
            # Iterate sources/blogs/*.md first so that a newly-added blog
            # whose bundle hasn't been generated is detected as drift.
            # Previous revisions iterated artifacts/blogs/ which silently
            # skipped any source blog that lacked a bundle.
            checked = set()
            for blog_md in sorted(BLOGS_DIR.glob("*.md")) if BLOGS_DIR.is_dir() else []:
                slug = blog_md.stem
                checked.add(slug)
                all_errors.extend(check_one_blog(slug))
            # Also surface orphan bundles (bundle present, source markdown
            # deleted) — check_one_blog will report the missing-source error.
            if OUT_DIR.is_dir():
                for bundle in sorted(OUT_DIR.iterdir()):
                    if bundle.is_dir() and bundle.name not in checked:
                        all_errors.extend(check_one_blog(bundle.name))
        elif args.check:
            all_errors.extend(check_one_blog(args.check))
        if all_errors:
            for e in all_errors:
                print(f"  FAIL: {e}", file=sys.stderr)
            sys.exit(1)
        print("All checked blogs match their source markdown.")
        sys.exit(0)

    if not BLOGS_DIR.is_dir():
        print(f"No sources/blogs/ directory; nothing to extract.")
        sys.exit(0)

    total = 0
    with_code = 0
    for blog_md in sorted(BLOGS_DIR.glob("*.md")):
        if args.only and blog_md.stem != args.only:
            continue
        n, had_code = extract_one_blog(blog_md)
        total += 1
        if had_code:
            with_code += 1
            print(f"  {blog_md.stem}: extracted {n} code block(s)")
        else:
            print(f"  {blog_md.stem}: code_present: false (no fenced blocks)")

    print(f"\nProcessed {total} blog(s), {with_code} with extractable code.")


if __name__ == "__main__":
    main()
