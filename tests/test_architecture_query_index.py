import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import query  # noqa: E402


def load_hyphenated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


indices = load_hyphenated("generate_indices", SCRIPTS / "generate-indices.py")


def page(path, architectures, disposition=None):
    frontmatter = {
        "id": path,
        "title": path,
        "architectures": architectures,
    }
    if disposition is not None:
        frontmatter["architecture_disposition"] = disposition
    return {
        "path": path,
        "body": "",
        "fm": frontmatter,
    }


def args(architecture):
    return SimpleNamespace(
        type=None,
        tag=None,
        repo=None,
        language=None,
        architecture=architecture,
        symptom=None,
        confidence=None,
        has_code=False,
    )


class QueryIndexConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.pages = [
            page("exact-sm100", ["sm100"], "exact"),
            page("exact-sm100a", ["sm100a"], "exact"),
            page("exact-sm120", ["sm120"], "exact"),
            page("family", ["blackwell"], "family"),
            page("hand-authored-family", ["blackwell"]),
            page("mixed", ["sm100", "hopper"], "mixed"),
            page("hopper", ["sm90"], "exact"),
            page("unknown", [], "unknown"),
            page("non-pr-empty", []),
        ]

    def paths(self, architecture):
        return {p["path"] for p in query.filter_pages(self.pages, args(architecture))}

    def index_pages(self):
        rows = []
        for value in self.pages:
            fm = dict(value["fm"])
            fm["_path"] = value["path"]
            rows.append(fm)
        return rows

    def test_exact_query_has_no_family_false_positive(self):
        self.assertEqual({"exact-sm100", "mixed"}, self.paths("sm100"))

    def test_blackwell_query_is_family_hierarchy(self):
        self.assertEqual(
            {"exact-sm100", "exact-sm100a", "exact-sm120", "family", "hand-authored-family", "mixed"},
            self.paths("blackwell"),
        )

    def test_unknown_query_exposes_every_validated_unknown(self):
        self.assertEqual({"unknown"}, self.paths("unknown"))

    def test_exact_query_discloses_excluded_family_and_unknown_counts(self):
        self.assertEqual(
            (1, 1), query.architecture_filter_exclusion_counts(self.pages, "sm100")
        )
        self.assertIsNone(
            query.architecture_filter_exclusion_counts(self.pages, "blackwell")
        )

    def test_index_membership_equals_declared_metadata_sets(self):
        exact, family, unknown = indices.architecture_index_sets(self.index_pages())
        self.assertEqual({"exact-sm100", "mixed"}, {p["_path"] for p in exact["sm100"]})
        self.assertEqual(
            {"family", "hand-authored-family"},
            {p["_path"] for p in family["blackwell"]},
        )
        self.assertEqual({"unknown"}, {p["_path"] for p in unknown})
        self.assertIn("mixed", {p["_path"] for p in family["hopper"]})
        rendered = indices.generate_by_architecture(self.index_pages())
        self.assertIn("## Blackwell family-only", rendered)
        self.assertIn("no supported exact SM in that family", rendered)
        self.assertIn("## Architecture unknown", rendered)
        self.assertIn("validated source-PR unknowns", rendered)
        self.assertIn("Source-PR pages whose validated evidence review", rendered)
        self.assertIn("### `sm100`", rendered)

    def test_documented_architecture_aliases_match_family_and_product_semantics(self):
        query._ALIAS_CACHE = None
        aliases = query.load_alias_expansions()
        self.assertEqual("blackwell", aliases["blackwell"])
        for product in ("b200", "gb200"):
            self.assertEqual("sm100", aliases[product])
        for product in ("b300", "gb300", "sm103"):
            self.assertEqual("sm103", aliases[product])


if __name__ == "__main__":
    unittest.main()
