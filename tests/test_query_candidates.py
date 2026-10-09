"""Candidate recall, evidence display and pagination without relevance scores."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import query


def page(path, title="", body="", **metadata):
    return {"path": path, "fm": {"id": path, "title": title, **metadata}, "body": body}


class CandidateTests(unittest.TestCase):
    def test_title_metadata_and_repeated_body_hits_do_not_set_priority(self):
        pages = [page("z", "TMA"), page("b", tags=["tma"]), page("a", body="tma " * 100)]
        candidates = query.select_candidates(pages, ["tma"])
        self.assertEqual(["a", "b", "z"], [p["path"] for p in candidates])
        self.assertTrue(all("_score" not in p for p in candidates))
        self.assertEqual({"metadata": ["tma"]}, candidates[1]["_matches"])
        self.assertNotIn("_matches", pages[0])

    def test_aliases_and_or_matching_preserve_candidate_recall(self):
        pages = [page("a", "tcgen05"), page("b", body="UMMA"),
                 page("c", tags=["tmem"]), page("d", "unrelated")]
        candidates = query.select_candidates(pages, ["UMMA", "tmem"])
        self.assertEqual(["a", "b", "c"], [p["path"] for p in candidates])
        self.assertEqual({"title": ["tcgen05"]}, candidates[0]["_matches"])
        self.assertEqual({"body": ["umma"]}, candidates[1]["_matches"])

    def test_filter_only_lists_all_and_unknown_keyword_lists_none(self):
        pages = [page("b"), page("a")]
        for mode in ("any", "all"):
            self.assertEqual(["a", "b"], [p["path"] for p in query.select_candidates(pages, [], mode)])
            self.assertEqual([], query.select_candidates(pages, ["nonexistentidentifier"], mode))
        self.assertEqual([], query.select_candidates([], []))

    def test_all_requires_each_keyword_but_accepts_aliases_across_fields(self):
        pages = [page("a", "tcgen05", body="consumer wait"),
                 page("b", "UMMA", tags=["wait"]), page("c", "tcgen05 UMMA"),
                 page("d", body="wait"), page("e", "unrelated")]
        candidates = query.select_candidates(pages, ["UMMA", "wait"], "all")
        self.assertEqual(["a", "b"], [p["path"] for p in candidates])
        self.assertEqual(["a", "b", "c", "d"],
                         [p["path"] for p in query.select_candidates(pages, ["UMMA", "wait"])])

    def test_excerpt_locates_body_evidence_beyond_the_opening(self):
        body = "opening " * 100 + "mbarrier wait\nconsumer release " + "trailing " * 100
        excerpt = query.match_excerpt(body, ["MBARRIER", "release"])
        self.assertIn("mbarrier wait consumer release", excerpt)
        self.assertTrue(excerpt.startswith("…") and excerpt.endswith("…"))
        self.assertLessEqual(len(excerpt), 242)
        self.assertNotIn("\n", excerpt)
        self.assertEqual("", query.match_excerpt(body, ["absent"]))

    def test_formatted_candidates_show_evidence_in_both_output_modes(self):
        candidate = query.select_candidates([page("a", "TMA", body="consumer wait")], ["wait"])[0]
        for compact in (False, True):
            with self.subTest(compact=compact):
                output = query.format_result(candidate, compact=compact)
                self.assertIn("body: wait", output)
                self.assertIn("consumer wait", output)
                self.assertIn("first body match", output)


class CandidateCLITests(unittest.TestCase):
    def setUp(self):
        self.pages = [
            page("wiki/hardware/c.md", "TMA", type="hardware", architectures=["sm90"]),
            page("wiki/hardware/a.md", "TMA", type="hardware", architectures=["sm100"],
                 tags=["tma"], languages=["cute-dsl"], confidence="verified"),
            page("sources/prs/cutlass/PR-1.md", body="TMA", repo="NVIDIA/cutlass"),
        ]

    def cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(query, "load_all_pages", return_value=self.pages), \
                patch.object(sys, "argv", ["query.py", *arguments]), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            query.main()
        return stdout.getvalue(), stderr.getvalue()

    def test_pagination_reaches_every_candidate_without_duplicates(self):
        found = []
        for offset in range(3):
            stdout, stderr = self.cli("TMA", "--limit", "1", "--offset", str(offset), "--paths-only")
            found.extend(stdout.splitlines())
            self.assertIn("3 total matches", stderr)
            self.assertIn("path order, not relevance order", stderr)
            if offset < 2:
                self.assertIn(f"--offset {offset + 1}", stderr)
            else:
                self.assertNotIn("More candidates", stderr)
        self.assertEqual(sorted(p["path"] for p in self.pages), found)

    def test_empty_page_distinguishes_out_of_range_and_no_matches(self):
        stdout, _ = self.cli("TMA", "--offset", "3")
        self.assertIn("No candidates at offset 3; 3 total matches", stdout)
        stdout, _ = self.cli("nonexistentidentifier")
        self.assertEqual("No matching pages.\n", stdout)

    def test_invalid_pagination_is_rejected(self):
        for arguments in (("--limit", "0"), ("--limit", "-1"), ("--offset", "-1")):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                self.cli(*arguments)
            self.assertEqual(2, raised.exception.code)

    def test_existing_filters_apply_before_pagination(self):
        stdout, stderr = self.cli("TMA", "--type", "hardware", "--architecture", "B200",
                                  "--tag", "tma", "--language", "cute-dsl", "--confidence",
                                  "verified", "--paths-only")
        self.assertEqual("wiki/hardware/a.md\n", stdout)
        self.assertIn("1 total matches", stderr)
        stdout, _ = self.cli("TMA", "--repo", "cutlass", "--paths-only")
        self.assertEqual("sources/prs/cutlass/PR-1.md\n", stdout)

    def test_default_and_compact_outputs_disclose_total_and_next_batch(self):
        for output_args in ([], ["--compact"]):
            with self.subTest(output_args=output_args):
                stdout, _ = self.cli("TMA", "--limit", "1", *output_args)
                self.assertIn("1 candidate(s) shown; 3 total matches", stdout)
                self.assertIn("--offset 1", stdout)
                self.assertIn("first body match", stdout)

    def test_all_keyword_mode_handles_quoted_and_separate_words(self):
        self.pages = [page("a", "TMA", body="wait"), page("b", "TMA"), page("c", body="wait")]
        for words in (("TMA wait",), ("TMA", "wait")):
            stdout, stderr = self.cli(*words, "--match", "all", "--paths-only")
            self.assertEqual("a\n", stdout)
            self.assertIn("1 total matches", stderr)
        stdout, _ = self.cli("TMA wait", "--match", "any", "--paths-only")
        self.assertEqual("a\nb\nc\n", stdout)

    def test_type_counts_cover_all_matches_after_filters_before_pagination(self):
        for output_args in ([], ["--compact"], ["--paths-only"]):
            stdout, stderr = self.cli("TMA", "--limit", "1", "--offset", "1", *output_args)
            summary = stderr if "--paths-only" in output_args else stdout
            self.assertIn("Types across all matches: source-pr=1, wiki-hardware=2", summary)
            if "--paths-only" in output_args:
                self.assertEqual("wiki/hardware/a.md\n", stdout)
        stdout, stderr = self.cli("TMA", "--type", "hardware", "--limit", "1", "--paths-only")
        self.assertIn("Types across all matches: wiki-hardware=2", stderr)
        self.assertNotIn("source-pr=", stderr)


if __name__ == "__main__":
    unittest.main()
