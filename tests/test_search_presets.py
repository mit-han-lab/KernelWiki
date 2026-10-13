"""Preset behavior at the CLI boundary, including decisions and replay receipts."""

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import query
from _retrieval_log import digest


def page(identifier, title, body="", **metadata):
    return {"path": f"wiki/patterns/{identifier}.md",
            "fm": {"id": identifier, "title": title, "type": "pattern", **metadata}, "body": body}


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.run = self.directory / "run"
        self.pages = [page("a", "pipeline waits"), page("b", "pipeline throughput"),
                      page("c", "pipeline", "throughput"), page("d", "unrelated throughput")]

    def cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(query, "load_all_pages", return_value=self.pages), \
                patch.object(sys, "argv", ["query.py", "--run-dir", str(self.run), *map(str, args)]), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            query.main()
        text = stdout.getvalue() + stderr.getvalue()
        match = re.search(r"query_id=([0-9a-f]{32})", text)
        folder = self.run / "queries" / match.group(1)
        return text, json.loads((folder / "result.json").read_text()), folder

    def context_file(self, value):
        path = self.directory / "context.json"
        path.write_text(json.dumps(value))
        return path

    def test_keyword_alias_and_guided_without_decisions_keep_master_order(self):
        self.pages += [page(f"tail-{i:02d}", "pipeline") for i in range(35)]
        _, baseline, _ = self.cli("pipeline", "--strategy", "master", "--paths-only")
        context = self.context_file({"state": {"evidence": ["waits resolved"], "reads": ["a"]}})
        for strategy in ("keyword", "guided"):
            with self.subTest(strategy=strategy):
                _, result, _ = self.cli("pipeline", "--strategy", strategy, "--context", context, "--paths-only")
                self.assertEqual(baseline["proposed_ids"], result["proposed_ids"])
                self.assertEqual(38, len(result["emitted_ids"]))

    def test_multi_query_gives_each_direction_a_turn_skipping_duplicates(self):
        self.pages = [page("a", "GEMM TMA"), page("b", "GEMM"),
                      page("c", "GEMM"), page("d", "TMA")]
        _, result, _ = self.cli("--strategy", "multi-query", "--query", "GEMM", "--query", "TMA",
                                "--limit", 2, "--paths-only")
        self.assertEqual(["a", "d", "b", "c"], result["proposed_ids"])
        self.assertEqual(["a", "d"], result["emitted_ids"])
        self.assertEqual(["b", "c"], result["omitted_by_output_limits"])
        self.assertEqual(["GEMM", "TMA"], [s["query"] for s in result["searches"]])
        self.assertEqual("round-robin", result["execution"]["selection"]["merge"])

    def test_multi_query_json_supports_per_search_filters_under_global_constraints(self):
        self.pages = [page("a", "TMA", architectures=["sm100"], type="hardware"),
                      page("b", "wait", architectures=["sm100"]),
                      page("c", "wait", architectures=["sm90"]),
                      page("d", "TMA", architectures=["sm100"])]
        context = self.context_file({"queries": [{"query": "TMA", "type": "hardware"}, "wait"]})
        _, result, _ = self.cli("--strategy", "multi-query", "--context", context,
                                "--architecture", "sm100", "--paths-only")
        self.assertEqual(["a", "b"], result["proposed_ids"])

    def test_guided_focus_changes_order_before_limit_without_expanding_recall(self):
        for focus, first, matching in (("waits", "a", 1), ("throughput", "b", 2), ("unmatched", "a", 0)):
            with self.subTest(focus=focus):
                output, result, _ = self.cli("pipeline", "--strategy", "guided", "--focus", focus,
                                             "--limit", 1, "--max-chars", 2048)
                self.assertEqual([first], result["emitted_ids"])
                self.assertEqual({"a", "b", "c"}, set(result["proposed_ids"]))
                selection = result["execution"]["selection"]
                self.assertEqual(focus, selection["focus"])
                self.assertEqual(matching, selection["focus_match_count"])
                self.assertIn('"focus":', output)
                self.assertLessEqual(len(output), 2048)

    def test_guided_preference_and_deferral_override_focus_without_excluding_pages(self):
        _, result, _ = self.cli("pipeline", "--strategy", "guided", "--focus", "throughput",
                                "--prefer", "c", "--prefer", "c", "--defer", "b", "--paths-only")
        self.assertEqual(["c", "a", "b"], result["proposed_ids"])
        self.assertEqual(["c"], result["execution"]["selection"]["priority_ids"])

    def test_guided_focus_excerpt_uses_alias_without_changing_recall_or_query_matches(self):
        self.pages = [
            page("a", "pipeline", "pipeline QUERY_ONLY_EVIDENCE. " + "padding " * 100 +
                 "tcgen05 FOCUS_ONLY_EVIDENCE."),
            page("b", "unrelated", "tcgen05 should not expand recall"),
        ]
        for flags in ((), ("--compact",)):
            with self.subTest(flags=flags):
                baseline_output, baseline, _ = self.cli(
                    "pipeline", "--strategy", "keyword", "--max-chars", 4096, *flags)
                output, result, folder = self.cli(
                    "pipeline", "--strategy", "guided", "--focus", "UMMA",
                    "--max-chars", 4096, *flags)
                self.assertIn("QUERY_ONLY_EVIDENCE", baseline_output)
                self.assertNotIn("FOCUS_ONLY_EVIDENCE", baseline_output)
                self.assertIn("first focus body match", output)
                self.assertIn("tcgen05 FOCUS_ONLY_EVIDENCE", output)
                self.assertNotIn("QUERY_ONLY_EVIDENCE", output)
                self.assertEqual(["a"], result["proposed_ids"])
                for key in ("proposed_ids", "emitted_ids", "searches", "candidate_matches"):
                    self.assertEqual(baseline[key], result[key])
                self.assertEqual(output, (folder / "output.txt").read_text())
                self.assertLessEqual(len(output), 4096)

    def test_guided_excerpt_falls_back_when_focus_has_no_body_match(self):
        query_body = "Opening. " + "padding " * 100 + "pipeline QUERY_BODY_EVIDENCE."
        preview_body = "OPENING_PREVIEW_EVIDENCE. " + "padding " * 100
        cases = [
            ("throughput", query_body, "first body match", "QUERY_BODY_EVIDENCE"),
            ("unmatched", query_body, "first body match", "QUERY_BODY_EVIDENCE"),
            ("", query_body, "first body match", "QUERY_BODY_EVIDENCE"),
            ("throughput", preview_body, "body preview", "OPENING_PREVIEW_EVIDENCE"),
        ]
        for focus, body, label, evidence in cases:
            self.pages = [page("a", "pipeline throughput", body)]
            for flags in ((), ("--compact",)):
                with self.subTest(focus=focus, label=label, flags=flags):
                    output, result, _ = self.cli(
                        "pipeline", "--strategy", "guided", "--focus", focus,
                        "--max-chars", 4096, *flags)
                    self.assertIn(label, output)
                    self.assertIn(evidence, output)
                    self.assertNotIn("first focus body match", output)
                    self.assertEqual(["a"], result["emitted_ids"])

    def test_guided_extra_queries_add_new_evidence_and_keep_global_filters(self):
        self.pages[0]["fm"]["architectures"] = ["sm100"]
        self.pages[-1]["fm"]["architectures"] = ["sm100"]
        context = self.context_file({"decision": {"filters": {"tag": "base"},
                                                  "extra_queries": [{"query": "unrelated"}]}})
        self.pages[0]["fm"]["tags"] = ["base"]
        _, result, _ = self.cli("pipeline", "--strategy", "guided", "--context", context,
                                "--architecture", "sm100", "--prefer", "d", "--paths-only")
        self.assertEqual(["d", "a"], result["proposed_ids"])
        self.assertEqual("round-robin", result["execution"]["selection"]["merge"])

    def test_guided_cli_overrides_only_supplied_fields_and_snapshots_effective_decision(self):
        original = {"query": "old", "state": {"question": "current evidence"}, "decision": {
            "focus": "waits", "priority_ids": ["a"], "extra_queries": ["old"],
            "defer_ids": ["c"], "filters": {"type": "pattern"}, "reason": "old reason"}}
        context = self.context_file(original)
        _, result, folder = self.cli("pipeline", "--strategy", "guided", "--context", context,
                                     "--focus", "throughput", "--prefer", "b", "--extra-query", "unrelated",
                                     "--reason", "trace shows math bottleneck", "--paths-only")
        effective = json.loads((folder / "context.json").read_text())
        self.assertEqual("pipeline", effective["query"])
        self.assertEqual(original["state"], effective["state"])
        self.assertEqual({"focus": "throughput", "priority_ids": ["b"], "defer_ids": ["c"],
                          "filters": {"type": "pattern"}, "extra_queries": [{"query": "unrelated"}],
                          "reason": "trace shows math bottleneck"}, effective["decision"])
        self.assertEqual(original, json.loads(context.read_text()))
        self.assertEqual("b", result["proposed_ids"][0])
        self.assertEqual("c", result["proposed_ids"][-1])

    def test_ambiguous_inputs_and_wrong_preset_flags_fail_instead_of_being_ignored(self):
        context = self.context_file({"queries": ["TMA", "wait"]})
        cases = [("--strategy", "keyword", "--focus", "wait"),
                 ("--strategy", "guided", "--query", "TMA"),
                 ("--strategy", "multi-query", "--query", "TMA", "--query", "wait", "pipeline"),
                 ("--strategy", "multi-query", "--context", context, "pipeline"),
                 ("--strategy", "multi-query", "--context", context, "--query", "TMA"),
                 ("--strategy", "multi-query", "--query", "TMA"),
                 ("--strategy", "multi-query", "--query", "", "--query", "wait")]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                self.cli(*args)
            self.assertEqual(2, raised.exception.code)

    def test_guided_invalid_or_stale_decisions_fail_with_error_receipts(self):
        cases = [{"priority_ids": ["missing"]}, {"priority_ids": ["d"]},
                 {"defer_ids": ["a"], "priority_ids": ["a"]}, {"defer_ids": "a"},
                 {"focus": []}, {"reason": {}}, {"filters": []}, {"focus_typo": "TMA"},
                 {"extra_queries": [{}]}, {"extra_queries": "TMA"}]
        for decision in cases:
            with self.subTest(decision=decision), self.assertRaises(SystemExit):
                self.cli("pipeline", "--strategy", "guided", "--context",
                         self.context_file({"decision": decision}))
        for path in (self.run / "queries").glob("*/result.json"):
            self.assertEqual("error", json.loads(path.read_text())["status"])

    def test_presets_snapshot_helper_and_replay_the_same_candidate_order(self):
        _, result, folder = self.cli("pipeline", "--strategy", "guided", "--focus", "throughput",
                                     "--defer", "a", "--paths-only")
        helper = (folder / "engine/_search_presets.py").read_bytes()
        self.assertEqual((ROOT / "scripts/_search_presets.py").read_bytes(), helper)
        request = json.loads((folder / "request.json").read_text())
        self.assertEqual(digest(helper), request["engine_sha256"]["_search_presets.py"])
        # Load the captured helper, not the installed policy implementation.
        import types
        module = types.ModuleType("_search_presets")
        exec(compile(helper, str(folder / "engine/_search_presets.py"), "exec"), module.__dict__)
        with patch.dict(sys.modules, _search_presets=module):
            replay = query.execute_strategy((folder / "strategy.py").read_bytes(), folder / "strategy.py",
                                            query.StrategyIndex(self.pages),
                                            json.loads((folder / "context.json").read_text()))
        self.assertEqual(result["proposed_ids"], [p["fm"]["id"] for p in replay])


if __name__ == "__main__":
    unittest.main()
