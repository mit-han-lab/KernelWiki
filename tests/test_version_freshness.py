import importlib.util
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_version_freshness", ROOT / "scripts" / "check_version_freshness.py"
)
FRESHNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRESHNESS)


class ToolSnapshotFreshnessTests(unittest.TestCase):
    def findings(self, checked_at="2026-08-18", released_at="2026-06-18"):
        tool = {
            "tool": "triton",
            "releases": [{"name": "3.7.1", "released_at": released_at}],
        }
        if checked_at is not None:
            tool["upstream_checked_at"] = checked_at
        return list(FRESHNESS.check_tool_versions(
            {"tools": [tool]}, date(2026, 8, 18), 180
        ))

    def test_recent_upstream_receipt_is_clean(self):
        self.assertFalse([row for row in self.findings() if row[0] == "warn"])

    def test_missing_upstream_receipt_warns(self):
        messages = [message for severity, message in self.findings(None) if severity == "warn"]
        self.assertTrue(any("missing upstream_checked_at" in message for message in messages))

    def test_stale_upstream_receipt_warns(self):
        messages = [
            message for severity, message in self.findings("2025-12-01")
            if severity == "warn"
        ]
        self.assertTrue(any("upstream_checked_at" in message and "ago" in message for message in messages))

    def test_release_cannot_postdate_upstream_receipt(self):
        messages = [
            message for severity, message in self.findings("2026-05-01", "2026-06-18")
            if severity == "warn"
        ]
        self.assertTrue(any("newer than upstream_checked_at" in message for message in messages))


class ReleaseReceiptEqualityTests(unittest.TestCase):
    def fixtures(self):
        registry = {
            "tools": [{
                "tool": "triton",
                "upstream_checked_at": "2026-08-18",
                "releases": [{"name": "3.7.1", "released_at": "2026-06-18"}],
            }]
        }
        receipt = {
            "tools": {
                "triton": {
                    "upstream_checked_at": "2026-08-18",
                    "releases": [["3.7.1", "2026-06-18"]],
                }
            }
        }
        return registry, receipt

    def warnings(self, registry, receipt):
        return [
            message for severity, message in FRESHNESS.check_release_receipt(registry, receipt)
            if severity == "warn"
        ]

    def test_exact_registry_receipt_match_is_clean(self):
        registry, receipt = self.fixtures()
        self.assertEqual([], self.warnings(registry, receipt))

    def test_missing_receipted_release_warns(self):
        registry, receipt = self.fixtures()
        registry["tools"][0]["releases"] = []
        self.assertTrue(any("missing receipted releases" in row for row in self.warnings(registry, receipt)))

    def test_unreceipted_registry_release_warns(self):
        registry, receipt = self.fixtures()
        registry["tools"][0]["releases"].append(
            {"name": "3.8.0", "released_at": "2026-08-19"}
        )
        self.assertTrue(any("unreceipted releases" in row for row in self.warnings(registry, receipt)))

    def test_checked_at_mismatch_warns(self):
        registry, receipt = self.fixtures()
        registry["tools"][0]["upstream_checked_at"] = "2026-08-19"
        self.assertTrue(any("upstream_checked_at mismatch" in row for row in self.warnings(registry, receipt)))


if __name__ == "__main__":
    unittest.main()
