import copy
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("validate", ROOT / "scripts" / "validate.py")
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


class ArchitectureReceiptTests(unittest.TestCase):
    def fixture(self):
        fm = {
            "upstream_body_sha256": "a" * 64,
            "upstream_files_sha256": "b" * 64,
            "architectures": [],
            "architecture_disposition": "unknown",
            "architecture_evidence": [{
                "architecture": "unknown",
                "basis": "explicit-unknown",
                "locator": "complete upstream evidence",
                "evidence": "No supported architecture signal.",
            }],
        }
        rows = {"tilelang/PR-2198.md": VALIDATE.architecture_receipt_record(fm)}
        receipt = {
            "schema_version": 1,
            "policy_sha256": hashlib.sha256(
                (ROOT / "scripts" / "pr_policy.py").read_bytes()
            ).hexdigest(),
            "row_fields": [
                "upstream_body_sha256",
                "upstream_files_sha256",
                "architectures",
                "architecture_disposition",
                "architecture_evidence_sha256",
            ],
            "rows": copy.deepcopy(rows),
        }
        return fm, rows, receipt

    def test_exact_receipt_match_is_clean(self):
        _, rows, receipt = self.fixture()
        self.assertEqual([], VALIDATE.compare_pr_architecture_receipt(rows, receipt))

    def test_coherently_fabricated_page_fails_receipt(self):
        fm, _, receipt = self.fixture()
        fm["architectures"] = ["sm100"]
        fm["architecture_disposition"] = "exact"
        fm["architecture_evidence"] = [{
            "architecture": "sm100",
            "basis": "exact-sm-token",
            "locator": "fabricated",
            "evidence": "SM100",
        }]
        fabricated = {"tilelang/PR-2198.md": VALIDATE.architecture_receipt_record(fm)}
        errors = VALIDATE.compare_pr_architecture_receipt(fabricated, receipt)
        self.assertTrue(any("architecture receipt mismatch" in row for row in errors))


if __name__ == "__main__":
    unittest.main()
