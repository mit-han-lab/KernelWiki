import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "refresh_candidate_ledger", SCRIPTS / "refresh_candidate_ledger.py"
)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class RefreshCandidatePolicyTests(unittest.TestCase):
    def test_refresh_uses_positive_kernel_policy(self):
        def fetcher(_repo, _number):
            return (
                {"title": "Add kernel", "body": ""},
                [{"filename": "csrc/kernel.cu", "patch": "+__global__ void kernel() {}"}],
            )

        decision, reason, paths = refresh.classify_search_hit(
            "example/project", {"number": 1, "title": "stale"}, fetcher
        )
        self.assertEqual("include", decision)
        self.assertTrue(reason.startswith("retain:"))
        self.assertEqual(["csrc/kernel.cu"], paths)

    def test_refresh_excludes_host_python_even_under_kernelish_path(self):
        def fetcher(_repo, _number):
            return (
                {"title": "Update fused MoE placement", "body": "host configuration"},
                [{"filename": "python/fused_moe/layer.py", "patch": "+def configure():\n+    pass"}],
            )

        decision, reason, _ = refresh.classify_search_hit(
            "example/project", {"number": 2, "title": "stale"}, fetcher
        )
        self.assertEqual("exclude", decision)
        self.assertTrue(reason.startswith("remove:"))

    def test_refresh_preserves_external_evidence_failure_as_defer(self):
        decision, reason, paths = refresh.classify_search_hit(
            "example/project", {"number": 3, "title": "unknown"}, lambda *_: None
        )
        self.assertEqual(("defer", "authoritative PR evidence unavailable; needs-triage", []), (decision, reason, paths))


if __name__ == "__main__":
    unittest.main()
