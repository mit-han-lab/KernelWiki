import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "verify_pr_architecture_upstream",
    ROOT / "scripts" / "verify_pr_architecture_upstream.py",
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class LiveArchitectureBoundaryTests(unittest.TestCase):
    def test_documented_3000_file_cap_preserves_authoritative_total(self):
        pull = {"changed_files": 3750}
        batches = [[{"filename": f"src/{page}-{index}.h"} for index in range(100)]
                   for page in range(30)]
        full_diff_files = [
            {"filename": f"src/full-{index}.h", "patch": "", "status": "modified"}
            for index in range(3750)
        ]
        with mock.patch.object(VERIFY, "github_json", side_effect=[pull, *batches, []]), \
             mock.patch.object(VERIFY, "fetch_full_diff_files", return_value=full_diff_files):
            actual_pull, files, complete, evidence_files = VERIFY.fetch_upstream(
                "example/project", 9
            )
        self.assertEqual(3750, actual_pull["changed_files"])
        self.assertEqual(3000, len(files))
        self.assertFalse(complete)
        self.assertEqual(3750, len(evidence_files))

    def test_capped_tail_is_evaluated_with_full_scope_policy(self):
        listed = [{"filename": f"docs/{index}.md", "patch": "", "status": "modified"}
                  for index in range(3000)]
        full = [*listed, {
            "filename": "src/tail_kernel.h",
            "status": "added",
            "patch": "+CUTLASS_DEVICE void tail_kernel();",
        }]
        expected = VERIFY.derive_upstream_record(
            {"title": "Kernel update", "body": "", "changed_files": 3001},
            listed,
            False,
            full,
        )
        self.assertTrue(expected["retain"])
        self.assertEqual("device-code-signal", expected["scope_rule"])
        self.assertEqual(["src/tail_kernel.h"], expected["scope_paths"])
        self.assertTrue(expected["changed_files_evidence_complete"])

    def test_live_complete_file_evidence_distinguishes_device_and_host_cuda(self):
        files = [
            {
                "filename": "csrc/device.cu",
                "status": "modified",
                "patch": "+update_dispatch();",
            },
            {
                "filename": "csrc/host.cu",
                "status": "modified",
                "patch": '+m.def("op", &op);',
            },
        ]
        def payload(url):
            if url.endswith("device.cu"):
                return b"__global__ void kernel() {}"
            return b"void register_op() {}"

        with mock.patch.object(VERIFY, "_fetch_bytes", side_effect=payload):
            actual = VERIFY.attach_complete_cuda_evidence(
                "example/project", "a" * 40, files
            )
        self.assertTrue(actual[0]["complete_file_device_signal"])
        self.assertFalse(actual[1]["complete_file_device_signal"])
        self.assertTrue(VERIFY.derive_upstream_record(
            {"title": "Update", "body": "", "changed_files": 1}, [actual[0]]
        )["retain"])
        self.assertFalse(VERIFY.derive_upstream_record(
            {"title": "Update", "body": "", "changed_files": 1}, [actual[1]]
        )["retain"])

    def test_live_complete_file_evidence_distinguishes_python_dsl_from_host(self):
        files = [
            {
                "filename": "ops/kernel.py",
                "status": "modified",
                "patch": "+num_warps = 8",
            },
            {
                "filename": "layers/config.py",
                "status": "modified",
                "patch": "+num_blocks = (n + block_n - 1) // block_n",
            },
        ]

        def payload(url):
            if url.endswith("kernel.py"):
                return b"@triton.jit\ndef kernel(x):\n    return tl.load(x)\n"
            return b"def blocks(n, block_n):\n    return (n + block_n - 1) // block_n\n"

        with mock.patch.object(VERIFY, "_fetch_bytes", side_effect=payload):
            actual = VERIFY.attach_complete_file_evidence(
                "example/project", "a" * 40, files
            )
        self.assertTrue(actual[0]["complete_file_python_dsl_signal"])
        self.assertEqual(["triton"], actual[0]["complete_file_python_dsl_languages"])
        self.assertFalse(actual[1]["complete_file_python_dsl_signal"])
        self.assertTrue(VERIFY.derive_upstream_record(
            {"title": "Update", "body": "", "changed_files": 1}, [actual[0]]
        )["retain"])
        self.assertFalse(VERIFY.derive_upstream_record(
            {"title": "Update", "body": "", "changed_files": 1}, [actual[1]]
        )["retain"])

    def test_joint_local_fabrication_fails_external_record(self):
        files = [{
            "filename": "src/mma_sm75.cu",
            "status": "added",
            "patch": "+__global__ void mma_sm75() {}",
            "additions": 1,
            "deletions": 0,
        }]
        expected = VERIFY.derive_upstream_record(
            {"title": "Add SM75 MMA support", "body": "Turing kernel."}, files
        )
        fabricated = {
            "title": expected["title"],
            "architectures": ["sm100"],
            "architecture_disposition": "exact",
            "architecture_evidence": [{"architecture": "sm100"}],
            "upstream_body_sha256": expected["upstream_body_sha256"],
            "upstream_files_sha256": expected["upstream_files_sha256"],
            "changed_files_count": 1,
            "changed_paths": ["src/mma_sm75.cu"],
            "changed_paths_complete": True,
            "scope_disposition": "retained",
            "scope_evidence": {
                "rule": expected["scope_rule"],
                "paths": expected["scope_paths"],
            },
            **expected["metadata"],
        }
        errors = VERIFY.compare_local_page("fixture", fabricated, expected)
        self.assertTrue(any("architectures" in error for error in errors))

    def test_removed_upstream_record_rejects_local_page(self):
        expected = VERIFY.derive_upstream_record(
            {"title": "Add host UVA view", "body": ""},
            [{
                "filename": "csrc/view.cu",
                "status": "added",
                "patch": "+void view() { cudaHostGetDevicePointer(0, 0, 0); }",
                "additions": 1,
                "deletions": 0,
            }],
        )
        self.assertFalse(expected["retain"])
        self.assertTrue(VERIFY.compare_local_page("fixture", {}, expected))


if __name__ == "__main__":
    unittest.main()
