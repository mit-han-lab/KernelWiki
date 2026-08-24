import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_verbatim", ROOT / "scripts" / "verify_verbatim.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CanonicalizeUpstreamPatchTests(unittest.TestCase):
    def test_git_oid_abbreviation_length_is_not_semantic(self):
        local = (
            b"diff --git a/a.c b/a.c\n"
            b"index 3053d4b98d9..f06dc141324 100644\n"
            b"--- a/a.c\n+++ b/a.c\n@@ -1 +1 @@\n-old\n+new\n"
        )
        upstream = local.replace(
            b"3053d4b98d9..f06dc141324",
            b"3053d4b98d9f..f06dc1413244",
        )
        self.assertNotEqual(local, upstream)
        self.assertEqual(
            MODULE.canonicalize_upstream_patch(local),
            MODULE.canonicalize_upstream_patch(upstream),
        )

    def test_patch_hunk_change_remains_discriminating(self):
        baseline = (
            b"diff --git a/a.c b/a.c\n"
            b"index 1111111..2222222 100644\n"
            b"--- a/a.c\n+++ b/a.c\n@@ -1 +1 @@\n-old\n+new\n"
        )
        corrupted = baseline.replace(b"+new\n", b"+wrong\n")
        self.assertNotEqual(
            MODULE.canonicalize_upstream_patch(baseline),
            MODULE.canonicalize_upstream_patch(corrupted),
        )

    def test_file_mode_remains_discriminating(self):
        baseline = b"index abcdef1..abcdef2 100644\n"
        changed_mode = b"index abcdef1..abcdef2 100755\n"
        self.assertNotEqual(
            MODULE.canonicalize_upstream_patch(baseline),
            MODULE.canonicalize_upstream_patch(changed_mode),
        )


if __name__ == "__main__":
    unittest.main()
