import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from repo_import_support import prepared, commit, import_repo, load_config, read
from repo_import.checkout import canonical_repo
from repo_import import scan
from repo_import.store import file_hash


class SnapshotTests(unittest.TestCase):
    def test_remote_identity_normalization(self):
        self.assertEqual(canonical_repo('git@github.com:owner/repo.git'), canonical_repo('https://github.com/owner/repo/'))

    def test_dirty_worktree_does_not_change_pinned_input(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, old = prepared(Path(directory))
            (repo / 'kernel.py').write_text('uncommitted source')
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            new = read(run / 'review-request.yaml')
            self.assertTrue(new['snapshot']['ignored_worktree_changes'])
            self.assertEqual(old['candidates'], new['candidates'])

    def test_modified_cached_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, request = prepared(Path(directory))
            (Path(request['snapshot']['checkout']) / 'kernel.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'modified'):
                import_repo.check_snapshot(request)

    def test_shared_dependencies_are_parsed_once_without_losing_transitive_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            code = 'import triton\nfrom shared import helper\n@triton.jit\ndef kernel(x):\n    return helper(x)\n'
            (repo / 'kernel.py').write_text(code)
            (repo / 'other.py').write_text(code)
            (repo / 'shared.py').write_text('from leaf import VALUE\ndef helper(x):\n    return x + VALUE\n')
            (repo / 'leaf.py').write_text('from shared import helper\nVALUE = 2\n')
            commit(repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            with patch.object(scan, 'dependency_paths', wraps=scan.dependency_paths) as dependency_reads:
                actual = scan.scan_snapshot(request['snapshot'], request['scan_config'])
            self.assertEqual(4, dependency_reads.call_count)
            self.assertEqual(request['candidates'], actual['candidates'])
            checkout = Path(request['snapshot']['checkout'])
            for c in actual['candidates']:
                expected = {c['path'], 'shared.py', 'leaf.py'}
                self.assertEqual(expected, {r['path'] for r in c['dependencies']})
                self.assertTrue(all(r['sha256'] == file_hash(checkout / r['path']) for r in c['dependencies']))
            # Cache must be local to one scan: changes cannot reuse old hashes.
            (checkout / 'leaf.py').write_text('VALUE = 3\n')
            changed = scan.scan_snapshot(request['snapshot'], request['scan_config'])
            self.assertNotEqual(actual['candidates'][0]['content_digest'], changed['candidates'][0]['content_digest'])
