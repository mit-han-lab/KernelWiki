import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from repo_import_support import prepared, document, import_repo, write, read
from repo_import.store import contained
from repo_import import store


class StoreTests(unittest.TestCase):
    def test_rebuild_from_persisted_review_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = prepared(Path(directory))
            write(run / 'decisions.yaml', document(request))
            first = import_repo.build(wiki, run)
            self.assertEqual(first, import_repo.build(wiki, run))
            self.assertEqual('built', read(run / 'result.yaml')['status'])

    def test_paths_cannot_escape_persistent_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                contained(Path(directory), '../../outside')

    def test_accelerated_and_bundled_safe_readers_preserve_the_same_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.yaml'
            path.write_text('version: 1\nfirst: &shared {enabled: true, hash: "0123", count: 2}\nsecond: *shared\n')
            accelerated = read(path)
            fallback = SimpleNamespace(safe_load=store.yaml.safe_load)
            with patch.object(store, 'yaml', fallback):
                self.assertEqual(accelerated, read(path))
                path.write_text('')
                self.assertEqual({'default': True}, read(path, {'default': True}))
