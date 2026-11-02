import tempfile
import unittest
from pathlib import Path
from repo_import_support import prepared, document
from repo_import.decisions import rebind_manifest
from repo_import.config import load_preset, load_config


class PresetTests(unittest.TestCase):
    def test_only_matching_content_is_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, request = prepared(Path(directory))
            manifest = document(request)
            request['candidates'][0]['candidate_id'] = 'new-snapshot-id'
            self.assertEqual('new-snapshot-id', rebind_manifest(manifest, request)['decisions'][0]['candidate_id'])
            request['candidates'][0]['content_digest'] = 'changed'
            self.assertEqual([], rebind_manifest(manifest, request)['decisions'])
            exclusion = document(request)
            exclusion['decisions'][0]['decision'] = 'exclude'
            request['context_digest'] = 'new-context'
            rebound = rebind_manifest(exclusion, request)
            self.assertEqual('exclude', rebound['decisions'][0]['decision'])
            self.assertEqual('new-context', rebound['context_digest'])

    def test_unknown_preset_is_not_an_implicit_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, _ = prepared(Path(directory))
            with self.assertRaisesRegex(ValueError, 'Unknown preset'):
                load_preset(wiki, 'anything', load_config(wiki))
