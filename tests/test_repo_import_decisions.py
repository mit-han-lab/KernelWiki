import copy
import tempfile
import unittest
from pathlib import Path
from repo_import_support import prepared, admission, document
from repo_import.decisions import validate_decisions


class DecisionTests(unittest.TestCase):
    def check(self, wiki, request, doc):
        return validate_decisions(doc, request, wiki, Path(request['snapshot']['checkout']))

    def test_pending_never_admitted_and_valid_include_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = prepared(Path(directory))
            from repo_import_support import read
            selected, pending = self.check(wiki, request, read(run / 'decisions.yaml'))
            self.assertEqual([], selected)
            self.assertEqual(1, len(pending))
            self.assertEqual(1, len(self.check(wiki, request, document(request))[0]))

    def test_stale_binding_and_duplicate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            doc = document(request)
            doc['decisions'][0]['content_digest'] = 'stale'
            with self.assertRaisesRegex(ValueError, 'binding'):
                self.check(wiki, request, doc)
            doc = document(request)
            doc['decisions'] *= 2
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                self.check(wiki, request, doc)

    def test_unregistered_target_and_missing_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            decision = admission(request)
            decision.update(action='extend_page', relation='new_version', target_implementation_id='missing')
            with self.assertRaisesRegex(ValueError, 'not registered'):
                self.check(wiki, request, document(request, decision))
            decision = admission(request)
            decision['evidence'][0]['sha256'] = 'fake'
            with self.assertRaisesRegex(ValueError, 'receipt'):
                self.check(wiki, request, document(request, decision))
