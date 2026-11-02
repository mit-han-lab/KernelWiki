"""Missing evidence must fail closed at every admission and reuse boundary."""
import tempfile
import unittest
from pathlib import Path

from repo_import_support import prepared, document
from repo_import.decisions import validate_decisions, unresolved_overlaps, validate_target_evidence
from repo_import.discovery import definition_receipt, page_receipt, validate_discoveries
from repo_import.selection import propose_selection, validate_selection
from repo_import.store import file_hash, read, valid_receipt
from repo_import.updates import _receipts_valid


class EvidenceReceiptTests(unittest.TestCase):
    def test_missing_digest_or_file_is_never_matching_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.txt'
            missing = {'path': path.name, 'sha256': None, 'locator': 'Selected entry'}
            self.assertFalse(valid_receipt(path, missing))
            path.write_text('Actual evidence\n')
            receipt = {**missing, 'sha256': file_hash(path)}
            self.assertTrue(valid_receipt(path, receipt))
            for checksum in (None, '', 'f' * 63, 'z' * 64, 42):
                with self.subTest(checksum=checksum):
                    self.assertFalse(valid_receipt(path, {**receipt, 'sha256': checksum}))
            for locator in (None, '', '   ', 42):
                with self.subTest(locator=locator):
                    self.assertFalse(valid_receipt(path, {**receipt, 'locator': locator}))
            self.assertFalse(valid_receipt(path.parent, receipt))
            path.unlink()
            self.assertFalse(valid_receipt(path, receipt))

    def test_missing_source_receipt_rejects_all_decision_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            checkout = Path(request['snapshot']['checkout'])
            for verdict in ('include', 'exclude', 'defer'):
                doc = document(request)
                doc['decisions'][0]['decision'] = verdict
                doc['decisions'][0]['evidence'].append(
                    {'path': 'missing-source.cu', 'sha256': None, 'locator': 'Helper'})
                with self.subTest(verdict=verdict), self.assertRaisesRegex(ValueError, 'receipt'):
                    validate_decisions(doc, request, wiki, checkout)

    def test_missing_selection_receipt_cannot_omit_a_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, run, request = prepared(Path(directory))
            inventory = read(run / 'inventory.yaml')
            checkout = Path(request['snapshot']['checkout'])
            selection = propose_selection(inventory, checkout)
            row = selection['entries'][0]
            row.update(role='helper', rationale='Claimed support-only purpose.',
                       caller_basis='Claimed caller evidence.',
                       evidence=[definition_receipt(inventory['candidates'][0]),
                                 {'path': 'missing-caller.py', 'sha256': None, 'locator': 'launch'}])
            with self.assertRaisesRegex(ValueError, 'receipt'):
                validate_selection(selection, inventory, checkout)

    def test_missing_additional_overlap_evidence_cannot_clear_a_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'kernel.cu'
            path.write_text('__global__ void kernel() { work(); }\n')
            evidence = {'path': path.name, 'sha256': file_hash(path), 'locator': 'kernel'}
            for kind in ('batch', 'pr', 'kernel'):
                hint = {'kind': kind, 'match_id': 'match', 'candidate_id': 'other', 'evidence': evidence}
                decision = {'candidate_id': 'self', 'overlap_resolutions': [
                    {'match_id': 'match', 'outcome': 'distinct', 'rationale': 'Claimed difference.',
                     'evidence': [evidence, {'path': 'missing.cu', 'sha256': None, 'locator': 'Other body'}]}]}
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'receipt'):
                    unresolved_overlaps(decision, [hint], {}, root, root, {'other': []})

    def test_missing_target_receipt_cannot_authorize_existing_page_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = {'equivalence_basis': 'Claimed same implementation.', 'target_evidence': [
                {'path': 'wiki/kernels/missing.md', 'sha256': None, 'locator': 'Existing implementation'}]}
            with self.assertRaisesRegex(ValueError, 'receipt'):
                validate_target_evidence(decision, Path(directory))

    def test_missing_run_report_cannot_raise_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            doc = document(request)
            decision = doc['decisions'][0]
            decision['metadata']['reproducibility'] = 'runnable'
            decision['reproducibility_evidence'] = {
                key: {'path': 'missing-report.md', 'sha256': None, 'locator': key,
                      'basis': 'upstream-reported'} for key in ('build', 'run', 'environment')}
            with self.assertRaisesRegex(ValueError, 'reproducibility receipt'):
                validate_decisions(doc, request, wiki, Path(request['snapshot']['checkout']))

    def test_update_reuse_rejects_missing_or_null_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'caller.py'
            path.write_text('launch()\n')
            row = {'evidence': [{'path': path.name, 'sha256': file_hash(path), 'locator': 'launch'}]}
            self.assertTrue(_receipts_valid([row], root))
            path.unlink()
            self.assertFalse(_receipts_valid([row], root))
            row['evidence'][0]['sha256'] = None
            self.assertFalse(_receipts_valid([row], root))

    def test_discovery_rejects_missing_known_page_with_null_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            candidate = request['candidates'][0]
            page = {'page_id': 'kernel-missing', 'path': 'wiki/kernels/missing.md', 'sha256': None}
            discovery = {'version': 1, 'entries': [
                {'candidate_id': candidate['candidate_id'], 'page_id': page['page_id'],
                 'rationale': 'Claimed related existing implementation.',
                 'input_evidence': [definition_receipt(candidate)], 'matched_evidence': [page_receipt(page)]}]}
            with self.assertRaisesRegex(ValueError, 'evidence hash'):
                validate_discoveries(discovery, [candidate], [page], wiki,
                                     Path(request['snapshot']['checkout']))
