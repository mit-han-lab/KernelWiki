import tempfile
import unittest
from pathlib import Path
from repo_import_support import prepared, admission, document
from repo_import.decisions import validate_decisions


class ReproducibilityTests(unittest.TestCase):
    def test_higher_levels_require_bound_receipts_and_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            for level in ('runnable', 'benchmarked', 'concept', 'invented-level'):
                decision = admission(request)
                decision['metadata']['reproducibility'] = level
                with self.subTest(level=level), self.assertRaises(ValueError):
                    validate_decisions(document(request, decision), request, wiki, Path(request['snapshot']['checkout']))

    def test_verified_is_not_assigned_on_review_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, _, request = prepared(Path(directory))
            decision = admission(request)
            decision['metadata']['confidence'] = 'verified'
            with self.assertRaisesRegex(ValueError, 'evidence_basis'):
                validate_decisions(document(request, decision), request, wiki, Path(request['snapshot']['checkout']))

    def test_content_bound_upstream_reports_support_existing_higher_levels(self):
        from repo_import_support import commit, load_config, import_repo, read, write
        from repo_import.store import file_hash
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            (repo / 'report.md').write_text('Build, run, environment, inputs, method and performance report.\n')
            commit(repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            for level in ('runnable', 'benchmarked'):
                decision = admission(request)
                decision['metadata']['reproducibility'] = level
                decision['reproducibility_evidence'] = {
                    key: {'path': 'report.md', 'sha256': file_hash(repo / 'report.md'),
                          'locator': key, 'basis': 'upstream-reported'}
                    for key in ('build', 'run', 'environment', 'benchmark', 'inputs', 'method', 'results')}
                if level == 'benchmarked':
                    decision['performance_claims'] = [{'gpu': 'B200', 'dtype': 'fp32', 'shape': '1024',
                                                       'metric': 'GB/s', 'value': 100, 'source_locator': 'report.md'}]
                doc = document(request, decision)
                self.assertEqual(1, len(validate_decisions(doc, request, wiki, Path(request['snapshot']['checkout']))[0]))
                write(run / 'decisions.yaml', doc)
                plan = import_repo.build(wiki, run)
                from repo_import.store import digest
                self.assertEqual(digest(read(run / 'decisions.yaml')), plan['decisions_digest'])
