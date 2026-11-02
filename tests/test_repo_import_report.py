import tempfile
import unittest
from pathlib import Path

from repo_import_support import prepared, document, admission, read, write
from repo_import.report import report_run


class ReportTests(unittest.TestCase):
    def test_validated_list_does_not_build_or_change_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = prepared(Path(directory))
            write(run / 'decisions.yaml', document(request))
            before = {path.relative_to(wiki): path.read_bytes() for path in wiki.rglob('*')
                      if path.is_file() and '.cache' not in path.parts}
            out = Path(directory) / 'lists'
            summary = report_run(wiki, run, out)
            after = {path.relative_to(wiki): path.read_bytes() for path in wiki.rglob('*')
                     if path.is_file() and '.cache' not in path.parts}
            self.assertEqual(before, after)
            self.assertEqual(1, summary['effective_kernels'])
            self.assertEqual(0, summary['review_unfinished'])
            self.assertFalse((run / 'publication-plan.yaml').exists())
            self.assertFalse((run / 'staging').exists())
            self.assertEqual('kernel.py', read(out / 'valid-kernels.yaml')['implementations'][0]['representatives'][0]['path'])

    def test_list_only_admission_enforces_field_vocabulary_and_unique_lists(self):
        from repo_import_support import make_repo, make_wiki, import_repo, load_config
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wiki, repo = make_wiki(base), make_repo(base)
            write(wiki / 'data/tags.yaml', {'architectures': ['sm100'], 'kernel_types': ['reduction'],
                'languages': ['triton'], 'techniques': [], 'hardware_features': ['tma']})
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            for field, values in [('tags', ['sm100']), ('kernel_types', ['triton']),
                                  ('architectures', ['unlisted']), ('languages', ['triton', 'triton']),
                                  ('tags', 'reduction')]:
                with self.subTest(field=field, values=values):
                    verdict = admission(request)
                    verdict['metadata'][field] = values
                    write(run / 'decisions.yaml', document(request, verdict))
                    with self.assertRaisesRegex(ValueError, 'Metadata|metadata'):
                        report_run(wiki, run, base / 'lists')
            verdict = admission(request)
            verdict['metadata'].update(tags=['tma'], hardware_features=[])
            write(run / 'decisions.yaml', document(request, verdict))
            with self.assertRaisesRegex(ValueError, 'Hardware tags'):
                report_run(wiki, run, base / 'lists')
            verdict['metadata']['hardware_features'] = ['tma']
            write(run / 'decisions.yaml', document(request, verdict))
            self.assertEqual(1, report_run(wiki, run, base / 'lists')['effective_kernels'])

    def test_unfinished_and_explicit_defer_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = prepared(Path(directory))
            out = Path(directory) / 'lists'
            summary = report_run(wiki, run, out)
            self.assertEqual(0, summary['effective_kernels'])
            self.assertEqual(1, summary['review_unfinished'])
            self.assertEqual(0, summary['review_defer'])
            decision = read(run / 'decisions.yaml')['decisions'][0]
            decision['rationale'] = 'Missing pinned dependency prevents a justified admission.'
            write(run / 'decisions.yaml', document(request, decision))
            summary = report_run(wiki, run, out)
            self.assertEqual(0, summary['review_unfinished'])
            self.assertEqual(1, summary['review_defer'])

    def test_stale_decisions_or_corpus_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = prepared(Path(directory))
            decision = admission(request)
            decision['content_digest'] = 'stale'
            write(run / 'decisions.yaml', document(request, decision))
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                report_run(wiki, run, Path(directory) / 'lists')
            with self.assertRaisesRegex(ValueError, 'outside corpus'):
                report_run(wiki, run, wiki / 'data/lists')

    def test_members_count_once_and_retain_provenance(self):
        from test_repo_import_selection import SelectionTests
        from repo_import.selection import select_run
        from repo_import_support import import_repo
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionTests()
            wiki, run = fixture.prepare(Path(directory))
            write(run / 'selection.yaml', fixture.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            write(run / 'decisions.yaml', fixture.reviewed(request))
            out = Path(directory) / 'lists'
            summary = report_run(wiki, run, out)
            self.assertEqual(1, summary['effective_kernels'])
            self.assertEqual(2, summary['included_definitions'])
            self.assertEqual((1, 1, 1), (summary['helpers'], summary['representatives'], summary['members']))
            group = read(out / 'valid-kernels.yaml')['implementations'][0]
            self.assertEqual('second', group['members'][0]['entry'])

    def test_unresolved_include_does_not_appear_as_effective(self):
        from repo_import.publish import apply_plan
        from repo_import_support import import_repo, load_config, digest
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, first_run, request = prepared(Path(directory))
            write(first_run / 'decisions.yaml', document(request))
            plan = import_repo.build(wiki, first_run)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, first_run, plan)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            write(run / 'decisions.yaml', document(request))
            summary = report_run(wiki, run, Path(directory) / 'lists')
            self.assertEqual(1, summary['reviewed_include'])
            self.assertEqual(1, summary['blocked_include'])
            self.assertEqual(0, summary['effective_kernels'])

    def test_retained_existing_implementation_counts_without_new_page(self):
        from repo_import.publish import apply_plan
        from repo_import_support import import_repo, load_config, digest, resolve_review
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, first_run, request = prepared(Path(directory))
            write(first_run / 'decisions.yaml', document(request))
            plan = import_repo.build(wiki, first_run)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, first_run, plan)
            identity, implementation = next(iter(read(wiki / 'data/repo-import/registry.yaml')['implementations'].items()))
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            verdict = admission(request)
            verdict.update(action='record_duplicate', relation='exact_duplicate', target_implementation_id=identity,
                           target_page_id=implementation['page_id'], target_version_id=implementation['current_version'])
            resolve_review(request, verdict, 'same_implementation', implementation['page_id'])
            write(run / 'decisions.yaml', document(request, verdict))
            out = Path(directory) / 'lists'
            summary = report_run(wiki, run, out)
            self.assertEqual(1, summary['effective_kernels'])
            self.assertEqual(identity, read(out / 'valid-kernels.yaml')['implementations'][0]['implementation_id'])
