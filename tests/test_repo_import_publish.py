import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from repo_import_support import prepared, document, write, import_repo, read
from repo_import.store import digest, file_hash
from repo_import.publish import apply_plan, recover_transaction


class PublishTests(unittest.TestCase):
    def test_formal_manifest_does_not_expand_with_unpublished_selection(self):
        from repo_import.publish import compact_manifest
        from repo_import_support import admission
        with tempfile.TemporaryDirectory() as directory:
            _, _, run, request = prepared(Path(directory))
            decision = admission(request)
            selected = [(request['candidates'][0], decision)]
            request['selection'] = {'entries': []}
            before = compact_manifest(run, request, document(request), selected)
            request['selection']['entries'] = [
                {'candidate_id': 'helper-' + str(i), 'role': 'helper', 'rationale': 'x' * 1000}
                for i in range(1000)]
            after = compact_manifest(run, request, document(request), selected)
            self.assertEqual(len(str(before)), len(str(after)))
            self.assertNotIn('selection', after)
            self.assertNotIn('inventory', after)
            self.assertNotIn('helper-999', str(after))
            self.assertNotEqual(before['run_reference']['selection_digest'], after['run_reference']['selection_digest'])

    def test_member_paths_render_without_duplicating_full_versions(self):
        from test_repo_import_selection import SelectionTests
        from repo_import.selection import select_run
        with tempfile.TemporaryDirectory() as directory:
            fixture = SelectionTests()
            wiki, run = fixture.prepare(Path(directory))
            write(run / 'selection.yaml', fixture.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            write(run / 'decisions.yaml', fixture.reviewed(request))
            plan = import_repo.build(wiki, run)
            page = next(record['path'] for record in plan['files'] if record['path'].startswith('wiki/kernels/'))
            text = (run / 'staging' / page).read_text()
            self.assertIn('Related source entries', text)
            self.assertIn('| kernel.py | second | variant |', text)
            self.assertIn('do not imply that the representative entry supports every variant', text)
            implementation = next(iter(read(run / 'staging/data/repo-import/registry.yaml')['implementations'].values()))
            self.assertEqual(1, len(implementation['versions']))
            self.assertNotIn('candidate', implementation['records'][0])
            self.assertTrue(implementation['records'][0]['evidence'])
            formal = read(run / 'staging' / next(record['path'] for record in plan['files']
                                               if record['path'].startswith('data/repo-import/decisions/')))
            self.assertNotIn('selection', formal)
            self.assertEqual(1, len(formal['members']))

    def plan(self, directory):
        wiki, _, run, request = prepared(directory)
        write(run / 'decisions.yaml', document(request))
        plan = import_repo.build(wiki, run)
        plan['validated_digest'] = digest(plan['files'])
        return wiki, run, plan

    def test_concurrent_target_edit_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, plan = self.plan(Path(directory))
            target = wiki / plan['files'][0]['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('concurrent edit')
            with self.assertRaisesRegex(ValueError, 'Concurrent'):
                apply_plan(wiki, run, plan)
            self.assertEqual('concurrent edit', target.read_text())

    def test_partial_write_failure_restores_all_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, plan = self.plan(Path(directory))
            from repo_import import publish
            original = publish.atomic_write
            calls = 0
            def fail_once(path, text):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('injected write failure')
                original(path, text)
            with patch.object(publish, 'atomic_write', side_effect=fail_once), self.assertRaises(OSError):
                apply_plan(wiki, run, plan)
            for record in plan['files']:
                self.assertEqual(record['before'], file_hash(wiki / record['path']))
            self.assertEqual('rolled_back', read(run / 'transaction.yaml')['status'])

    def test_interrupted_apply_recovers_before_context_check(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, plan = self.plan(Path(directory))
            write(run / 'publication-plan.yaml', plan)
            from repo_import import publish
            original = publish.atomic_write
            calls = 0
            def crash(path, text):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise SystemExit('simulated process interruption')
                original(path, text)
            with patch.object(publish, 'atomic_write', side_effect=crash), self.assertRaises(SystemExit):
                apply_plan(wiki, run, plan)
            self.assertEqual('applying', read(run / 'transaction.yaml')['status'])
            import_repo.apply(wiki, run)
            self.assertEqual('committed', read(run / 'transaction.yaml')['status'])

    def test_existing_page_duplicate_adds_source_without_replacing_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            from repo_import.publish import markdown
            from repo_import_support import admission, load_config
            text = markdown({'id': 'kernel-existing', 'title': 'Existing kernel', 'type': 'kernel',
                             'sources': [], 'reproducibility': 'benchmarked'}, '# Existing page content\n')
            page = wiki / 'wiki/kernels/existing.md'
            page.write_text(text)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            decision = admission(request)
            decision.update(action='attach_source', relation='exact_duplicate', target_page_id='kernel-existing',
                            equivalence_basis='Fixture source corresponds to this existing kernel example.',
                            target_evidence=[{'path': 'wiki/kernels/existing.md', 'sha256': file_hash(page), 'locator': 'Existing page content'}])
            write(run / 'decisions.yaml', document(request, decision))
            plan = import_repo.build(wiki, run)
            staged = (run / 'staging/wiki/kernels/existing.md').read_text()
            self.assertIn('# Existing page content', staged)
            self.assertIn('reproducibility: benchmarked', staged)
            self.assertNotIn('Fixture reduction', staged)
            self.assertIn('Related source entries', staged)
            self.assertIn('| kernel.py | reduce_kernel | exact_duplicate |', staged)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, run, plan)
            from repo_import.publish import registry_errors
            self.assertEqual([], registry_errors(wiki))

    def test_changed_decision_after_build_cannot_be_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, plan = self.plan(Path(directory))
            write(run / 'publication-plan.yaml', plan)
            doc = read(run / 'decisions.yaml')
            doc['decisions'][0]['rationale'] = 'changed after build'
            write(run / 'decisions.yaml', doc)
            with self.assertRaisesRegex(ValueError, 'changed after build'):
                import_repo.apply(wiki, run)
