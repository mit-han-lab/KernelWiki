import tempfile
import unittest
from pathlib import Path
from repo_import_support import prepared, commit, import_repo, load_config, read, registered, write, digest, document
from repo_import.publish import build_plan, registry_errors


class VersionTests(unittest.TestCase):
    def test_signature_and_helper_changes_invalidate_content(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, old = prepared(Path(directory))
            (repo / 'kernel.py').write_text((repo / 'kernel.py').read_text().replace('n: int', 'n: float'))
            commit(repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            new = read(run / 'review-request.yaml')
            self.assertNotEqual(old['candidates'][0]['content_digest'], new['candidates'][0]['content_digest'])
            self.assertEqual('defer', read(run / 'decisions.yaml')['decisions'][0]['decision'])

    def test_same_file_helpers_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, old = prepared(Path(directory))
            with (repo / 'kernel.py').open('a') as stream:
                stream.write('\ndef helper(x):\n    return x + 1\n')
            commit(repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            new = read(run / 'review-request.yaml')
            self.assertEqual(old['candidates'][0]['fingerprints']['entry_sha256'], new['candidates'][0]['fingerprints']['entry_sha256'])
            self.assertNotEqual(old['candidates'][0]['content_digest'], new['candidates'][0]['content_digest'])

    def test_imported_helper_changes_invalidate_unchanged_body(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            (repo / 'helper.py').write_text('def helper(x):\n    return x + 1\n')
            with (repo / 'kernel.py').open('a') as stream:
                stream.write('\nfrom helper import helper\n')
            commit(repo)
            first_run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            first = read(first_run / 'review-request.yaml')['candidates'][0]
            (repo / 'helper.py').write_text('def helper(x):\n    return x + 2\n')
            commit(repo)
            second_run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            second = read(second_run / 'review-request.yaml')['candidates'][0]
            self.assertEqual(first['fingerprints']['entry_sha256'], second['fingerprints']['entry_sha256'])
            self.assertNotEqual(first['content_digest'], second['content_digest'])

    def test_version_cycle_is_rejected_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = prepared(Path(directory))
            write(run / 'decisions.yaml', document(request))
            plan = import_repo.build(wiki, run)
            from repo_import.publish import apply_plan
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, run, plan)
            data = read(wiki / 'data/repo-import/registry.yaml')
            implementation = next(iter(data['implementations'].values()))
            version_id = implementation['current_version']
            implementation['versions'][version_id]['relation'] = {'kind': 'new_version', 'target': version_id}
            write(wiki / 'data/repo-import/registry.yaml', data)
            self.assertTrue(any('cyclic' in error for error in registry_errors(wiki)))

    def test_new_version_keeps_compact_history_and_independent_reproducibility(self):
        from repo_import_support import admission
        from repo_import.publish import apply_plan
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, run, request = prepared(Path(directory))
            write(run / 'decisions.yaml', document(request))
            plan = import_repo.build(wiki, run)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, run, plan)
            old_registry = read(wiki / 'data/repo-import/registry.yaml')
            identity, implementation = next(iter(old_registry['implementations'].items()))
            old_id = implementation['current_version']
            (repo / 'kernel.py').write_text((repo / 'kernel.py').read_text().replace('x + n', 'x * n'))
            commit(repo)
            new_run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            new_request = read(new_run / 'review-request.yaml')
            decision = admission(new_request)
            decision.update(action='extend_page', relation='new_version', target_implementation_id=identity,
                            target_page_id=implementation['page_id'], target_version_id=old_id,
                            selection_reason='Choose the changed implementation for this fixture.')
            from repo_import_support import resolve_review
            resolve_review(new_request, decision, 'same_implementation')
            write(new_run / 'decisions.yaml', document(new_request, decision))
            import_repo.build(wiki, new_run)
            updated = read(new_run / 'staging/data/repo-import/registry.yaml')['implementations'][identity]
            self.assertEqual(1, len(updated['versions']))
            self.assertEqual(implementation['versions'][old_id]['candidate']['content_digest'], updated['history'][old_id]['content_digest'])
            self.assertNotIn('candidate', updated['history'][old_id])
            self.assertEqual(old_id, updated['versions'][updated['current_version']]['relation']['target'])
            self.assertEqual('snippet', updated['versions'][updated['current_version']]['metadata']['reproducibility'])
            new_plan = read(new_run / 'publication-plan.yaml')
            new_plan['validated_digest'] = digest(new_plan['files'])
            apply_plan(wiki, new_run, new_plan)
            (repo / 'kernel.py').write_text((repo / 'kernel.py').read_text().replace('x * n', 'x + n'))
            commit(repo, 'restore prior representative')
            restored_run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            restored_request = read(restored_run / 'review-request.yaml')
            restored = admission(restored_request)
            restored.update(action='extend_page', relation='new_version', target_implementation_id=identity,
                            target_page_id=implementation['page_id'], target_version_id=updated['current_version'],
                            selection_reason='Select the older code after comparing current evidence.')
            resolve_review(restored_request, restored, 'same_implementation')
            write(restored_run / 'decisions.yaml', document(restored_request, restored))
            restored_plan = import_repo.build(wiki, restored_run)
            restored_plan['validated_digest'] = digest(restored_plan['files'])
            apply_plan(wiki, restored_run, restored_plan)
            self.assertEqual([], registry_errors(wiki))
