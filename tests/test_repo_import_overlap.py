"""Unresolved overlap blocks only related candidates; one representative is published."""
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from repo_import_support import (prepared, admission, document, commit, import_repo, load_config,
                                 read, write, resolve_review, make_repo, make_wiki, digest)
from repo_import.publish import apply_plan, markdown, registry_errors
from repo_import.store import file_hash


class OverlapTests(unittest.TestCase):
    def publish_unit(self, wiki, run, request, decisions):
        write(run / 'decisions.yaml', {'version': 1, 'context_digest': request['context_digest'], 'decisions': decisions})
        plan = import_repo.build(wiki, run)
        plan['validated_digest'] = digest(plan['files'])
        apply_plan(wiki, run, plan)
        return read(wiki / 'data/repo-import/registry.yaml')

    def batch(self, base):
        wiki, repo, _, _ = prepared(base)
        (repo / 'second.py').write_text((repo / 'kernel.py').read_text())
        (repo / 'unrelated.py').write_text('import triton\n@triton.jit\ndef other_operation(x):\n    return x - 37\n')
        commit(repo, 'batch fixture')
        run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
        return wiki, repo, run, read(run / 'review-request.yaml')

    def test_missing_overlap_blocks_only_related_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = self.batch(Path(directory))
            decisions = [admission(request, i) for i in range(3)]
            self.publish_unit(wiki, run, request, decisions)
            result = read(run / 'result.yaml')
            self.assertEqual(1, result['included'])
            self.assertEqual(2, len(result['pending']))
            self.assertEqual(1, len(list((wiki / 'wiki/kernels/imported').glob('*.md'))))

    def test_batch_same_implementation_publishes_one_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = self.batch(Path(directory))
            chosen, other = admission(request, 0), admission(request, 1)
            resolve_review(request, chosen, 'same_implementation', chosen['candidate_id'])
            other.update(action='merge_candidate', relation='variant', target_candidate_id=chosen['candidate_id'],
                         selection_reason='Use the first reviewed entry as the representative.')
            resolve_review(request, other, 'same_implementation', chosen['candidate_id'])
            inconsistent = copy.deepcopy([other, chosen])
            inconsistent[0]['overlap_resolutions'][0]['outcome'] = 'distinct'
            write(run / 'decisions.yaml', {'version': 1, 'context_digest': request['context_digest'], 'decisions': inconsistent})
            import_repo.build(wiki, run)
            self.assertEqual(0, read(run / 'result.yaml')['included'])
            data = self.publish_unit(wiki, run, request, [other, chosen])
            self.assertEqual(1, len(data['implementations']))
            item = next(iter(data['implementations'].values()))
            self.assertEqual(1, len(item['versions']))
            self.assertEqual(1, len(item['records']))
            self.assertNotIn('candidate', item['records'][0])
            self.assertEqual([], registry_errors(wiki))

    def representative_family(self, base):
        wiki, repo, _, _ = self.batch(base)
        for filename in ('third.py', 'fourth.py'):
            (repo / filename).write_text((repo / 'kernel.py').read_text())
        commit(repo, 'four related implementations')
        run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
        request = read(run / 'review-request.yaml')
        chosen = next(c['candidate_id'] for c in request['candidates'] if c['path'] == 'kernel.py')
        decisions = []
        for index, candidate in enumerate(request['candidates']):
            decision = admission(request, index)
            if candidate['path'] == 'unrelated.py':
                resolve_review(request, decision)
            else:
                if candidate['candidate_id'] != chosen:
                    decision.update(action='merge_candidate', relation='variant', target_candidate_id=chosen,
                                    selection_reason='Keep one common representative for the reviewed family.')
                resolve_review(request, decision, 'same_implementation', chosen)
            decisions.append(decision)
        return wiki, run, request, decisions, chosen

    def test_multi_candidate_family_publishes_one_common_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, request, decisions, chosen = self.representative_family(Path(directory))
            data = self.publish_unit(wiki, run, request, list(reversed(decisions)))
            self.assertEqual(5, read(run / 'result.yaml')['included'])
            self.assertEqual([], read(run / 'result.yaml')['pending'])
            self.assertEqual(2, len(data['implementations']))
            family = next(item for item in data['implementations'].values()
                          if item['versions'][item['current_version']]['candidate']['candidate_id'] == chosen)
            self.assertEqual(1, len(family['versions']))
            self.assertEqual(3, len(family['records']))
            self.assertEqual(2, len(list((wiki / 'wiki/kernels/imported').glob('*.md'))))
            self.assertEqual([], registry_errors(wiki))

    def test_common_representative_must_be_included_and_unblocked(self):
        from repo_import.decisions import validate_decisions
        for missing in (False, True):
            with self.subTest(representative_deferred=missing), tempfile.TemporaryDirectory() as directory:
                wiki, _, request, decisions, chosen = self.representative_family(Path(directory))
                canonical = next(d for d in decisions if d['candidate_id'] == chosen)
                if missing:
                    canonical['decision'] = 'defer'
                else:
                    canonical['overlap_resolutions'] = []
                selected, pending = validate_decisions(
                    {'version': 1, 'context_digest': request['context_digest'], 'decisions': decisions},
                    request, wiki, Path(request['snapshot']['checkout']))
                self.assertEqual(1, len(selected))
                self.assertEqual('unrelated.py', selected[0][0]['path'])
                self.assertEqual(4, len(pending))

    def test_family_reciprocal_comparisons_cannot_choose_conflicting_representatives(self):
        from repo_import.decisions import validate_decisions
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, request, decisions, chosen = self.representative_family(Path(directory))
            variants = [d for d in decisions if d.get('action') == 'merge_candidate']
            left, right = variants[:2]
            match_id = next(h['match_id'] for h in request['matches'][left['candidate_id']]
                            if h['candidate_id'] == right['candidate_id'])
            next(r for r in left['overlap_resolutions'] if r['match_id'] == match_id)['representative'] = 'missing'
            selected, pending = validate_decisions(
                {'version': 1, 'context_digest': request['context_digest'], 'decisions': decisions},
                request, wiki, Path(request['snapshot']['checkout']))
            self.assertIn(left['candidate_id'], pending)
            self.assertIn(right['candidate_id'], pending)
            self.assertEqual(3, len(selected))

    def test_distinct_batch_requires_comparison_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = self.batch(Path(directory))
            decisions = [resolve_review(request, admission(request, i)) for i in range(2)]
            decisions[0]['overlap_resolutions'][0]['evidence'][0]['sha256'] = 'tampered'
            write(run / 'decisions.yaml', {'version': 1, 'context_digest': request['context_digest'], 'decisions': decisions})
            with self.assertRaisesRegex(ValueError, 'matched evidence'):
                import_repo.build(wiki, run)

    def test_fresh_import_reuses_unchanged_distinct_batch_comparison(self):
        from repo_import.decisions import rebind_manifest
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            source = repo / 'kernel.py'
            source.write_text(source.read_text() + '\n@triton.jit\ndef reduce_kernel(x, n: int):\n    return x * n\n')
            commit(repo, 'distinct entries in one file')
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            decision = resolve_review(request, admission(request))
            original = document(request, copy.deepcopy(decision))
            self.publish_unit(wiki, run, request, [decision])
            repo = Path(directory) / 'input'
            repeated = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            repeated_request = read(repeated / 'review-request.yaml')
            import_repo.build(wiki, repeated)
            self.assertEqual(1, read(repeated / 'result.yaml')['included'])
            rebound = rebind_manifest(original, repeated_request)
            write(repeated / 'decisions.yaml', rebound)
            import_repo.build(wiki, repeated)
            self.assertEqual(1, read(repeated / 'result.yaml')['included'])
            self.assertEqual(1, len(read(repeated / 'result.yaml')['pending']))
            source.write_text(source.read_text().replace('x * n', 'x - n'))
            commit(repo, 'change comparison evidence')
            changed = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            import_repo.build(wiki, changed)
            self.assertEqual(0, read(changed / 'result.yaml')['included'])

    def test_matches_cannot_be_removed_from_review_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, run, request = self.batch(Path(directory))
            request['matches'] = {candidate['candidate_id']: [] for candidate in request['candidates']}
            write(run / 'review-request.yaml', request)
            write(run / 'decisions.yaml', document(request))
            with self.assertRaisesRegex(ValueError, 'matching materials'):
                import_repo.build(wiki, run)

    def pr_fixture(self, wiki, repo):
        page = wiki / 'sources/prs/fixture/PR-1.md'
        page.parent.mkdir(parents=True)
        page.write_text(markdown({'id': 'pr-fixture-1', 'repo': 'owner/fixture', 'title': 'Existing reduction PR'}, '# PR evidence\n'))
        asset = wiki / 'artifacts/prs/fixture/PR-1/key-files/kernel.py'
        asset.parent.mkdir(parents=True)
        asset.write_text((repo / 'kernel.py').read_text())
        return page, asset

    def test_pr_only_overlap_requires_resolution_and_links_source(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            self.pr_fixture(wiki, repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            self.assertEqual('pr', request['matches'][request['candidates'][0]['candidate_id']][0]['kind'])
            from repo_import.decisions import validate_decisions
            selected, pending = validate_decisions(document(request), request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], selected)
            self.assertEqual(1, len(pending))
            decision = resolve_review(request, admission(request), 'same_implementation')
            data = self.publish_unit(wiki, run, request, [decision])
            page = wiki / next(iter(data['implementations'].values()))['page_path']
            self.assertIn('pr-fixture-1', page.read_text())

    def test_pr_linked_kernel_cannot_be_bypassed_with_create_page(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            self.pr_fixture(wiki, repo)
            (wiki / 'wiki/kernels/existing.md').write_text(markdown(
                {'id': 'kernel-existing', 'title': 'Existing', 'sources': ['pr-fixture-1'], 'reproducibility': 'snippet'}, '# Existing\n'))
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            decision = resolve_review(request, admission(request), 'same_implementation')
            write(run / 'decisions.yaml', document(request, decision))
            import_repo.build(wiki, run)
            self.assertEqual(0, read(run / 'result.yaml')['included'])
            decision.update(action='attach_source', relation='exact_duplicate', target_page_id='kernel-existing',
                            equivalence_basis='Both sources identify this reviewed fixture.',
                            target_evidence=[{'path': 'wiki/kernels/existing.md', 'sha256': file_hash(wiki / 'wiki/kernels/existing.md'), 'locator': 'Existing'}])
            resolve_review(request, decision, 'same_implementation', 'kernel-existing')
            self.publish_unit(wiki, run, request, [decision])
            self.assertFalse((wiki / 'wiki/kernels/imported').exists())

    def test_multi_implementation_pr_requires_comparison_with_every_linked_page(self):
        from repo_import.decisions import validate_decisions
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            self.pr_fixture(wiki, repo)
            for suffix in ('one', 'two'):
                (wiki / ('wiki/kernels/' + suffix + '.md')).write_text(markdown(
                    {'id': 'kernel-' + suffix, 'title': 'Separate implementation ' + suffix,
                     'sources': ['pr-fixture-1'], 'reproducibility': 'snippet'}, '# Different source contract\n'))
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            decision = resolve_review(request, admission(request), 'distinct')
            hints = request['matches'][decision['candidate_id']]
            self.assertEqual({'kernel-one', 'kernel-two'}, {h['page_id'] for h in hints if h['kind'] == 'kernel'})
            pr = next(h for h in hints if h['kind'] == 'pr')
            next(r for r in decision['overlap_resolutions'] if r['match_id'] == pr['match_id']).update(
                outcome='same_implementation', representative=decision['candidate_id'])
            selected, pending = validate_decisions(document(request, decision), request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual((1, []), (len(selected), pending))
            saved = copy.deepcopy(decision)
            decision['overlap_resolutions'].pop()
            selected, pending = validate_decisions(document(request, decision), request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual(0, len(selected))
            self.assertEqual(1, len(pending))
            decision = copy.deepcopy(saved)
            canonical = next(r for r in decision['overlap_resolutions'] if r['match_id'] != pr['match_id'])
            canonical['outcome'] = 'same_implementation'
            selected, pending = validate_decisions(document(request, decision), request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual(0, len(selected))
            decision = copy.deepcopy(saved)
            decision['overlap_resolutions'][-1]['evidence'][0]['sha256'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'matched evidence'):
                validate_decisions(document(request, decision), request, wiki, Path(request['snapshot']['checkout']))
            write(run / 'decisions.yaml', document(request, saved))
            plan = import_repo.build(wiki, run)
            page = next(record['path'] for record in plan['files'] if record['path'].startswith('wiki/kernels/imported/'))
            self.assertIn('pr-fixture-1', (run / 'staging' / page).read_text())
            self.assertNotIn('wiki/kernels/one.md', [record['path'] for record in plan['files']])

    def test_existing_unlinked_target_can_receive_reviewed_multi_branch_pr_attribution(self):
        from repo_import.decisions import validate_decisions
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            self.pr_fixture(wiki, repo)
            target = wiki / 'wiki/kernels/target.md'
            target.write_text(markdown(
                {'id': 'kernel-target', 'title': 'Known implementation', 'repo': repo.as_uri(),
                 'changed_paths': ['kernel.py'], 'sources': [], 'reproducibility': 'snippet'},
                '# Existing implementation, not previously attributed to this PR\n'))
            other = wiki / 'wiki/kernels/other.md'
            other.write_text(markdown(
                {'id': 'kernel-other', 'title': 'Other PR branch',
                 'sources': ['pr-fixture-1'], 'reproducibility': 'snippet'}, '# Different operation\n'))
            other_before = other.read_bytes()
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            decision = resolve_review(request, admission(request), 'distinct')
            decision.update(action='attach_source', relation='exact_duplicate', target_page_id='kernel-target',
                            equivalence_basis='Source comparison identifies the existing target, not the other PR branch.',
                            target_evidence=[{'path': 'wiki/kernels/target.md', 'sha256': file_hash(target),
                                              'locator': 'Existing implementation'}])
            hints = {h['match_id']: h for h in request['matches'][decision['candidate_id']]}
            for resolution in decision['overlap_resolutions']:
                hint = hints[resolution['match_id']]
                if hint['kind'] == 'pr' or hint.get('page_id') == 'kernel-target':
                    resolution.update(outcome='same_implementation', representative='kernel-target')
            selected, pending = validate_decisions(document(request, decision), request, wiki,
                                                  Path(request['snapshot']['checkout']))
            self.assertEqual((1, []), (len(selected), pending))
            for omitted in ('kernel-target', 'kernel-other'):
                with self.subTest(omitted=omitted):
                    changed = copy.deepcopy(decision)
                    changed['overlap_resolutions'] = [r for r in changed['overlap_resolutions']
                        if hints[r['match_id']].get('page_id') != omitted]
                    selected, pending = validate_decisions(document(request, changed), request, wiki,
                                                          Path(request['snapshot']['checkout']))
                    self.assertEqual((0, 1), (len(selected), len(pending)))
            changed = copy.deepcopy(decision)
            next(r for r in changed['overlap_resolutions']
                 if hints[r['match_id']].get('page_id') == 'kernel-other').update(
                     outcome='same_implementation', representative='kernel-target')
            selected, pending = validate_decisions(document(request, changed), request, wiki,
                                                  Path(request['snapshot']['checkout']))
            self.assertEqual((0, 1), (len(selected), len(pending)))
            self.publish_unit(wiki, run, request, [decision])
            self.assertIn('pr-fixture-1', target.read_text())
            self.assertEqual(other_before, other.read_bytes())
            self.assertFalse((wiki / 'wiki/kernels/imported').exists())

    def test_changed_pr_asset_invalidates_context(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            _, asset = self.pr_fixture(wiki, repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            asset.write_text('# replaced evidence\n')
            with self.assertRaisesRegex(ValueError, 'prepare again'):
                import_repo.build(wiki, run)

    def test_existing_page_can_remain_representative_for_a_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            page = wiki / 'wiki/kernels/existing.md'
            page.write_text(markdown({'id': 'kernel-existing', 'title': 'Existing reduction', 'type': 'kernel',
                                      'sources': [], 'confidence': 'verified', 'reproducibility': 'benchmarked'}, '# Original representative\n'))
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            decision = admission(request)
            decision.update(action='retain_page', relation='variant', target_page_id='kernel-existing',
                            equivalence_basis='Fixture compares the incoming variant with the existing representative.',
                            selection_reason='Keep the existing example and its applicable evidence.',
                            target_evidence=[{'path': 'wiki/kernels/existing.md', 'sha256': file_hash(page), 'locator': 'Original representative'}])
            resolve_review(request, decision, 'same_implementation', 'kernel-existing')
            data = self.publish_unit(wiki, run, request, [decision])
            item = next(iter(data['implementations'].values()))
            selected_version = item['versions'][item['current_version']]
            self.assertNotIn('candidate', selected_version)
            self.assertIn('page_evidence', selected_version)
            self.assertEqual('benchmarked', selected_version['metadata']['reproducibility'])
            self.assertNotIn('Implementation', page.read_text())
            self.assertEqual([], registry_errors(wiki))
            repeated = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            self.assertEqual('retain_page', read(repeated / 'decisions.yaml')['decisions'][0]['action'])

    def test_variant_can_retain_existing_page_without_inheriting_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, run, request = prepared(Path(directory))
            data = self.publish_unit(wiki, run, request, [admission(request)])
            identity, item = next(iter(data['implementations'].items()))
            before = (wiki / item['page_path']).read_text()
            (repo / 'kernel.py').write_text((repo / 'kernel.py').read_text().replace('x + n', 'x * n'))
            commit(repo, 'variant')
            new = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            request = read(new / 'review-request.yaml')
            decision = admission(request)
            decision.update(action='retain_page', relation='variant', target_implementation_id=identity,
                            target_page_id=item['page_id'], target_version_id=item['current_version'],
                            selection_reason='Keep the existing representative with its own evidence.')
            resolve_review(request, decision, 'same_implementation', item['page_id'])
            data = self.publish_unit(wiki, new, request, [decision])
            self.assertEqual(1, len(data['implementations'][identity]['versions']))
            self.assertIn('return x + n', (wiki / item['page_path']).read_text())
            self.assertNotIn('return x * n', (wiki / item['page_path']).read_text())
            self.assertEqual([], registry_errors(wiki))
            repeated = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            self.assertEqual('retain_page', read(repeated / 'decisions.yaml')['decisions'][0]['action'])
            repeated_request = read(repeated / 'review-request.yaml')
            repeated_decision = read(repeated / 'decisions.yaml')['decisions'][0]
            repeated_data = self.publish_unit(wiki, repeated, repeated_request, [repeated_decision])
            self.assertEqual(1, len(repeated_data['implementations'][identity]['records']))


class InputScopeTests(unittest.TestCase):
    def test_submodule_and_its_code_are_never_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            child_base, parent_base = base / 'child', base / 'parent'
            child_base.mkdir()
            parent_base.mkdir()
            child = make_repo(child_base)
            parent = make_repo(parent_base)
            subprocess.run(['git', '-C', str(parent), '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(child), 'child'], check=True)
            commit(parent, 'gitlink')
            wiki = make_wiki(base)
            run = import_repo.prepare(wiki, str(parent), 'HEAD', load_config(wiki))
            request = read(run / 'review-request.yaml')
            self.assertEqual(1, len(request['candidates']))
            self.assertEqual('child', request['snapshot']['skipped_submodules'][0]['path'])
            self.assertFalse((Path(request['snapshot']['checkout']) / 'child/kernel.py').exists())

    def test_non_git_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            folder = base / 'folder'
            folder.mkdir()
            wiki = make_wiki(base)
            with self.assertRaises(subprocess.CalledProcessError):
                import_repo.prepare(wiki, str(folder), None, load_config(wiki))
