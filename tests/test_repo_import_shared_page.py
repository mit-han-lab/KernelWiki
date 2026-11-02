"""Existing multi-branch pages retain one identity and explicit source scopes."""
import copy
import tempfile
import unittest
from pathlib import Path

from repo_import_support import (prepared, admission, commit, import_repo, load_config,
                                 read, write, resolve_review, digest)
from repo_import.decisions import validate_decisions, rebind_manifest
from repo_import.publish import apply_plan, markdown, registry_errors
from repo_import.report import report_run
from repo_import.store import file_hash


class SharedExistingPageTests(unittest.TestCase):
    def fixture(self, base):
        wiki, repo, _, _ = prepared(base)
        (repo / 'second.py').write_text((repo / 'kernel.py').read_text())
        commit(repo, 'two branch fixture')
        page = wiki / 'wiki/kernels/existing.md'
        page.write_text(markdown({'id': 'kernel-existing', 'title': 'Existing reduction', 'type': 'kernel',
                                  'sources': [], 'confidence': 'verified', 'reproducibility': 'benchmarked'},
                                 '# Existing reduction\n\nDenseScheduleBranch\n\nPointerArrayScheduleBranch\n'))
        run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
        request = read(run / 'review-request.yaml')
        decisions = []
        for index, anchor in enumerate(('DenseScheduleBranch', 'PointerArrayScheduleBranch')):
            decision = admission(request, index)
            decision.update(action='retain_page', relation='variant', target_page_id='kernel-existing',
                            equivalence_basis='Fixture binds each source to a branch explicitly covered by the existing page.',
                            selection_reason='Preserve the existing page and separately reviewed branch attribution.',
                            target_evidence=[{'path': 'wiki/kernels/existing.md', 'sha256': file_hash(page),
                                              'locator': anchor, 'anchor': anchor}])
            resolve_review(request, decision)
            for resolution, hint in zip(decision['overlap_resolutions'], request['matches'][decision['candidate_id']]):
                if hint['kind'] != 'batch':
                    resolution.update(outcome='same_implementation', representative='kernel-existing')
            decisions.append(decision)
        return wiki, repo, run, request, decisions, page

    def document(self, request, decisions):
        return {'version': 1, 'context_digest': request['context_digest'], 'decisions': decisions}

    def check(self, wiki, request, decisions):
        return validate_decisions(self.document(request, decisions), request, wiki, Path(request['snapshot']['checkout']))

    def test_distinct_scoped_branches_share_page_identity_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, run, request, decisions, page = self.fixture(Path(directory))
            original = page.read_text()
            document = self.document(request, decisions)
            write(run / 'decisions.yaml', document)
            summary = report_run(wiki, run, run / 'report')
            self.assertEqual(1, summary['effective_kernels'])
            self.assertEqual(2, summary['included_definitions'])
            plan = import_repo.build(wiki, run)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, run, plan)
            registry = read(wiki / 'data/repo-import/registry.yaml')
            self.assertEqual(1, len(registry['implementations']))
            item = next(iter(registry['implementations'].values()))
            self.assertEqual(2, len(item['origins']))
            self.assertEqual(2, len(item['records']))
            self.assertEqual(1, len(item['versions']))
            self.assertFalse(item['managed'])
            self.assertEqual({'DenseScheduleBranch', 'PointerArrayScheduleBranch'},
                             {r['target_evidence'][0]['anchor'] for r in item['records']})
            self.assertIn(original.split('---', 2)[2].strip(), page.read_text())
            self.assertEqual([], registry_errors(wiki))
            repeated = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            new_request = read(repeated / 'review-request.yaml')
            replay = rebind_manifest(document, new_request)
            self.assertEqual(2, len(replay['decisions']))
            self.assertEqual([], self.check(wiki, new_request, replay['decisions'])[1])
            self.assertEqual({'DenseScheduleBranch', 'PointerArrayScheduleBranch'},
                             {d['target_evidence'][0]['anchor'] for d in replay['decisions']})
            for decision in replay['decisions']:
                self.assertEqual(file_hash(page), decision['target_evidence'][0]['sha256'])

    def test_unscoped_or_shared_anchor_blocks_every_branch(self):
        for mode in ('missing', 'same'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                wiki, _, _, request, decisions, _ = self.fixture(Path(directory))
                if mode == 'missing':
                    decisions[1]['target_evidence'][0].pop('anchor')
                else:
                    decisions[1]['target_evidence'][0]['anchor'] = 'DenseScheduleBranch'
                selected, pending = self.check(wiki, request, decisions)
                self.assertEqual([], selected)
                self.assertEqual(2, len(pending))

    def test_changed_existing_page_invalidates_saved_attribution_even_if_anchors_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, run, request, decisions, page = self.fixture(Path(directory))
            document = self.document(request, decisions)
            write(run / 'decisions.yaml', document)
            plan = import_repo.build(wiki, run)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, run, plan)
            page.write_text(page.read_text().replace('DenseScheduleBranch',
                                                    'DenseScheduleBranch with changed canonical computation'))
            repeated = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            new_request = read(repeated / 'review-request.yaml')
            replay = rebind_manifest(document, new_request)
            self.assertEqual([], replay['decisions'])
            self.assertEqual([], self.check(wiki, new_request, replay['decisions'])[0])
            self.assertTrue(all(d['decision'] == 'defer' for d in read(repeated / 'decisions.yaml')['decisions']))

    def test_forged_or_stale_anchor_receipt_rejected(self):
        for mode in ('forged', 'empty', 'stale'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                wiki, _, _, request, decisions, _ = self.fixture(Path(directory))
                evidence = decisions[1]['target_evidence'][0]
                evidence['sha256' if mode == 'stale' else 'anchor'] = '' if mode == 'empty' else 'absent'
                with self.assertRaises(ValueError):
                    self.check(wiki, request, decisions)

    def test_distinct_reciprocal_comparisons_are_required(self):
        for mode in ('missing_resolution', 'no_hint', 'conflict', 'attach_source'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                wiki, _, _, request, decisions, _ = self.fixture(Path(directory))
                identity = decisions[1]['candidate_id']
                batch_ids = {h['match_id'] for h in request['matches'][identity] if h['kind'] == 'batch'}
                self.assertTrue(batch_ids)
                if mode in {'missing_resolution', 'no_hint'}:
                    decisions[1]['overlap_resolutions'] = [r for r in decisions[1]['overlap_resolutions'] if r['match_id'] not in batch_ids]
                    if mode == 'no_hint':
                        request = copy.deepcopy(request)
                        request['matches'][identity] = [h for h in request['matches'][identity] if h['kind'] != 'batch']
                elif mode == 'conflict':
                    for resolution in decisions[1]['overlap_resolutions']:
                        if resolution['match_id'] in batch_ids:
                            resolution.update(outcome='same_implementation', representative=decisions[0]['candidate_id'])
                else:
                    decisions[1].update(action='attach_source', relation='exact_duplicate')
                selected, pending = self.check(wiki, request, decisions)
                self.assertEqual([], selected)
                self.assertEqual(2, len(pending))
