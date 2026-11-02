import copy
import tempfile
import shutil
import unittest
from unittest.mock import patch
from pathlib import Path

from repo_import_support import (ROOT, make_repo, make_wiki, commit, read, write, digest,
                                 import_repo, admission, resolve_review)
from repo_import.config import load_config
from repo_import.selection import validate_selection, select_run, review_matches, reuse_unchanged_reviews
from repo_import.decisions import validate_decisions, rebind_manifest
from repo_import.scan import scan_snapshot

CODE = '''import triton
@triton.jit
def helper(x):
    return x + 1
@triton.jit
def first(x):
    return helper(x)
@triton.jit
def second(x):
    return helper(x)
def launch(x):
    first[(1,)](x)
    second[(1,)](x)
'''


class SelectionTests(unittest.TestCase):
    def prepare(self, base):
        repo = make_repo(base)
        (repo / 'kernel.py').write_text(CODE)
        commit(repo)
        wiki = make_wiki(base)
        config = load_config(wiki)
        config['selection']['enabled'] = True
        run = import_repo.prepare(wiki, str(repo), 'HEAD', config)
        return wiki, run

    def confirmed(self, run):
        selection = read(run / 'selection.yaml')
        inventory = read(run / 'inventory.yaml')
        candidates = {c['entry']: c for c in inventory['candidates']}
        rows = {r['entry']: r for r in selection['entries']}
        def receipt(c):
            return {'path': c['path'], 'sha256': c['fingerprints']['source_sha256'], 'locator': c['entry']}
        for name, row in rows.items():
            row.update(rationale='Source fixture launch and call relationships inspected.',
                       evidence=[receipt(candidates[name])])
        rows['helper'].update(role='helper', caller_basis='Called by first/second; both launched by launch; helper is device-only.')
        rows['first']['role'] = 'representative'
        rows['second'].update(role='member', representative=rows['first']['candidate_id'],
                              relation='variant', difference_summary='Same implementation, alternate entry name.')
        return selection

    def reviewed(self, request):
        index = next(i for i,c in enumerate(request['candidates']) if c['entry'] == 'first')
        verdict = resolve_review(request, admission(request, index))
        return {'version': 1, 'context_digest': request['context_digest'],
                'selection_digest': digest(request['selection']), 'decisions': [verdict]}

    def test_prepare_does_not_create_per_definition_review_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run = self.prepare(Path(directory))
            self.assertFalse((run / 'decisions.yaml').exists())
            self.assertFalse((run / 'matches.yaml').exists())
            selection = read(run / 'selection.yaml')
            self.assertTrue(all(r['role'] == 'pending' for r in selection['entries']))
            self.assertEqual(1, len(selection['suggestions']))
            rows = {r['entry']: r for r in selection['entries']}
            self.assertEqual('direct_call', rows['helper']['call_hints'][0]['kind'])
            self.assertEqual('grid_launch', rows['first']['call_hints'][0]['kind'])

    def test_confirmed_helper_and_variant_need_only_one_full_review(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            write(run / 'selection.yaml', self.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            self.assertEqual(2, len(request['candidates']))
            self.assertEqual(1, len(read(run / 'decisions.yaml')['decisions']))
            document = self.reviewed(request)
            selected, pending = validate_decisions(document, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], pending)
            self.assertEqual(2, len(selected))
            member = next(d for _,d in selected if d['action'] == 'merge_candidate')
            self.assertEqual('snippet', member['metadata']['reproducibility'])
            self.assertEqual('inferred', member['metadata']['confidence'])
            self.assertNotIn('performance_claims', member)
            self.assertNotIn('reproducibility_evidence', member)
            write(run / 'decisions.yaml', document)
            plan = import_repo.build(wiki, run)
            pages = [f for f in plan['files'] if f['path'].startswith('wiki/kernels/')]
            self.assertEqual(1, len(pages))
            self.assertEqual(1, len(read(run / 'staging/data/repo-import/registry.yaml')['implementations']))
            # Staging leaves the actual wiki untouched.
            self.assertFalse((wiki / 'data/repo-import/registry.yaml').exists())

    def test_unknown_relationships_stay_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            doc = read(run / 'decisions.yaml')
            selected, pending = validate_decisions(doc, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], selected)
            self.assertEqual(3, len(pending))

    def test_invalid_grouping_and_stale_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run = self.prepare(Path(directory))
            selection = self.confirmed(run)
            inventory = read(run / 'inventory.yaml')
            checkout = Path(read(run / 'snapshot.yaml')['checkout'])
            for change in ('cycle', 'missing', 'stale', 'no_difference', 'no_caller', 'bad_evidence'):
                with self.subTest(change=change):
                    altered = copy.deepcopy(selection)
                    rows = {r['entry']: r for r in altered['entries']}
                    if change == 'cycle': rows['second']['representative'] = rows['second']['candidate_id']
                    if change == 'missing': altered['entries'].pop()
                    if change == 'stale': altered['inventory_digest'] = 'old'
                    if change == 'no_difference': rows['second'].pop('difference_summary')
                    if change == 'no_caller': rows['helper'].pop('caller_basis')
                    if change == 'bad_evidence': rows['second']['evidence'][0]['sha256'] = 'bad'
                    with self.assertRaises(ValueError):
                        validate_selection(altered, inventory, checkout)

    def test_reviewer_can_split_group_and_previous_selection_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            selection = self.confirmed(run)
            write(run / 'selection.yaml', selection)
            select_run(wiki, run, import_repo.prepare_review)
            old_request = read(run / 'review-request.yaml')
            old_document = self.reviewed(old_request)
            write(run / 'decisions.yaml', old_document)
            rows = {r['entry']: r for r in selection['entries']}
            rows['second']['role'] = 'representative'
            rows['second'].pop('representative')
            write(run / 'selection.yaml', selection)
            with self.assertRaisesRegex(ValueError, 'Selection changed'):
                import_repo.build(wiki, run)
            select_run(wiki, run, import_repo.prepare_review)
            self.assertEqual(2, len(read(run / 'decisions.yaml')['decisions']))
            request = read(run / 'review-request.yaml')
            with self.assertRaisesRegex(ValueError, 'stale selection'):
                validate_decisions(old_document, request, wiki, Path(request['snapshot']['checkout']))
            self.assertTrue(list((run / 'review-attachments').glob('selection-*/decisions.yaml')))
            # Similar entries in separate groups still block unreviewed overlap.
            doc = self.reviewed(request)
            doc['decisions'][0].pop('overlap_resolutions')
            selected, pending = validate_decisions(doc, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], selected)
            self.assertEqual(2, len(pending))

    def test_excluded_group_resolves_members_without_admitting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            write(run / 'selection.yaml', self.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            document = self.reviewed(request)
            document['decisions'][0].update(decision='exclude', rationale='Whole fixture implementation is outside scope.')
            selected, pending = validate_decisions(document, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], selected)
            self.assertEqual([], pending)
            document['decisions'][0]['decision'] = 'defer'
            selected, pending = validate_decisions(document, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], selected)
            self.assertEqual(2, len(pending))

    def test_member_external_evidence_projects_to_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            write(run / 'selection.yaml', self.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            member = next(c for c in request['candidates'] if c['entry'] == 'second')
            # An asset matching only the member must still reach representative review.
            request['existing_pages'] = [{'page_id': 'kernel-existing', 'kind': 'kernel', 'path': 'wiki/kernels/existing.md',
                                         'sha256': 'page-hash', 'repo': '', 'title': 'Unrelated title', 'sources': [],
                                         'code_assets': [{'path': 'artifacts/old.cu', 'sha256': 'asset-hash',
                                                          'entries': [{'body_sha256': member['fingerprints']['entry_sha256']}]}]}]
            matches = review_matches(request)
            representative = next(r['candidate_id'] for r in request['selection']['entries'] if r['role'] == 'representative')
            self.assertTrue(any(h['page_id'] == 'kernel-existing' for h in matches[representative]))
            self.assertEqual([], matches[member['candidate_id']])

    def test_detailed_duplicate_refreshes_selection_and_preserves_unaffected_review(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            guide = wiki / 'references/repo-import-review.md'
            guide.parent.mkdir(parents=True, exist_ok=True)
            guide.write_text('Original review guidance.\n')
            selection = self.confirmed(run)
            rows = {r['entry']: r for r in selection['entries']}
            for row in rows.values():
                row['role'] = 'representative'
                row.pop('representative', None)
            write(run / 'selection.yaml', selection)
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            document = self.reviewed(request)
            support = next(c for c in request['candidates'] if c['entry'] == 'helper')
            exclusion = {
                'candidate_id': support['candidate_id'], 'content_digest': support['content_digest'],
                'decision': 'exclude', 'rationale': 'Device-only support called from the launched entries.',
                'evidence': rows['helper']['evidence']}
            document['decisions'].append(exclusion)
            write(run / 'decisions.yaml', document)
            previous_request = read(run / 'review-request.yaml')
            previous_targets = read(run / 'review-targets.yaml')
            previous_instructions = (run / 'review.md').read_text()
            guide.write_text('Updated review guidance.\n')

            # A new relationship belongs in selection, not a chain of full verdicts.
            invalid = copy.deepcopy(document)
            invalid['decisions'][0].update(action='merge_candidate', target_candidate_id=rows['second']['candidate_id'])
            with self.assertRaisesRegex(ValueError, 'revise selection'):
                validate_decisions(invalid, request, wiki, Path(request['snapshot']['checkout']))
            rows['second'].update(role='member', representative=rows['first']['candidate_id'])
            write(run / 'selection.yaml', selection)
            select_run(wiki, run, import_repo.prepare_review)
            archive = run / 'review-attachments' / ('selection-' + digest(previous_request)[:24])
            self.assertEqual('Original review guidance.\n', (archive / 'review-guide.md').read_text())
            self.assertEqual(previous_instructions, (archive / 'review.md').read_text())
            self.assertEqual(previous_targets, read(archive / 'review-targets.yaml'))
            self.assertEqual(document, read(archive / 'decisions.yaml'))
            self.assertEqual('Updated review guidance.\n', (run / 'review-guide.md').read_text())
            refreshed = read(run / 'decisions.yaml')
            self.assertEqual(exclusion, next(d for d in refreshed['decisions']
                                             if d['candidate_id'] == support['candidate_id']))
            self.assertEqual('defer', next(d for d in refreshed['decisions']
                                           if d['candidate_id'] == rows['first']['candidate_id'])['decision'])
            request = read(run / 'review-request.yaml')
            renewed = self.reviewed(request)
            renewed['decisions'].append(exclusion)
            selected, pending = validate_decisions(renewed, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], pending)
            self.assertEqual(2, len(selected))
            write(run / 'decisions.yaml', renewed)
            plan = import_repo.build(wiki, run)
            self.assertEqual(1, sum(f['path'].startswith('wiki/kernels/') for f in plan['files']))

    def test_unchanged_group_reuses_decision_and_members_never_inherit_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            write(run / 'selection.yaml', self.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            document = self.reviewed(request)
            write(run / 'decisions.yaml', document)
            select_run(wiki, run, import_repo.prepare_review)
            self.assertEqual('include', read(run / 'decisions.yaml')['decisions'][0]['decision'])
            altered = copy.deepcopy(document)
            altered['decisions'][0]['decision'] = 'defer'
            selected, pending = validate_decisions(altered, request, wiki, Path(request['snapshot']['checkout']))
            self.assertEqual([], selected)
            self.assertEqual(2, len(pending))
            # Independent member verdicts cannot silently override confirmed grouping.
            altered = copy.deepcopy(document)
            index = next(i for i,c in enumerate(request['candidates']) if c['entry'] == 'second')
            altered['decisions'].append(admission(request, index))
            with self.assertRaisesRegex(ValueError, 'split selection'):
                validate_decisions(altered, request, wiki, Path(request['snapshot']['checkout']))

    def test_multiple_unchanged_reviews_need_one_full_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            selection = self.confirmed(run)
            for row in selection['entries']:
                row['role'] = 'representative'
                row.pop('representative', None)
            write(run / 'selection.yaml', selection)
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            document = {**read(run / 'decisions.yaml'), 'decisions': [
                resolve_review(request, admission(request, i)) for i in range(len(request['candidates']))]}
            validate_decisions(document, request, wiki, Path(request['snapshot']['checkout']))
            write(run / 'decisions.yaml', document)
            with patch('repo_import.decisions.validate_decisions', wraps=validate_decisions) as validate:
                select_run(wiki, run, import_repo.prepare_review)
            self.assertEqual(1, validate.call_count)
            self.assertEqual(document['decisions'], read(run / 'decisions.yaml')['decisions'])

    def test_invalid_old_review_does_not_discard_independent_valid_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run = self.prepare(Path(directory))
            selection = self.confirmed(run)
            for row in selection['entries']:
                row['role'] = 'representative'
                row.pop('representative', None)
            write(run / 'selection.yaml', selection)
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            template = read(run / 'decisions.yaml')
            document = {**template, 'decisions': [resolve_review(request, admission(request, i))
                                                 for i in range(len(request['candidates']))]}
            for invalid_field in ('metadata', 'source_evidence', 'comparison'):
                with self.subTest(invalid_field=invalid_field):
                    altered = copy.deepcopy(document)
                    decision = altered['decisions'][0]
                    if invalid_field == 'metadata':
                        decision['metadata'].pop('title')
                    elif invalid_field == 'source_evidence':
                        decision['evidence'][0]['sha256'] = 'stale'
                    else:
                        decision.pop('overlap_resolutions')
                    write(run / 'decisions.yaml', template)
                    reuse_unchanged_reviews(wiki, run, request, altered)
                    refreshed = read(run / 'decisions.yaml')
                    self.assertEqual('defer', refreshed['decisions'][0]['decision'])
                    self.assertTrue(all(d['decision'] == 'include' for d in refreshed['decisions'][1:]))
                    selected, pending = validate_decisions(refreshed, request, wiki,
                                                           Path(request['snapshot']['checkout']))
                    self.assertEqual(2, len(selected))
                    self.assertEqual([decision['candidate_id']], pending)

    def test_real_validation_publication_and_selection_change_blocks_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wiki, run = self.prepare(base)
            # Use the real wiki tooling/data in an isolated target, never the workspace corpus.
            shutil.copytree(ROOT, wiki, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('.git', '.cache', '__pycache__', '.DS_Store'))
            write(run / 'selection.yaml', self.confirmed(run))
            select_run(wiki, run, import_repo.prepare_review)
            request = read(run / 'review-request.yaml')
            write(run / 'decisions.yaml', self.reviewed(request))
            import_repo.build(wiki, run)
            selection = read(run / 'selection.yaml')
            changed = copy.deepcopy(selection)
            changed['entries'][0]['rationale'] += ' changed'
            write(run / 'selection.yaml', changed)
            with self.assertRaisesRegex(ValueError, 'Selection changed'):
                import_repo.apply(wiki, run)
            write(run / 'selection.yaml', selection)
            import_repo.apply(wiki, run)
            implementations = read(wiki / 'data/repo-import/registry.yaml')['implementations']
            self.assertEqual(1, len(implementations))
            implementation = next(iter(implementations.values()))
            self.assertEqual(1, len(implementation['versions']))
            self.assertEqual(1, len(implementation['records']))
            formal = next((wiki / 'data/repo-import/decisions').glob('*/*.yaml'))
            self.assertNotIn('selection', read(formal))
            self.assertEqual(1, len(read(formal)['decisions']))
            # Full portable packs are external inputs, never formal corpus output.
            audit_pack = {**read(run / 'decisions.yaml'), 'selection': selection}
            source_request = read(run / 'request.yaml')
            second = import_repo.prepare(wiki, source_request['repo'], source_request['revision'], source_request['config'])
            write(second / 'selection.yaml', audit_pack['selection'])
            select_run(wiki, second, import_repo.prepare_review)
            rebound_request = read(second / 'review-request.yaml')
            rebound = rebind_manifest(audit_pack, rebound_request)
            selected, pending = validate_decisions(rebound, rebound_request, wiki, Path(rebound_request['snapshot']['checkout']))
            self.assertEqual([], pending)
            self.assertEqual(2, len(selected))
            self.assertEqual('record_duplicate', rebound['decisions'][0]['action'])

    def test_entry_kind_filter_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run = self.prepare(Path(directory))
            snapshot = read(run / 'snapshot.yaml')
            config = read(run / 'request.yaml')['config']['scan']
            config['entry_kinds'] = ['cuda_global', 'cute_kernel']
            self.assertEqual([], scan_snapshot(snapshot, config)['candidates'])
            config['entry_kinds'] = ['unknown']
            with self.assertRaises(ValueError): scan_snapshot(snapshot, config)
