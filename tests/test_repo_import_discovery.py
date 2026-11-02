"""Real Git snapshots and CLI guards for reviewer-discovered comparisons."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from repo_import_support import (ROOT, prepared, read, write, document, admission,
                                 commit, load_config, resolve_review)
import import_repo
from repo_import.discovery import definition_receipt, page_receipt, validate_discoveries
from repo_import.store import digest, file_hash
from repo_import.materials import prepare_review, export_audit_pack, prepare_selection, reuse_update_reviews
from repo_import.match import existing_index
from repo_import.decisions import validate_decisions
from repo_import.selection import select_run
from repo_import.updates import prepare_update


class DiscoveredOverlapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.wiki, self.repo, self.run, self.request = prepared(self.base)
        self.page('sources/prs/unrelated.md', 'pr-unrelated', [])
        self.page('wiki/kernels/one.md', 'kernel-one', ['pr-unrelated'])
        self.page('wiki/kernels/two.md', 'kernel-two', ['pr-unrelated'])
        self.discovery = {'version': 1, 'entries': [self.row(self.request['candidates'][0])]}

    def page(self, path, identity, sources):
        destination = self.wiki / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('---\nid: ' + identity + '\ntitle: Other operation\nsources: ' + str(sources) + '\n---\nOriginal source evidence.\n')

    def row(self, candidate):
        page = next(p for p in existing_index(self.wiki) if p['page_id'] == 'pr-unrelated')
        return {'candidate_id': candidate['candidate_id'], 'page_id': page['page_id'],
                'rationale': 'Source inspection found a renamed implementation in another repository.',
                'input_evidence': [definition_receipt(candidate)], 'matched_evidence': [page_receipt(page)]}

    def refresh(self):
        write(self.run / 'discovered-overlaps.yaml', self.discovery)
        prepare_review(self.wiki, self.run, read(self.run / 'snapshot.yaml'), read(self.run / 'inventory.yaml'))
        return read(self.run / 'review-request.yaml')

    def batch_discovery(self, same_file=False):
        code = 'import triton\n@triton.jit\ndef negate(value):\n    return -value\n'
        if same_file:
            path = self.repo / 'kernel.py'
            path.write_text(path.read_text() + code)
        else:
            (self.repo / 'second.py').write_text(code)
        commit(self.repo)
        self.run = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki))
        candidates = read(self.run / 'inventory.yaml')['candidates']
        first, second = candidates
        self.discovery = {'version': 1, 'entries': [{
            'candidate_id': first['candidate_id'], 'target_candidate_id': second['candidate_id'],
            'rationale': 'Source inspection found a possible relationship outside automatic matching.',
            'input_evidence': [definition_receipt(first)], 'matched_evidence': [definition_receipt(second)]}]}
        return self.refresh()

    def test_discovered_batch_overlap_is_reciprocal_mandatory_and_exported(self):
        request = self.batch_discovery()
        for candidate in request['candidates']:
            hints = request['matches'][candidate['candidate_id']]
            self.assertEqual(1, len(hints))
            self.assertEqual('batch', hints[0]['kind'])
            self.assertNotEqual(candidate['candidate_id'], hints[0]['candidate_id'])
        decisions = [admission(request, i) for i in range(2)]
        saved = {**document(request), 'decisions': decisions}
        selected, pending = validate_decisions(saved, request, self.wiki, Path(request['snapshot']['checkout']))
        self.assertFalse(selected)
        self.assertEqual(2, len(pending))
        for decision in decisions:
            resolve_review(request, decision)
        selected, pending = validate_decisions(saved, request, self.wiki, Path(request['snapshot']['checkout']))
        self.assertEqual((2, []), (len(selected), pending))
        write(self.run / 'decisions.yaml', saved)
        self.assertEqual(self.discovery, export_audit_pack(self.wiki, self.run, self.run / 'batch-pack.yaml')['discovered_overlaps'])
        self.discovery['entries'][0]['rationale'] += ' changed'
        write(self.run / 'discovered-overlaps.yaml', self.discovery)
        self.assertIn('Discovered overlaps changed', self.cli('report').stderr)

    def test_batch_discovery_rejects_invalid_or_reversed_duplicate_pairs(self):
        request = self.batch_discovery()
        original = copy.deepcopy(self.discovery)
        for change in ('unknown', 'self', 'mixed', 'reverse', 'locator', 'hash'):
            with self.subTest(change=change):
                self.discovery = copy.deepcopy(original)
                row = self.discovery['entries'][0]
                if change == 'unknown': row['target_candidate_id'] = 'unknown'
                elif change == 'self': row['target_candidate_id'] = row['candidate_id']
                elif change == 'mixed': row['page_id'] = 'pr-unrelated'
                elif change == 'reverse':
                    self.discovery['entries'].append({**row, 'candidate_id': row['target_candidate_id'],
                                                      'target_candidate_id': row['candidate_id'],
                                                      'input_evidence': row['matched_evidence'],
                                                      'matched_evidence': row['input_evidence']})
                elif change == 'locator': row['matched_evidence'][0]['locator'] = 'uninspected'
                else: row['matched_evidence'][0]['sha256'] = '0' * 64
                with self.assertRaises(ValueError): self.refresh()

    def test_all_reciprocal_batch_comparisons_must_agree(self):
        request = self.batch_discovery(same_file=True)
        decisions = [resolve_review(request, admission(request, i)) for i in range(2)]
        self.assertEqual(2, len(decisions[1]['overlap_resolutions']))
        # A heuristic and an independently discovered request address the same pair.
        decisions[1]['overlap_resolutions'][1].update(outcome='same_implementation',
                                                     representative=decisions[0]['candidate_id'])
        saved = {**document(request), 'decisions': decisions}
        selected, pending = validate_decisions(saved, request, self.wiki, Path(request['snapshot']['checkout']))
        self.assertFalse(selected)
        self.assertEqual(2, len(pending))

    def test_discovered_batch_member_projects_both_directions_and_skips_internal_pair(self):
        self.batch_discovery()
        (self.repo / 'member.py').write_text((self.repo / 'kernel.py').read_text().replace('reduce_kernel', 'member_kernel'))
        commit(self.repo)
        self.run = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki))
        selection = self.selection(self.run)
        candidates = {c['path']: c for c in read(self.run / 'inventory.yaml')['candidates']}
        first, member, other = [candidates[p] for p in ('kernel.py', 'member.py', 'second.py')]
        row = next(r for r in selection['entries'] if r['candidate_id'] == member['candidate_id'])
        row.update(role='member', representative=first['candidate_id'], relation='variant',
                   difference_summary='Fixture actual copied-body relation.',
                   evidence=[definition_receipt(member), definition_receipt(first)])
        write(self.run / 'selection.yaml', selection)
        def pair(a, b):
            return {'candidate_id': a['candidate_id'], 'target_candidate_id': b['candidate_id'],
                    'rationale': 'Fixture source-discovered comparison.',
                    'input_evidence': [definition_receipt(a)], 'matched_evidence': [definition_receipt(b)]}
        write(self.run / 'discovered-overlaps.yaml', {'version': 1, 'entries': [pair(member, other), pair(first, member)]})
        select_run(self.wiki, self.run)
        request = read(self.run / 'review-request.yaml')
        self.assertEqual([], request['matches'][member['candidate_id']])
        for a, b in ((first, other), (other, first)):
            hints = [h for h in request['matches'][a['candidate_id']] if h.get('discovered_overlap')]
            self.assertEqual(1, len(hints))
            self.assertEqual(b['candidate_id'], hints[0]['candidate_id'])

    def test_batch_update_reuses_only_when_both_endpoints_are_unchanged(self):
        request = self.batch_discovery()
        self.selection(self.run)
        select_run(self.wiki, self.run)
        request = read(self.run / 'review-request.yaml')
        write(self.run / 'decisions.yaml', {**document(request), 'selection_digest': digest(request['selection']),
              'decisions': [resolve_review(request, admission(request, i)) for i in range(2)]})
        (self.repo / 'README.md').write_text('unrelated commit\n')
        commit(self.repo)
        new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, new)
        prepare_update(self.wiki, new, self.run)
        select_run(self.wiki, new)
        self.assertEqual(2, len(reuse_update_reviews(self.wiki, new)))
        # A changed counterpart invalidates the previous comparison on BOTH sides.
        path = self.repo / 'second.py'
        path.write_text(path.read_text().replace('-value', 'value * value'))
        commit(self.repo)
        changed = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, changed)
        report = prepare_update(self.wiki, changed, new)
        self.assertEqual(1, len(report['changed']))
        self.selection(changed)
        select_run(self.wiki, changed)
        self.assertEqual([], reuse_update_reviews(self.wiki, changed))
        refreshed = read(changed / 'discovered-overlaps.yaml')['entries'][0]
        target = next(c for c in read(changed / 'inventory.yaml')['candidates'] if c['candidate_id'] == refreshed['target_candidate_id'])
        self.assertEqual(definition_receipt(target), refreshed['matched_evidence'][0])

    def test_batch_update_retains_renames_and_missing_target_history(self):
        self.batch_discovery()
        self.selection(self.run)
        original = (self.repo / 'second.py').read_text()
        (self.repo / 'second.py').unlink()
        (self.repo / 'renamed.py').write_text(original)
        (self.repo / 'copied.py').write_text(original)
        commit(self.repo)
        new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, new)
        report = prepare_update(self.wiki, new, self.run)
        self.assertEqual(2, len(read(new / 'discovered-overlaps.yaml')['entries']))
        self.assertEqual(2, len(report['discovered_overlap_history'][0]['target_successors']))
        (self.repo / 'renamed.py').unlink()
        (self.repo / 'copied.py').unlink()
        commit(self.repo)
        missing = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, missing)
        removed = prepare_update(self.wiki, missing, new)
        self.assertFalse(read(missing / 'discovered-overlaps.yaml')['entries'])
        self.assertTrue(all(h['status'] == 'no_observed_successor' for h in removed['discovered_overlap_history']))

    def cli(self, command):
        return subprocess.run([sys.executable, str(ROOT / 'scripts/import_repo.py'), '--wiki-root', str(self.wiki),
                               command, '--run', str(self.run)], text=True, capture_output=True)

    def test_discovered_pr_projects_all_canonical_pages_and_blocks_admission(self):
        request = self.refresh()
        hints = request['matches'][request['candidates'][0]['candidate_id']]
        self.assertEqual({'pr-unrelated', 'kernel-one', 'kernel-two'}, {h['page_id'] for h in hints})
        selected, pending = validate_decisions(document(request), request, self.wiki, Path(request['snapshot']['checkout']))
        self.assertFalse(selected)
        self.assertTrue(pending)
        self.assertEqual(0, self.cli('report').returncode)
        pack = export_audit_pack(self.wiki, self.run, self.run / 'portable.yaml')
        self.assertEqual(self.discovery, pack['discovered_overlaps'])

    def test_unknown_duplicate_stale_tampered_and_invalid_locator_are_rejected(self):
        for change in ('unknown_candidate', 'unknown_page', 'duplicate', 'input_hash', 'target_hash', 'locator'):
            with self.subTest(change=change):
                discovery = copy.deepcopy(self.discovery)
                row = discovery['entries'][0]
                if change == 'unknown_candidate': row['candidate_id'] = 'unknown'
                elif change == 'unknown_page': row['page_id'] = 'unknown'
                elif change == 'duplicate': discovery['entries'].append(copy.deepcopy(row))
                elif change == 'input_hash': row['input_evidence'][0]['sha256'] = '0' * 64
                elif change == 'target_hash': row['matched_evidence'][0]['sha256'] = '0' * 64
                else: row['input_evidence'][0]['locator'] = 'uninspected'
                write(self.run / 'discovered-overlaps.yaml', discovery)
                with self.assertRaises(ValueError):
                    prepare_review(self.wiki, self.run, read(self.run / 'snapshot.yaml'), read(self.run / 'inventory.yaml'))
        request = self.refresh()
        path = self.wiki / 'sources/prs/unrelated.md'
        path.write_text(path.read_text() + 'Changed target.\n')
        with self.assertRaisesRegex(ValueError, 'Stale discovered'):
            validate_decisions(document(request), request, self.wiki, Path(request['snapshot']['checkout']))
        self.page('sources/prs/unrelated.md', 'pr-unrelated', [])
        checkout = Path(request['snapshot']['checkout'])
        source = checkout / request['candidates'][0]['path']
        source.write_text(source.read_text() + '# tampered pinned source\n')
        with self.assertRaisesRegex(ValueError, 'Stale discovered'):
            validate_decisions(document(request), request, self.wiki, checkout)

    def test_only_known_target_code_assets_and_locators_are_accepted(self):
        asset = self.wiki / 'artifacts/prs/unrelated/target.cu'
        asset.parent.mkdir(parents=True)
        asset.write_text('__global__ void original(float* x) { x[0] = 1; }\n')
        row = self.discovery['entries'][0]
        row['matched_evidence'] = [{'path': asset.relative_to(self.wiki).as_posix(),
                                    'sha256': file_hash(asset), 'locator': 'original'}]
        self.refresh()
        row['matched_evidence'][0]['locator'] = 'uninspected'
        with self.assertRaisesRegex(ValueError, 'exact known'):
            self.refresh()
        row['matched_evidence'][0]['locator'] = 'original'
        row['matched_evidence'][0]['path'] = 'artifacts/prs/another/target.cu'
        with self.assertRaisesRegex(ValueError, 'exact known'):
            self.refresh()

    def test_added_changed_or_removed_input_blocks_all_cli_outputs_until_refresh(self):
        for mutation in ('add', 'change', 'remove'):
            with self.subTest(mutation=mutation):
                request = self.refresh()
                import_repo.build(self.wiki, self.run)
                if mutation == 'add':
                    write(self.run / 'discovered-overlaps.yaml', {**self.discovery, 'entries': []})
                elif mutation == 'change':
                    value = copy.deepcopy(self.discovery)
                    value['entries'][0]['rationale'] = 'Changed comparison basis.'
                    write(self.run / 'discovered-overlaps.yaml', value)
                else:
                    (self.run / 'discovered-overlaps.yaml').unlink()
                for command in ('report', 'build', 'apply', 'export-audit'):
                    result = self.cli(command)
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn('Discovered overlaps changed', result.stderr)
                self.refresh()
                self.assertEqual(0, self.cli('report').returncode)
        # Addition to a run prepared without this independent input is stale too.
        (self.run / 'discovered-overlaps.yaml').unlink()
        prepare_review(self.wiki, self.run, read(self.run / 'snapshot.yaml'), read(self.run / 'inventory.yaml'))
        write(self.run / 'discovered-overlaps.yaml', self.discovery)
        self.assertIn('Discovered overlaps changed', self.cli('build').stderr)

    def selection(self, run):
        selection = prepare_selection(self.wiki, run)
        candidates = read(run / 'inventory.yaml')['candidates']
        for row, candidate in zip(selection['entries'], candidates):
            row.update(role='representative', rationale='Fixture inspected source.', evidence=[definition_receipt(candidate)])
        write(run / 'selection.yaml', selection)
        return selection

    def test_grouped_member_discovery_projects_to_representative(self):
        (self.repo / 'second.py').write_text((self.repo / 'kernel.py').read_text().replace('reduce_kernel', 'different_kernel'))
        commit(self.repo)
        self.run = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki))
        selection = self.selection(self.run)
        first, member = selection['entries']
        member.update(role='member', representative=first['candidate_id'], relation='variant',
                      difference_summary='Fixture confirms implementation relationship.')
        member['evidence'] += first['evidence']
        write(self.run / 'selection.yaml', selection)
        candidate = read(self.run / 'inventory.yaml')['candidates'][1]
        write(self.run / 'discovered-overlaps.yaml', {'version': 1, 'entries': [self.row(candidate)]})
        select_run(self.wiki, self.run)
        request = read(self.run / 'review-request.yaml')
        self.assertEqual([], request['matches'][member['candidate_id']])
        self.assertEqual({'pr-unrelated', 'kernel-one', 'kernel-two'},
                         {h['page_id'] for h in request['matches'][first['candidate_id']] if h['kind'] != 'batch'})

    def test_update_retains_changed_and_possible_rename_hints_with_current_receipts(self):
        for rename in (False, True):
            with self.subTest(rename=rename):
                self.refresh()
                self.selection(self.run)
                old_run = self.run
                original = (self.repo / 'kernel.py').read_text()
                if rename:
                    (self.repo / 'kernel.py').unlink()
                    (self.repo / 'moved.py').write_text(original)
                    (self.repo / 'copied.py').write_text(original)
                else:
                    (self.repo / 'kernel.py').write_text(original.replace('x + n', 'x - n'))
                commit(self.repo)
                new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
                prepare_selection(self.wiki, new)
                report = prepare_update(self.wiki, new, old_run)
                discovery = read(new / 'discovered-overlaps.yaml')
                self.assertEqual(2 if rename else 1, len(discovery['entries']))
                validate_discoveries(discovery, read(new / 'inventory.yaml')['candidates'], existing_index(self.wiki),
                                     self.wiki, Path(read(new / 'snapshot.yaml')['checkout']))
                self.assertEqual('retained_navigation_only', report['discovered_overlap_history'][0]['status'])
                self.assertEqual([], report['reused_selection'])
                if not rename:
                    # The next iteration starts with this changed source and run.
                    self.run = new
                    self.request = {'candidates': read(new / 'inventory.yaml')['candidates']}
                    self.discovery = {'version': 1, 'entries': [self.row(self.request['candidates'][0])]}

    def test_unchanged_update_reuses_content_bound_discovered_comparison(self):
        self.refresh()
        self.selection(self.run)
        select_run(self.wiki, self.run)
        request = read(self.run / 'review-request.yaml')
        verdict = resolve_review(request, admission(request))
        write(self.run / 'decisions.yaml', {**document(request, verdict),
              'selection_digest': digest(request['selection'])})
        (self.repo / 'README.md').write_text('unrelated commit\n')
        commit(self.repo)
        new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, new)
        report = prepare_update(self.wiki, new, self.run)
        self.assertEqual(1, len(report['unchanged']))
        self.assertEqual(1, len(read(new / 'discovered-overlaps.yaml')['entries']))
        select_run(self.wiki, new)
        reused = reuse_update_reviews(self.wiki, new)
        self.assertEqual(1, len(reused))
        self.assertEqual('include', read(new / 'decisions.yaml')['decisions'][0]['decision'])
        current = read(new / 'review-request.yaml')
        selected, pending = validate_decisions(read(new / 'decisions.yaml'), current,
                self.wiki, Path(current['snapshot']['checkout']))
        self.assertEqual((1, []), (len(selected), pending))

    def test_changed_target_refreshes_navigation_but_does_not_reuse_distinct_verdict(self):
        self.refresh()
        self.selection(self.run)
        select_run(self.wiki, self.run)
        request = read(self.run / 'review-request.yaml')
        verdict = resolve_review(request, admission(request))
        saved = {**document(request, verdict), 'selection_digest': digest(request['selection'])}
        write(self.run / 'decisions.yaml', saved)
        selected, pending = validate_decisions(saved, request, self.wiki, Path(request['snapshot']['checkout']))
        self.assertTrue(selected)
        self.assertFalse(pending)
        page = self.wiki / 'sources/prs/unrelated.md'
        page.write_text(page.read_text() + 'New target evidence.\n')
        (self.repo / 'README.md').write_text('unrelated source commit\n')
        commit(self.repo)
        new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, new)
        prepare_update(self.wiki, new, self.run)
        select_run(self.wiki, new)
        self.assertEqual([], reuse_update_reviews(self.wiki, new))
        self.assertEqual('defer', read(new / 'decisions.yaml')['decisions'][0]['decision'])
        receipt = read(new / 'discovered-overlaps.yaml')['entries'][0]['matched_evidence'][0]
        self.assertEqual(file_hash(page), receipt['sha256'])

    def test_ambiguous_successors_all_retain_the_known_comparison(self):
        code = '__global__ void overload(float* x) { x[0] = 1; }\n' + '__global__ void overload(int* x) { x[0] = 2; }\n'
        (self.repo / 'overloaded.cu').write_text(code)
        commit(self.repo)
        baseline = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki))
        candidate = next(c for c in read(baseline / 'inventory.yaml')['candidates'] if c['path'] == 'overloaded.cu')
        write(baseline / 'discovered-overlaps.yaml', {'version': 1, 'entries': [self.row(candidate)]})
        prepare_review(self.wiki, baseline, read(baseline / 'snapshot.yaml'), read(baseline / 'inventory.yaml'))
        self.selection(baseline)
        (self.repo / 'overloaded.cu').write_text(code + '// changed translation unit\n')
        commit(self.repo)
        new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, new)
        report = prepare_update(self.wiki, new, baseline)
        self.assertEqual(1, len(report['ambiguous_identities']))
        rows = read(new / 'discovered-overlaps.yaml')['entries']
        self.assertEqual(2, len(rows))
        self.assertEqual(set(report['ambiguous_identities'][0]['current']), {r['candidate_id'] for r in rows})

    def test_audit_preset_replay_reprepares_discovery_input(self):
        self.refresh()
        manifest = self.wiki / 'examples/audit.yaml'
        export_audit_pack(self.wiki, self.run, manifest)
        write(self.wiki / 'data/repo-import-presets.yaml', {'version': 1, 'presets': {
            'fixture': {'repo': str(self.repo), 'revision': 'HEAD', 'decision_manifest': 'examples/audit.yaml'}}})
        old_runs = set(self.run.parent.iterdir())
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/import_repo.py'), '--wiki-root', str(self.wiki),
                                 'import', '--preset', 'fixture'], text=True, capture_output=True)
        self.assertNotEqual(0, result.returncode)  # Saved deferral remains pending.
        self.assertIn('need external review', result.stderr)
        replay = next(p for p in set(self.run.parent.iterdir()) - old_runs if (p / 'review-request.yaml').exists())
        request = read(replay / 'review-request.yaml')
        self.assertEqual(self.discovery, request['discovered_overlaps'])
        self.assertEqual(self.discovery, read(replay / 'discovered-overlaps.yaml'))
        self.assertIn('pr-unrelated', {h.get('page_id') for h in request['matches'][request['candidates'][0]['candidate_id']]})

    def test_update_reports_missing_current_target_without_changing_history(self):
        self.refresh()
        self.selection(self.run)
        before = (self.run / 'discovered-overlaps.yaml').read_bytes()
        (self.wiki / 'sources/prs/unrelated.md').unlink()
        (self.repo / 'README.md').write_text('new commit\n')
        commit(self.repo)
        new = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, new)
        report = prepare_update(self.wiki, new, self.run)
        self.assertEqual('missing_current_page', report['discovered_overlap_history'][0]['status'])
        self.assertEqual(before, (self.run / 'discovered-overlaps.yaml').read_bytes())
        self.assertEqual(1, len(report['unresolved_discovered_overlaps']))
        # A temporarily absent target remains known through a later update.
        self.page('sources/prs/unrelated.md', 'pr-unrelated', [])
        (self.repo / 'README.md').write_text('another commit\n')
        commit(self.repo)
        successor = import_repo.prepare(self.wiki, str(self.repo), 'HEAD', load_config(self.wiki), scan_only=True)
        prepare_selection(self.wiki, successor)
        restored = prepare_update(self.wiki, successor, new)
        self.assertEqual(1, len(read(successor / 'discovered-overlaps.yaml')['entries']))
        self.assertEqual([], restored['unresolved_discovered_overlaps'])
        self.assertTrue(restored['baseline_discovered_overlap_history'])


if __name__ == '__main__':
    unittest.main()
