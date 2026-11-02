"""Real Git snapshots exercise update observations and bounded evidence reuse."""
import tempfile
import unittest
from pathlib import Path

from repo_import_support import make_repo, commit
from repo_import.checkout import resolve_snapshot
from repo_import.scan import scan_snapshot
from repo_import.selection import propose_selection
from repo_import.store import read, write, file_hash, digest
from repo_import.updates import (prepare_update, remap_review_candidate,
                                 record_observation, observed_state_path, record_publication)


class RepositoryUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = make_repo(self.root)
        self.number = 0
        self.config = {'include': ['*'], 'exclude': [],
                       'entry_kinds': ['cuda_global', 'cute_kernel', 'triton_jit']}

    def scan(self, config=None):
        self.number += 1
        run = self.root / ('run-' + str(self.number))
        snapshot = resolve_snapshot(str(self.repo), 'HEAD', self.root / 'checkouts')
        inventory = scan_snapshot(snapshot, config or self.config)
        write(run / 'snapshot.yaml', snapshot)
        write(run / 'inventory.yaml', inventory)
        write(run / 'request.yaml', {'extractor_contract': 'fixture-scan-v1', 'config': {'scan': config or self.config,
                                               'state_dir': str(self.root / 'state')}})
        self.receipt(run)
        write(run / 'selection.yaml', propose_selection(inventory, Path(snapshot['checkout'])))
        return run

    def receipt(self, run):
        write(run / 'scan-receipt.yaml', {
            'version': 1, 'inventory_digest': digest(read(run / 'inventory.yaml')),
            'snapshot_digest': digest(read(run / 'snapshot.yaml')),
            'scan_config_digest': digest(read(run / 'request.yaml')['config']['scan']),
            'extractor_contract': read(run / 'request.yaml').get('extractor_contract')})

    def reviewed(self, run, caller=None):
        selection = read(run / 'selection.yaml')
        inventory = read(run / 'inventory.yaml')
        checkout = Path(read(run / 'snapshot.yaml')['checkout'])
        representative = selection['entries'][0]['candidate_id']
        for index, (row, candidate) in enumerate(zip(selection['entries'], inventory['candidates'])):
            row.update(role='representative' if index == 0 else 'member', rationale='Compared fixture source.',
                       evidence=[{'path': candidate['path'], 'sha256': file_hash(checkout / candidate['path']),
                                  'locator': candidate['entry']}])
            if caller:
                row['evidence'].append({'path': caller, 'sha256': file_hash(checkout / caller), 'locator': 'launch'})
            if index:
                first = inventory['candidates'][0]
                row.update(representative=representative, relation='variant', difference_summary='Fixture variant.')
                row['evidence'].append({'path': first['path'], 'sha256': file_hash(checkout / first['path']),
                                        'locator': first['entry']})
        write(run / 'selection.yaml', selection)
        return selection

    def advance(self, path, contents):
        (self.repo / path).write_text(contents)
        commit(self.repo, 'update')

    def test_commit_only_rebinds_complete_group_without_mutating_baseline(self):
        old = self.scan()
        self.reviewed(old)
        baseline = (old / 'selection.yaml').read_bytes()
        self.advance('README.md', 'unrelated commit\n')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(len(report['unchanged']), 1)
        self.assertEqual(len(report['reused_selection']), 1)
        self.assertNotEqual(next(iter(report['id_map'])), next(iter(report['id_map'].values())))
        self.assertEqual((old / 'selection.yaml').read_bytes(), baseline)
        self.assertFalse(report['publication_advanced'])
        self.assertFalse((self.root / 'data').exists())

    def test_new_definition_is_pending_while_untouched_group_is_reused(self):
        old = self.scan()
        self.reviewed(old)
        self.advance('extra.cu', '__global__ void extra(float* x) { x[0] = 2; }\n')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(len(report['added']), 1)
        self.assertEqual(len(report['reused_selection']), 1)
        roles = {r['path']: r['role'] for r in read(new / 'selection.yaml')['entries']}
        self.assertEqual(roles['extra.cu'], 'pending')

    def test_body_and_dependency_changes_block_reuse(self):
        for change_dependency in (False, True):
            with self.subTest(dependency=change_dependency):
                self.advance('dep.py', 'VALUE = ' + ('3' if change_dependency else '1') + '\n')
                self.advance('kernel.py', 'import triton\nfrom dep import VALUE\n@triton.jit\ndef kernel(x):\n return x + VALUE\n')
                old = self.scan()
                self.reviewed(old)
                self.advance('dep.py' if change_dependency else 'kernel.py',
                             'VALUE = 2\n' if change_dependency else
                             'import triton\nfrom dep import VALUE\n@triton.jit\ndef kernel(x):\n return x - VALUE\n')
                new = self.scan()
                report = prepare_update(self.root, new, old)
                self.assertEqual(len(report['changed']), 1)
                self.assertEqual(report['reused_selection'], [])

    def test_changed_explicit_caller_receipt_blocks_reuse(self):
        self.advance('caller.py', 'def launch():\n return 1\n')
        old = self.scan()
        self.reviewed(old, 'caller.py')
        self.advance('caller.py', 'def launch():\n return 2\n')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(len(report['unchanged']), 1)
        self.assertEqual(report['reused_selection'], [])

    def test_missing_representative_invalidates_members(self):
        self.advance('z.cu', '__global__ void z(float* x) { x[0] = 3; }\n')
        old = self.scan()
        selection = self.reviewed(old)
        representative = selection['entries'][0]['path']
        (self.repo / representative).unlink()
        commit(self.repo, 'remove representative')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertTrue(report['deletion_allowed'])
        self.assertEqual(len(report['missing']), 1)
        self.assertEqual(report['reused_selection'], [])
        self.assertTrue(all(r['role'] == 'pending' for r in read(new / 'selection.yaml')['entries']))

    def test_configuration_or_parse_failure_prevents_deletion_assertion(self):
        old = self.scan()
        self.reviewed(old)
        new = self.scan({**self.config, 'exclude': ['kernel.py']})
        report = prepare_update(self.root, new, old)
        self.assertFalse(report['deletion_allowed'])
        self.assertIn('scan_configuration_changed', report['incompatible_scan'])
        self.advance('kernel.py', 'this is not valid python !!!\n')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertFalse(report['deletion_allowed'])
        self.assertIn('scan_has_parse_failures', report['incompatible_scan'])

    def test_ambiguous_rename_hashes_are_only_hints(self):
        old = self.scan()
        self.reviewed(old)
        code = (self.repo / 'kernel.py').read_text()
        (self.repo / 'kernel.py').unlink()
        (self.repo / 'moved.py').write_text(code)
        (self.repo / 'copy.py').write_text(code)
        commit(self.repo, 'move and duplicate')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(len(report['possible_renames'][0]['candidates']), 2)
        self.assertEqual(report['id_map'], {})
        self.assertEqual(report['reused_selection'], [])

    def test_fresh_review_disables_reuse_and_other_repo_is_rejected(self):
        old = self.scan()
        self.reviewed(old)
        self.advance('README.md', 'commit change')
        new = self.scan()
        report = prepare_update(self.root, new, old, fresh_review=True)
        self.assertEqual(report['reused_selection'], [])
        snapshot = read(new / 'snapshot.yaml')
        snapshot['repo'] = 'https://example.invalid/another'
        write(new / 'snapshot.yaml', snapshot)
        self.receipt(new)
        with self.assertRaisesRegex(ValueError, 'different repository'):
            prepare_update(self.root, new, old)

    def test_missing_member_invalidates_original_group_without_deleting_sources(self):
        self.advance('z.cu', '__global__ void z(float* x) { x[0] = 3; }\n')
        old = self.scan()
        selection = self.reviewed(old)
        member_path = selection['entries'][1]['path']
        (self.repo / member_path).unlink()
        commit(self.repo, 'remove member')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(report['reused_selection'], [])
        self.assertEqual(len(report['invalidated_selection']), 1)
        self.assertEqual(read(new / 'selection.yaml')['entries'][0]['role'], 'pending')
        self.assertTrue((old / 'inventory.yaml').exists())

    def test_extractor_contract_change_blocks_deletion_and_reuse(self):
        old = self.scan()
        self.reviewed(old)
        new = self.scan()
        inventory = read(new / 'inventory.yaml')
        inventory['candidates'][0]['fingerprints']['extraction_version'] = 'future-contract'
        write(new / 'inventory.yaml', inventory)
        snapshot = read(new / 'snapshot.yaml')
        write(new / 'selection.yaml', propose_selection(inventory, Path(snapshot['checkout'])))
        self.receipt(new)
        report = prepare_update(self.root, new, old)
        self.assertIn('extractor_contract_changed', report['incompatible_scan'])
        self.assertFalse(report['deletion_allowed'])
        self.assertEqual(report['reused_selection'], [])

    def test_changed_skipped_gitlink_does_not_reuse_dependency_assumptions(self):
        import subprocess
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True).strip()
        initial = git('rev-parse', 'HEAD')
        git('update-index', '--add', '--cacheinfo', '160000,' + initial + ',external-kernels')
        git('commit', '-m', 'pin skipped dependency')
        old = self.scan()
        self.reviewed(old)
        (self.repo / 'README.md').write_text('changed dependency pin next\n')
        git('add', 'README.md')
        git('commit', '-m', 'new dependency target')
        current = git('rev-parse', 'HEAD')
        git('update-index', '--cacheinfo', '160000,' + current + ',external-kernels')
        git('commit', '-m', 'update skipped dependency')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(1, len(report['unchanged']))
        self.assertIn('skipped_dependency_pins_changed', report['incompatible_scan'])
        self.assertEqual([], report['reused_selection'])
        self.assertFalse((Path(read(new / 'snapshot.yaml')['checkout']) / 'external-kernels/kernel.py').exists())

    def test_scan_coverage_contract_change_disables_reuse_and_deletion_assertions(self):
        old = self.scan()
        self.reviewed(old)
        new = self.scan()
        request = read(new / 'request.yaml')
        request['extractor_contract'] = 'expanded-definition-markers'
        write(new / 'request.yaml', request)
        self.receipt(new)
        report = prepare_update(self.root, new, old)
        self.assertIn('extractor_contract_changed', report['incompatible_scan'])
        self.assertFalse(report['deletion_allowed'])
        self.assertEqual([], report['reused_selection'])

    def test_unchanged_complete_file_disambiguates_same_named_definitions(self):
        self.advance('overloaded.cu', '__global__ void overload(float* x) { x[0] = 1; }\n'
                     '__global__ void overload(int* x) { x[0] = 2; }\n')
        old = self.scan()
        self.advance('README.md', 'documentation only\n')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual([], report['ambiguous_identities'])
        self.assertEqual(3, len(report['unchanged']))
        self.assertTrue(report['deletion_allowed'])

    def test_overloaded_definitions_are_ambiguous_observations_not_deletions(self):
        self.advance('overloaded.cu', '__global__ void overload(float* x) { x[0] = 1; }\n'
                     '__global__ void overload(int* x) { x[0] = 2; }\n')
        old = self.scan()
        self.advance('overloaded.cu', (self.repo / 'overloaded.cu').read_text() + '// changed compile context\n')
        new = self.scan()
        report = prepare_update(self.root, new, old)
        self.assertEqual(len(report['ambiguous_identities']), 1)
        self.assertEqual(len(report['ambiguous_identities'][0]['previous']), 2)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['added'], [])
        self.assertFalse(report['deletion_allowed'])
        self.assertIn('ambiguous_candidate_identity', report['incompatible_scan'])

    def test_observation_index_keeps_complete_scan_and_never_advances_publication(self):
        first = self.scan()
        state = record_observation(self.root, first)
        self.assertEqual(state['last_complete_scan'], str(first))
        path = observed_state_path(self.root, first)
        self.assertTrue(path.is_relative_to(self.root / 'state'))
        state['last_published_run'] = 'previous-published-run'
        write(path, state)
        self.advance('broken.py', 'this is invalid python !!!\n')
        partial = self.scan()
        state = record_observation(self.root, partial)
        self.assertEqual(state['last_complete_scan'], str(first))
        self.assertEqual(state['last_observed_scan'], str(partial))
        self.assertEqual(state['last_published_run'], 'previous-published-run')
        (self.repo / 'broken.py').unlink()
        commit(self.repo, 'fix parse error')
        failed = self.scan()
        write(failed / 'result.yaml', {'status': 'failed'})
        state = record_observation(self.root, failed)
        self.assertEqual(state['last_complete_scan'], str(first))
        self.assertFalse(state['scans'][str(failed)]['complete'])

    def test_receipts_reject_tampering_and_legacy_scans_are_conservative(self):
        old = self.scan()
        self.reviewed(old)
        new = self.scan()
        request = read(new / 'request.yaml')
        request['config']['scan']['exclude'] = ['other.py']
        write(new / 'request.yaml', request)
        with self.assertRaisesRegex(ValueError, 'Scan receipt'):
            prepare_update(self.root, new, old)
        with self.assertRaisesRegex(ValueError, 'Scan receipt'):
            record_observation(self.root, new)
        self.receipt(new)
        (old / 'scan-receipt.yaml').unlink()
        report = prepare_update(self.root, new, old)
        self.assertFalse(report['deletion_allowed'])
        self.assertEqual(report['reused_selection'], [])
        self.assertIn('legacy_scan_receipt_missing', report['incompatible_scan'])
        state = record_observation(self.root, old)
        self.assertNotIn('last_complete_scan', state)
        self.assertFalse(state['scans'][str(old)]['receipt_verified'])

    def test_publication_does_not_reset_newer_observations_or_promote_partial_run(self):
        old = self.scan()
        record_observation(self.root, old)
        self.advance('README.md', 'newer observation\n')
        newer = self.scan()
        record_observation(self.root, newer)
        with self.assertRaisesRegex(ValueError, 'committed apply'):
            record_publication(self.root, old)
        write(old / 'transaction.yaml', {'status': 'committed'})
        write(old / 'result.yaml', {'status': 'applied', 'included': 1, 'pending': []})
        state = record_publication(self.root, old)
        self.assertEqual(state['last_observed_scan'], str(newer))
        self.assertEqual(state['last_complete_scan'], str(newer))
        self.assertEqual(state['last_published_run'], str(old))
        write(newer / 'transaction.yaml', {'status': 'committed'})
        write(newer / 'result.yaml', {'status': 'applied', 'included': 1, 'pending': ['unresolved']})
        state = record_publication(self.root, newer)
        self.assertEqual(state['last_applied_run'], str(newer))
        self.assertEqual(state['last_published_run'], str(old))

    def test_concurrent_observations_preserve_all_runs(self):
        from concurrent.futures import ThreadPoolExecutor
        runs = [self.scan() for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda run: record_observation(self.root, run), runs))
        state = read(observed_state_path(self.root, runs[0]))
        self.assertEqual(set(state['scans']), {str(run) for run in runs})

    def test_review_remapping_drops_stale_overlap_authority(self):
        verdict = {'candidate_id': 'old', 'target_candidate_id': 'other',
                   'overlap_resolutions': [{'match_id': 'old-context'}]}
        remapped = remap_review_candidate(verdict, {'old': 'new', 'other': 'new-other'})
        self.assertEqual(remapped['candidate_id'], 'new')
        self.assertNotIn('overlap_resolutions', remapped)
        self.assertIsNone(remap_review_candidate(verdict, {'old': 'new'}))
        self.assertIn('overlap_resolutions', verdict)


if __name__ == '__main__':
    unittest.main()
