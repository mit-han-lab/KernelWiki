import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import test_repo_import_selection as fixtures
from repo_import_support import ROOT, make_repo, make_wiki, import_repo, read, write, commit, digest, admission
from repo_import.config import load_config
from repo_import.materials import prepare_selection, verify_inventory, reuse_update_reviews
from repo_import.selection import select_run
from repo_import.updates import prepare_update, observed_state_path


class MaterialTests(unittest.TestCase):
    def test_real_cli_scan_review_update_report_and_external_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            repo, wiki = make_repo(base), make_wiki(base)
            config = load_config(wiki)
            config['selection']['enabled'] = True
            write(wiki / 'data/repo-import.yaml', config)

            def cli(*args):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/import_repo.py'),
                    '--wiki-root', str(wiki), *map(str, args)], capture_output=True, text=True)
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                return result.stdout

            output = cli('scan', '--repo', repo, '--fresh-review')
            run = Path(output.strip().split('Run directory: ')[-1])
            self.assertFalse((run / 'selection.yaml').exists())
            cli('prepare-selection', '--run', run)
            inventory = read(run / 'inventory.yaml')
            selection = read(run / 'selection.yaml')
            c = inventory['candidates'][0]
            selection['entries'][0].update(role='representative', rationale='Fixture numerical entry.',
                evidence=[{'path': c['path'], 'sha256': c['fingerprints']['source_sha256'], 'locator': c['entry']}])
            write(run / 'selection.yaml', selection)
            cli('validate-selection', '--run', run)
            self.assertFalse((run / 'decisions.yaml').exists())
            cli('prepare-review', '--run', run)
            request = read(run / 'review-request.yaml')
            write(run / 'decisions.yaml', {'version': 1, 'context_digest': request['context_digest'],
                'selection_digest': digest(selection), 'decisions': [admission(request)]})
            cli('report', '--run', run, '--out', base / 'report')
            self.assertEqual(1, read(base / 'report/summary.yaml')['effective_kernels'])
            cli('export-audit', '--run', run, '--out', base / 'pack.yaml')
            self.assertEqual(selection, read(base / 'pack.yaml')['selection'])
            (repo / 'README.md').write_text('New commit with the same numerical source.')
            commit(repo)
            output = cli('update', '--repo', repo, '--revision', 'HEAD', '--baseline', run)
            current = Path(output.strip().split('Run directory: ')[-1])
            cli('prepare-review', '--run', current)
            self.assertEqual('include', read(current / 'decisions.yaml')['decisions'][0]['decision'])
            cli('report', '--run', current, '--out', base / 'updated-report')
            self.assertEqual(1, read(base / 'updated-report/summary.yaml')['effective_kernels'])
            self.assertFalse((wiki / 'data/repo-import/registry.yaml').exists())
            self.assertFalse((wiki / 'wiki/kernels/imported').exists())

    def test_scan_selection_and_review_are_independent_and_scan_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            repo, wiki = make_repo(base), make_wiki(base)
            config = load_config(wiki)
            config['selection']['enabled'] = True
            run = import_repo.prepare(wiki, str(repo), 'HEAD', config, scan_only=True)
            originals = {name: (run / name).read_bytes() for name in
                         ('snapshot.yaml', 'inventory.yaml', 'scan-receipt.yaml')}
            self.assertFalse((run / 'selection.yaml').exists())
            self.assertFalse((run / 'decisions.yaml').exists())
            prepare_selection(wiki, run)
            selection = read(run / 'selection.yaml')
            selection['entries'][0].update(role='representative', rationale='Launched numerical fixture.',
                evidence=[{'path': 'kernel.py', 'sha256': read(run / 'inventory.yaml')['candidates'][0]['fingerprints']['source_sha256'],
                           'locator': 'reduce_kernel'}])
            write(run / 'selection.yaml', selection)
            self.assertEqual(selection, prepare_selection(wiki, run))
            select_run(wiki, run)
            self.assertTrue((run / 'decisions.yaml').exists())
            for name, value in originals.items():
                self.assertEqual(value, (run / name).read_bytes())
            state = read(observed_state_path(wiki, run))
            self.assertEqual(str(run.resolve()), state['last_complete_scan'])
            self.assertNotIn('last_published_run', state)

    def test_scan_receipt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            repo, wiki = make_repo(base), make_wiki(base)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki), scan_only=True)
            inventory = read(run / 'inventory.yaml')
            inventory['candidates'][0]['entry'] = 'substituted'
            write(run / 'inventory.yaml', inventory)
            with self.assertRaisesRegex(ValueError, 'Stored scan inputs changed'):
                verify_inventory(run)

    def test_fresh_review_never_reuses_saved_exclusion(self):
        fixture = fixtures.SelectionTests()
        with tempfile.TemporaryDirectory() as directory:
            wiki, baseline = fixture.prepare(Path(directory).resolve())
            selection = fixture.confirmed(baseline)
            write(baseline / 'selection.yaml', selection)
            select_run(wiki, baseline)
            request = read(baseline / 'review-request.yaml')
            exclusion = dict(read(baseline / 'decisions.yaml')['decisions'][0],
                             decision='exclude', rationale='Prior fixture exclusion.')
            write(wiki / 'data/repo-import/decisions' / request['snapshot']['repo_id'] / 'prior.yaml',
                  {'decisions': [exclusion]})
            saved = read(baseline / 'request.yaml')
            current = import_repo.prepare(wiki, saved['repo'], 'HEAD', saved['config'], fresh_review=True)
            write(current / 'selection.yaml', selection)
            select_run(wiki, current)
            self.assertEqual('defer', read(current / 'decisions.yaml')['decisions'][0]['decision'])

    def test_update_rebinds_only_validated_unchanged_group_and_own_verdict(self):
        fixture = fixtures.SelectionTests()
        with tempfile.TemporaryDirectory() as directory:
            wiki, baseline = fixture.prepare(Path(directory).resolve())
            write(baseline / 'selection.yaml', fixture.confirmed(baseline))
            select_run(wiki, baseline)
            request = read(baseline / 'review-request.yaml')
            write(baseline / 'decisions.yaml', fixture.reviewed(request))
            saved = read(baseline / 'request.yaml')
            repo = Path(saved['repo'])
            (repo / 'README.md').write_text('A source-independent documentation update.')
            commit(repo)
            current = import_repo.prepare(wiki, saved['repo'], 'HEAD', saved['config'])
            update = prepare_update(wiki, current, baseline)
            self.assertEqual(3, len(update['reused_selection']))
            select_run(wiki, current)
            reused = reuse_update_reviews(wiki, current)
            self.assertEqual(1, len(reused))
            document = read(current / 'decisions.yaml')
            self.assertEqual('include', document['decisions'][0]['decision'])
            self.assertEqual(digest(read(current / 'selection.yaml')), document['selection_digest'])
            self.assertNotEqual(request['snapshot']['commit'], read(current / 'snapshot.yaml')['commit'])
            self.assertFalse((wiki / 'data/repo-import/registry.yaml').exists())

    def test_policy_change_blocks_old_admission_reuse(self):
        fixture = fixtures.SelectionTests()
        with tempfile.TemporaryDirectory() as directory:
            wiki, baseline = fixture.prepare(Path(directory).resolve())
            (wiki / 'CLAUDE.md').write_text('Original scope.')
            write(baseline / 'selection.yaml', fixture.confirmed(baseline))
            select_run(wiki, baseline)
            write(baseline / 'decisions.yaml', fixture.reviewed(read(baseline / 'review-request.yaml')))
            saved = read(baseline / 'request.yaml')
            (wiki / 'CLAUDE.md').write_text('Changed scope.')
            current = import_repo.prepare(wiki, saved['repo'], 'HEAD', saved['config'])
            prepare_update(wiki, current, baseline)
            select_run(wiki, current)
            self.assertEqual([], reuse_update_reviews(wiki, current))
            self.assertEqual('defer', read(current / 'decisions.yaml')['decisions'][0]['decision'])
