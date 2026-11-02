import copy
import io
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from repo_import_support import digest, read, write
from repo_import.decisions import rebind_manifest
from repo_import.review_archive import (Relocator, compact, export_reviews, file_hash,
                                        group_identity, pack_archive, state_check,
                                        verify_archive, write_record)


class ReviewArchiveTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'wiki'
        batch = root / '.cache/batch'
        run = batch / 'run'
        checkout = root / '.cache/checkout'
        checkout.mkdir(parents=True)
        (checkout / 'kernel.py').write_text('# Public test fixture only\n')
        note = batch / 'comparison.yaml'
        write(note, {'observation': 'Fixture source ' + str(checkout / 'kernel.py')})
        repo, commit = 'https://github.com/example/fixture', 'a' * 40
        inventory = {'candidates': [dict(candidate_id=key, content_digest=key * 64,
                       path='kernel.py', entry=key) for key in 'abcde']}
        selection = {'inventory_digest': digest(inventory), 'entries': []}
        for key, role in zip('abcde', ['representative', 'representative', 'pending', 'member', 'helper']):
            row = dict(candidate_id=key, content_digest=key * 64, role=role,
                       rationale='Fixture source comparison supports this status.')
            if role == 'member':
                row.update(representative='a', difference_summary='Ordinary fixture specialization.')
            selection['entries'].append(row)
        decisions = {'decisions': [dict(candidate_id=key, content_digest=key * 64,
                     decision=status, rationale='Fixture reviewed; unresolved evidence stays held.',
                     metadata={'confidence': 'experimental', 'reproducibility': 'snippet'},
                     evidence=[{'path': str(note), 'sha256': file_hash(note)},
                               {'path': str(checkout / 'kernel.py'), 'sha256': file_hash(checkout / 'kernel.py')}])
                     for key, status in [('a', 'include'), ('b', 'defer')]]}
        matches = {}
        request = dict(context_digest='context', matches=matches, existing_pages=[], registry={})
        summary = dict(definitions=5, representatives=2, members=1, helpers=1,
                       selection_pending=1, reviewed_include=1, reviewed_exclude=0,
                       review_defer=1, effective_kernels=1, included_definitions=2,
                       pending_definitions=2, selection_defer=1, review_unfinished=0,
                       selection_unfinished=0, blocked_include=0, scan_config={},
                       request_digest=digest(request), decisions_digest=digest(decisions))
        valid = {'summary': summary, 'implementations': [dict(
            implementation_id=group_identity(repo, 'a', decisions['decisions'][0]),
            representatives=[dict(inventory['candidates'][0], metadata=decisions['decisions'][0]['metadata'])],
            members=[inventory['candidates'][3]])]}
        for name, value in dict(snapshot=dict(repo=repo, repo_id='repo-fixture', commit=commit,
                               checkout=str(checkout)), inventory=inventory, selection=selection,
                               decisions=decisions, matches=matches, **{'review-request': request}).items():
            write(run / (name + '.yaml'), value)
        (run / 'review-guide.md').write_text('Review the fixed source; preserve unresolved statuses.\n')
        write(run / 'valid.yaml', valid)
        write(batch / 'manifest.yaml', {'created_at': '2026-10-16'})
        record = dict(name='Fixture', run=str(run), repo=repo, commit=commit, summary=summary,
                      valid_kernels=dict(path=str(run / 'valid.yaml'), sha256=file_hash(run / 'valid.yaml')))
        write(batch / 'final-review-acceptance.yaml', dict(completed_at='2026-10-28', repositories=[record]))
        return root, batch, root / 'examples/repo-import/reviews/fixture'

    def test_export_survives_removal_of_cache_and_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = export_reviews(root, batch, out)
            self.assertEqual((1, 1, 1, 2), tuple(result[k] for k in
                             ['effective_kernels', 'review_defer', 'selection_pending', 'pending_records']))
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            self.assertFalse((root / 'wiki').exists())
            self.assertFalse((root / 'data').exists())
            relocated = Path(directory) / 'relocated'
            shutil.copytree(out, relocated)
            shutil.rmtree(root)
            self.assertEqual(result, verify_archive(relocated))
            archived = read(relocated / 'fixture/decisions.yaml')
            self.assertTrue(archived['archive_only'])
            with self.assertRaisesRegex(ValueError, 'Read-only review archives'):
                rebind_manifest(archived, {})

    def test_tamper_missing_evidence_and_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            export_reviews(root, batch, out)
            index_path = out / 'index.yaml'
            original = read(index_path)
            index = copy.deepcopy(original)
            index['repositories'][0]['summary']['review_defer'] = 0
            write(index_path, index)
            with self.assertRaisesRegex(ValueError, 'Index summary'):
                verify_archive(out)
            write(index_path, original)
            decisions_path = out / 'fixture/decisions.yaml'
            decisions = read(decisions_path)
            decisions['decisions'][0]['evidence'][0]['path'] = 'archive://evidence/missing.yaml'
            write_record(decisions_path, decisions)
            original['files']['fixture/decisions.yaml'] = file_hash(decisions_path)
            write(index_path, original)
            with self.assertRaisesRegex(ValueError, 'portable evidence receipt'):
                verify_archive(out)

    def test_dropped_holds_and_promoted_confidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            run = batch / 'run'
            args = [read(run / (key + '.yaml')) for key in
                    ['inventory', 'selection', 'decisions', 'valid']]
            summary = args[3]['summary']
            repo = 'https://github.com/example/fixture'
            state_check(*args, summary, repo)
            for mutation in ['drop-review', 'drop-selection', 'unfinished', 'confidence', 'admit-hold']:
                with self.subTest(mutation=mutation):
                    inventory, selection, decisions, valid = copy.deepcopy(args)
                    if mutation == 'drop-review':
                        decisions['decisions'].pop()
                    elif mutation == 'drop-selection':
                        selection['entries'].pop(2)
                    elif mutation == 'unfinished':
                        decisions['decisions'][1]['rationale'] = 'Pending external review'
                    elif mutation == 'confidence':
                        valid['implementations'][0]['representatives'][0]['metadata']['confidence'] = 'verified'
                    else:
                        valid['implementations'][0]['representatives'].append(inventory['candidates'][1])
                    with self.assertRaises(ValueError):
                        state_check(inventory, selection, decisions, valid, summary, repo)

    def test_scope_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'isolated batch'):
                export_reviews(root, batch, root / 'wiki/kernels')
            export_reviews(root, batch, out)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                export_reviews(root, batch, out)

    def test_lossless_compaction_and_source_receipt_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'anchors.yaml'
            value = {'dependencies': [[{'path': 'shared.h', 'sha256': 'a' * 64}]] * 50}
            write_record(path, value)
            self.assertEqual(value, read(path))
            self.assertIn('*id', path.read_text())
            packed = compact(value)
            self.assertIs(packed['dependencies'][0], packed['dependencies'][1])
            root, batch, _ = self.fixture(directory)
            records = read(batch / 'final-review-acceptance.yaml')['repositories']
            relocator = Relocator(root, batch, records)
            with self.assertRaisesRegex(ValueError, 'Stale source receipt'):
                relocator.receipt({'path': str(batch / 'comparison.yaml'), 'sha256': '0' * 64})

    def test_pending_status_cannot_be_promoted_in_aggregate_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            export_reviews(root, batch, out)
            pending = read(out / 'pending.yaml')
            pending['entries'][0]['status'] = 'include'
            write_record(out / 'pending.yaml', pending)
            index = read(out / 'index.yaml')
            index['files']['pending.yaml'] = file_hash(out / 'pending.yaml')
            write_record(out / 'index.yaml', index)
            with self.assertRaisesRegex(ValueError, 'Held candidate statuses'):
                verify_archive(out)

    def test_compression_is_lossless_deterministic_and_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            original = export_reviews(root, batch, out)
            packed = Path(directory) / 'packed'
            second = Path(directory) / 'second'
            self.assertEqual(original, pack_archive(out, packed))
            pack_archive(out, second)
            self.assertEqual((packed / 'records.tar.gz').read_bytes(),
                             (second / 'records.tar.gz').read_bytes())
            shutil.rmtree(root)
            self.assertEqual(original, verify_archive(packed))
            with (packed / 'records.tar.gz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'compressed review archive'):
                verify_archive(packed)

    def test_compressed_members_cannot_escape_or_create_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root, batch, out = self.fixture(directory)
            export_reviews(root, batch, out)
            packed = Path(directory) / 'packed'
            pack_archive(out, packed)
            for name, kind in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
                with self.subTest(name=name):
                    with tarfile.open(packed / 'records.tar.gz', 'w:gz') as archive:
                        member = tarfile.TarInfo(name)
                        member.type = kind
                        member.linkname = '/outside'
                        archive.addfile(member, io.BytesIO())
                    manifest = read(packed / 'archive.yaml')
                    manifest['sha256'] = file_hash(packed / 'records.tar.gz')
                    write_record(packed / 'archive.yaml', manifest)
                    with self.assertRaises(ValueError):
                        verify_archive(packed)
                    self.assertFalse((Path(directory) / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
