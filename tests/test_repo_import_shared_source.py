"""A reviewed source cited by several existing pages still has one import target."""
import copy
import tempfile
import unittest
from pathlib import Path

from repo_import_support import prepared, admission, document, import_repo, load_config, read, write, resolve_review
from repo_import.decisions import validate_decisions
from repo_import.publish import markdown, apply_plan, registry_errors
from repo_import.store import file_hash, digest


class SharedSourceTests(unittest.TestCase):
    def fixture(self, base):
        wiki, repo, _, _ = prepared(base)
        pr = wiki / 'sources/prs/fixture/PR-1.md'
        pr.parent.mkdir(parents=True)
        pr.write_text(markdown({'id': 'pr-fixture-1', 'repo': 'owner/fixture',
                                'title': 'Reduction implementation'}, '# Source\n'))
        asset = wiki / 'artifacts/prs/fixture/PR-1/key-files/kernel.py'
        asset.parent.mkdir(parents=True)
        asset.write_text((repo / 'kernel.py').read_text())
        receipts = []
        for suffix in ('one', 'two'):
            path = 'wiki/kernels/' + suffix + '.md'
            page = wiki / path
            page.write_text(markdown({'id': 'kernel-' + suffix, 'title': 'Reduction ' + suffix,
                                      'sources': ['pr-fixture-1'], 'reproducibility': 'snippet'},
                                     '# Existing reduction\n'))
            receipts.append({'path': path, 'sha256': file_hash(page),
                             'locator': 'Retained source branch', 'anchor': 'pr-fixture-1'})
        run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
        request = read(run / 'review-request.yaml')
        decision = resolve_review(request, admission(request), 'same_implementation', 'kernel-one')
        decision.update(action='retain_page', relation='new_version', target_page_id='kernel-one',
                        selection_reason='Keep the existing reviewed representative.',
                        equivalence_basis='Both pages cite this exact reviewed implementation; retain one canonical attribution.',
                        target_evidence=receipts)
        return wiki, run, request, decision

    def validate(self, wiki, request, decision):
        return validate_decisions(document(request, decision), request, wiki,
                                  Path(request['snapshot']['checkout']))

    def test_shared_source_keeps_one_existing_page_and_preserves_both_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, request, decision = self.fixture(Path(directory))
            before = {e['path']: (wiki / e['path']).read_bytes() for e in decision['target_evidence']}
            selected, pending = self.validate(wiki, request, decision)
            self.assertEqual((1, []), (len(selected), pending))
            write(run / 'decisions.yaml', document(request, decision))
            plan = import_repo.build(wiki, run)
            self.assertEqual(1, read(run / 'result.yaml')['included'])
            self.assertFalse(any(f['path'].startswith('wiki/kernels/imported/') for f in plan['files']))
            self.assertNotIn('wiki/kernels/two.md', [f['path'] for f in plan['files']])
            self.assertIn('# Existing reduction', (run / 'staging/wiki/kernels/one.md').read_text())
            registry = read(run / 'staging/data/repo-import/registry.yaml')
            self.assertEqual(['kernel-one'], [i['page_id'] for i in registry['implementations'].values()])
            for path, content in before.items():
                self.assertEqual(content, (wiki / path).read_bytes())

    def test_shared_source_needs_explicit_anchors_for_both_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, request, decision = self.fixture(Path(directory))
            for missing in ('canonical', 'other', 'basis'):
                with self.subTest(missing=missing):
                    changed = copy.deepcopy(decision)
                    if missing == 'basis':
                        changed.pop('equivalence_basis')
                    else:
                        changed['target_evidence'][0 if missing == 'canonical' else 1].pop('anchor')
                    selected, pending = self.validate(wiki, request, changed)
                    self.assertEqual((0, 1), (len(selected), len(pending)))

    def test_common_pr_and_canonical_page_must_both_be_reviewed_as_same(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, request, decision = self.fixture(Path(directory))
            hints = request['matches'][decision['candidate_id']]
            for kind in ('pr', 'kernel'):
                with self.subTest(kind=kind):
                    changed = copy.deepcopy(decision)
                    hint = next(h for h in hints if h['kind'] == kind and
                                (kind == 'pr' or h['page_id'] == 'kernel-one'))
                    next(r for r in changed['overlap_resolutions'] if r['match_id'] == hint['match_id']).update(
                        outcome='distinct')
                    selected, pending = self.validate(wiki, request, changed)
                    self.assertEqual((0, 1), (len(selected), len(pending)))
            changed_request = copy.deepcopy(request)
            next(h for h in changed_request['matches'][decision['candidate_id']] if h['kind'] == 'pr')['kernel_pages'] = ['kernel-one']
            from repo_import.decisions import unresolved_overlaps
            self.assertTrue(unresolved_overlaps(decision, changed_request['matches'][decision['candidate_id']],
                                               {decision['candidate_id']: decision}, wiki,
                                               Path(request['snapshot']['checkout']), changed_request['matches']))

    def test_shared_source_rejects_invented_page_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, _, request, decision = self.fixture(Path(directory))
            decision['target_evidence'][1]['anchor'] = 'pr-fixture-1 missing exact source branch'
            with self.assertRaisesRegex(ValueError, 'anchor must occur literally'):
                self.validate(wiki, request, decision)

    def publish(self, wiki, run, request, decision):
        write(run / 'decisions.yaml', document(request, decision))
        plan = import_repo.build(wiki, run)
        plan['validated_digest'] = digest(plan['files'])
        apply_plan(wiki, run, plan)

    def test_unchanged_shared_source_attribution_reuses_registered_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, request, decision = self.fixture(Path(directory))
            self.publish(wiki, run, request, decision)
            self.assertEqual([], registry_errors(wiki))
            repeated = import_repo.prepare(wiki, str(Path(directory) / 'input'), 'HEAD', load_config(wiki))
            replay = read(repeated / 'decisions.yaml')['decisions'][0]
            self.assertEqual('retain_page', replay['action'])
            self.assertEqual('kernel-one', replay['target_page_id'])
            self.assertTrue(replay['target_implementation_id'])
            request = read(repeated / 'review-request.yaml')
            selected, pending = self.validate(wiki, request, replay)
            self.assertEqual((1, []), (len(selected), pending))

    def test_changed_other_page_requires_fresh_source_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, run, request, decision = self.fixture(Path(directory))
            self.publish(wiki, run, request, decision)
            page = wiki / 'wiki/kernels/two.md'
            page.write_text(page.read_text() + '\nChanged implementation evidence.\n')
            repeated = import_repo.prepare(wiki, str(Path(directory) / 'input'), 'HEAD', load_config(wiki))
            replay = read(repeated / 'decisions.yaml')['decisions'][0]
            self.assertEqual('Pending external review', replay['rationale'])
