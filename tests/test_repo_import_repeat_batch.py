import copy
import tempfile
import unittest
from pathlib import Path

from repo_import_support import (prepared, commit, load_config, import_repo,
                                 admission, resolve_review, read, write)
from repo_import.decisions import rebind_manifest, validate_decisions
from repo_import.publish import apply_plan, markdown
from repo_import.store import digest, file_hash


class RepeatedBatchTests(unittest.TestCase):
    def published_pair(self, base, external=False):
        wiki, repo, _, _ = prepared(base)
        (repo / 'kernel.py').write_text('import triton\n\nclass Add:\n    @triton.jit\n'
            '    def kernel(x, n: int):\n        return x + n\n\nclass Multiply:\n    @triton.jit\n'
            '    def kernel(x, n: int):\n        return x * n\n')
        commit(repo)
        if external:
            (wiki / 'wiki/kernels/external.md').write_text(markdown(dict(
                id='kernel-external', title='Existing separate implementation', repo=repo.as_uri(),
                changed_paths=['kernel.py'], sources=[], reproducibility='snippet'),
                '# Existing implementation with a different contract\n'))
        run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
        request = read(run / 'review-request.yaml')
        decisions = []
        for index in range(2):
            decision = resolve_review(request, admission(request, index), 'distinct')
            decision['metadata']['title'] += ' ' + str(index)
            decisions.append(decision)
        manifest = dict(version=1, context_digest=request['context_digest'], decisions=decisions)
        write(run / 'decisions.yaml', manifest)
        plan = import_repo.build(wiki, run)
        plan['validated_digest'] = digest(plan['files'])
        apply_plan(wiki, run, plan)
        return wiki, repo, manifest

    def prepare_again(self, wiki, repo):
        run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
        return run, read(run / 'review-request.yaml')

    def validate(self, wiki, request, document):
        return validate_decisions(document, request, wiki, Path(request['snapshot']['checkout']))

    def test_unchanged_published_pair_reuses_reciprocal_distinctions(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, manifest = self.published_pair(Path(directory))
            run, request = self.prepare_again(wiki, repo)
            for document in [read(run / 'decisions.yaml'), rebind_manifest(manifest, request)]:
                selected, pending = self.validate(wiki, request, document)
                self.assertEqual((2, []), (len(selected), pending))
                self.assertTrue(all(d['action'] == 'record_duplicate' for _, d in selected))
                self.assertTrue(all(any(row.get('prior_batch_match_id') for row in d['overlap_resolutions'])
                                    for _, d in selected))
            write(run / 'decisions.yaml', rebind_manifest(manifest, request))
            plan = import_repo.build(wiki, run)
            plan['validated_digest'] = digest(plan['files'])
            apply_plan(wiki, run, plan)
            registered = read(wiki / 'data/repo-import/registry.yaml')['implementations']
            self.assertEqual(2, len(registered))
            self.assertTrue(all(len(item['versions']) == 1 for item in registered.values()))
            self.assertTrue(all(item['page_receipt']['sha256'] == file_hash(wiki / item['page_path'])
                                for item in registered.values()))

    def test_changed_page_or_missing_publication_receipt_keeps_peer_pending(self):
        for mutation in ['changed-page', 'missing-receipt']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                wiki, repo, manifest = self.published_pair(Path(directory))
                path = wiki / 'data/repo-import/registry.yaml'
                registry = read(path)
                target = next(iter(registry['implementations'].values()))
                if mutation == 'changed-page':
                    page = wiki / target['page_path']
                    page.write_text(page.read_text() + '\nExternally changed implementation explanation.\n')
                else:
                    target.pop('page_receipt')
                    write(path, registry)
                _, request = self.prepare_again(wiki, repo)
                _, pending = self.validate(wiki, request, rebind_manifest(manifest, request))
                self.assertTrue(pending)

    def test_changed_target_source_or_missing_reciprocal_verdict_keeps_pending(self):
        for mutation in ['changed-source', 'missing-reciprocal']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                wiki, repo, manifest = self.published_pair(Path(directory))
                path = wiki / 'data/repo-import/registry.yaml'
                registry = read(path)
                target = next(iter(registry['implementations'].values()))
                version = target['versions'][target['current_version']]
                if mutation == 'changed-source':
                    version['candidate']['content_digest'] = '0' * 64
                else:
                    version['decision']['overlap_resolutions'] = []
                write(path, registry)
                _, request = self.prepare_again(wiki, repo)
                _, pending = self.validate(wiki, request, rebind_manifest(manifest, request))
                self.assertTrue(pending)

    def test_no_prior_distinction_does_not_resolve_new_registry_match(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, manifest = self.published_pair(Path(directory))
            _, request = self.prepare_again(wiki, repo)
            altered = copy.deepcopy(request)
            for item in altered['registry']['implementations'].values():
                item['versions'][item['current_version']]['decision']['overlap_resolutions'] = []
            document = rebind_manifest(manifest, altered)
            _, pending = self.validate(wiki, altered, document)
            self.assertEqual(2, len(pending))

    def test_existing_page_distinction_reuses_only_unchanged_matched_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, manifest = self.published_pair(Path(directory), external=True)
            _, request = self.prepare_again(wiki, repo)
            self.assertTrue(all(any(h['kind'] == 'kernel' and h['page_id'] == 'kernel-external'
                                    for h in request['matches'][candidate['candidate_id']])
                                for candidate in request['candidates']))
            selected, pending = self.validate(wiki, request, rebind_manifest(manifest, request))
            self.assertEqual((2, []), (len(selected), pending))
            page = wiki / 'wiki/kernels/external.md'
            page.write_text(page.read_text() + '\nChanged algorithm evidence.\n')
            _, changed = self.prepare_again(wiki, repo)
            _, pending = self.validate(wiki, changed, rebind_manifest(manifest, changed))
            self.assertEqual(2, len(pending))


if __name__ == '__main__':
    unittest.main()
