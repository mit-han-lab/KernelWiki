import copy
import tempfile
import unittest
from pathlib import Path
from repo_import_support import prepared, registered
from repo_import.match import find_matches, overlap_index


class MatchTests(unittest.TestCase):
    def test_generic_module_names_do_not_imply_overlap(self):
        empty = {'implementations': {}}
        for filename in ('_cute.py', '_kernels.py', '_utils.py', 'kernel.py', 'main.py', 'default.cu'):
            with self.subTest(filename=filename):
                candidate = {'candidate_id': 'a', 'origin': {'repo': 'https://github.com/owner/project'},
                             'path': 'first/' + filename, 'entry': 'kernel', 'line': 1,
                             'fingerprints': {'source_sha256': 'source-a', 'entry_sha256': 'body-a'}}
                other = copy.deepcopy(candidate)
                other.update(candidate_id='b', path='second/' + filename)
                other['fingerprints'] = {'source_sha256': 'source-b', 'entry_sha256': 'body-b'}
                page = {'page_id': 'pr-project-1', 'path': 'sources/prs/project/PR-1.md',
                        'sha256': 'page', 'kind': 'pr', 'repo': 'owner/project',
                        'title': 'Optimize CuTe kernels and utils in main default kernel', 'sources': [],
                        'changed_paths': ['second/' + filename],
                        'code_assets': [{'path': 'artifacts/prs/project/1/key-files/second/' + filename,
                                         'sha256': 'source-b', 'entries': [{'body_sha256': 'body-b'}]}]}
                self.assertEqual({'a': [], 'b': []}, overlap_index([candidate, other], empty, []))
                self.assertEqual([], overlap_index([candidate], empty, [page])['a'])
                title_only = {**page, 'changed_paths': ['unrelated/gemm.cu'], 'code_assets': []}
                self.assertEqual([], overlap_index([candidate], empty, [title_only])['a'])
                for reason in ('exact_path', 'exact_asset_path', 'exact_body', 'exact_source'):
                    related = copy.deepcopy(page)
                    if reason == 'exact_path': related['changed_paths'] = [candidate['path']]
                    if reason == 'exact_asset_path':
                        related['code_assets'][0]['path'] = 'artifacts/prs/project/1/key-files/' + candidate['path']
                    if reason == 'exact_body': related['code_assets'][0]['entries'][0]['body_sha256'] = 'body-a'
                    if reason == 'exact_source': related['code_assets'][0]['sha256'] = 'source-a'
                    self.assertEqual(1, len(overlap_index([candidate], empty, [related])['a']), reason)
                for reason in ('exact_path', 'exact_body'):
                    related = copy.deepcopy(other)
                    if reason == 'exact_path': related['path'] = candidate['path']
                    if reason == 'exact_body': related['fingerprints']['entry_sha256'] = 'body-a'
                    self.assertEqual(1, len(overlap_index([candidate, related], empty, [])['a']), reason)

    def test_same_snapshot_and_dependency_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, request = prepared(Path(directory))
            candidate = copy.deepcopy(request['candidates'][0])
            self.assertEqual('same_snapshot', find_matches(candidate, registered(request))[0]['relation'])
            candidate['candidate_id'] = 'new'
            candidate['origin']['repo_id'] = 'fork'
            candidate['unresolved_dependencies'] = []
            candidate['fingerprints']['dependencies_sha256'] = 'different'
            self.assertEqual('similar_body', find_matches(candidate, registered(request))[0]['relation'])

    def test_same_name_different_body_not_cross_repo_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, request = prepared(Path(directory))
            candidate = copy.deepcopy(request['candidates'][0])
            candidate['candidate_id'] = 'new'
            candidate['origin']['repo_id'] = 'fork'
            candidate['fingerprints']['entry_sha256'] = 'different'
            self.assertEqual([], find_matches(candidate, registered(request)))

    def test_generic_entry_names_are_not_tuning_evidence(self):
        def candidate(identity, path, name, body):
            return {'candidate_id': identity, 'path': path, 'entry': name, 'line': 1,
                    'fingerprints': {'source_sha256': identity, 'entry_sha256': body}}
        first = candidate('a', 'a.py', 'kernel', 'body-a')
        second = candidate('b', 'b.py', 'kernel', 'body-b')
        empty = {'implementations': {}}
        self.assertEqual({'a': [], 'b': []}, overlap_index([first, second], empty, []))
        for reason in ('same_body', 'same_file', 'same_basename', 'actual_tuning_suffix'):
            with self.subTest(reason=reason):
                left, right = copy.deepcopy(first), copy.deepcopy(second)
                if reason == 'same_body': right['fingerprints']['entry_sha256'] = 'body-a'
                if reason == 'same_file': right['path'] = 'a.py'
                if reason == 'same_basename': right['path'] = 'other/a.py'
                if reason == 'actual_tuning_suffix':
                    left['entry'], right['entry'] = 'kernel_128', 'kernel_256'
                matches = overlap_index([left, right], empty, [])
                self.assertEqual(1, len(matches['a']))
                self.assertEqual(1, len(matches['b']))

    def test_same_repository_alone_does_not_require_every_pr_comparison(self):
        candidate = {'candidate_id': 'candidate-a', 'origin': {'repo': 'https://github.com/owner/project'},
                     'path': 'csrc/rotary_embedding.py', 'entry': 'rotate_kernel', 'line': 1,
                     'fingerprints': {'source_sha256': 'source', 'entry_sha256': 'body'}}
        page = {'page_id': 'pr-project-1', 'path': 'sources/prs/project/PR-1.md', 'sha256': 'page',
                'kind': 'pr', 'repo': 'owner/project', 'title': 'Unrelated attention update',
                'sources': [], 'code_assets': [], 'changed_paths': ['csrc/attention.py']}
        empty = {'implementations': {}}
        self.assertEqual([], overlap_index([candidate], empty, [page])['candidate-a'])
        canonical = {**page, 'kind': 'kernel', 'repo': '', 'sources': ['pr-project-1'],
                     'kernel_types': ['attention']}
        self.assertEqual([], overlap_index([candidate], empty, [canonical])['candidate-a'])
        canonical['kernel_types'] = ['rotary-embedding']
        self.assertEqual(1, len(overlap_index([candidate], empty, [canonical])['candidate-a']))
        for reason in ('same_path', 'renamed_directory', 'normalized_title', 'matching_body'):
            with self.subTest(reason=reason):
                related = copy.deepcopy(page)
                if reason == 'same_path': related['changed_paths'] = ['csrc/rotary_embedding.py']
                if reason == 'renamed_directory': related['changed_paths'] = ['old/rotary_embedding.py']
                if reason == 'normalized_title': related['title'] = 'Optimize rotary embedding'
                if reason == 'matching_body':
                    related['code_assets'] = [{'path': 'artifacts/old.py', 'sha256': 'asset',
                                               'entries': [{'body_sha256': 'body'}]}]
                hints = overlap_index([candidate], empty, [related])['candidate-a']
                self.assertEqual(1, len(hints))
                self.assertEqual('pr-project-1', hints[0]['page_id'])
