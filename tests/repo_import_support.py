import copy
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import import_repo
from repo_import.config import load_config as base_load_config
from repo_import.store import read, write, digest

def load_config(root, config_path=None):
    config = base_load_config(root, config_path)
    config['selection'] = {'enabled': False}
    return config


CODE = 'import triton\n\n@triton.jit\ndef reduce_kernel(x, n: int):\n    return x + n\n'


def commit(repo, message='fixture'):
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, stdout=subprocess.PIPE)
    subprocess.run(['git', '-C', str(repo), 'commit', '-m', message], check=True, stdout=subprocess.PIPE)


def make_repo(root):
    repo = root / 'input'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'fixture@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Fixture'], check=True)
    (repo / 'kernel.py').write_text(CODE)
    commit(repo)
    return repo


def make_wiki(root):
    wiki = root / 'wiki-root'
    (wiki / 'wiki/kernels').mkdir(parents=True)
    (wiki / 'data').mkdir()
    write(wiki / 'data/repo-import.yaml', load_config(ROOT))
    write(wiki / 'data/repo-import-presets.yaml', {'version': 1, 'presets': {}})
    return wiki


def prepared(root):
    repo = make_repo(root)
    wiki = make_wiki(root)
    run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
    request = read(run / 'review-request.yaml')
    return wiki, repo, run, request


def admission(request, index=0):
    candidate = request['candidates'][index]
    return {'candidate_id': candidate['candidate_id'], 'content_digest': candidate['content_digest'],
            'decision': 'include', 'rationale': 'Fixture explicitly tests admission contracts, not real kernel correctness.',
            'evidence': [{'path': candidate['path'], 'sha256': candidate['fingerprints']['source_sha256'], 'locator': candidate['entry']}],
            'relation': 'distinct', 'action': 'create_page', 'confidence_basis': 'Experimental fixture implementation.',
            'metadata': {'title': 'Fixture reduction', 'architectures': [], 'tags': ['reduction'],
                         'kernel_types': ['reduction'], 'languages': ['triton'],
                         'confidence': 'experimental', 'reproducibility': 'snippet'}}


def document(request, decision=None):
    return {'version': 1, 'context_digest': request['context_digest'],
            'decisions': [decision or admission(request)]}


def resolve_review(request, decision, outcome='distinct', representative=None):
    decision['overlap_resolutions'] = [
        {'match_id': hint['match_id'], 'outcome': outcome, 'rationale': 'Fixture compares pinned source and matched evidence.',
         'evidence': [hint['evidence']], 'representative': representative or decision['candidate_id']}
        for hint in request['matches'][decision['candidate_id']]]
    return decision


def registered(request):
    candidate = request['candidates'][0]
    return {'version': 1, 'implementations': {'impl': {'page_id': 'kernel-test',
        'page_path': 'wiki/kernels/kernel-test.md', 'current_version': 'v1', 'origins': [],
        'versions': {'v1': {'candidate': candidate, 'decision': admission(request),
                           'metadata': admission(request)['metadata']}}}}}
