"""Read committed Git objects into an isolated checkout."""
import re
import subprocess
from pathlib import Path
from .store import digest


def git(cwd, *args):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True, stderr=subprocess.PIPE).strip()


def canonical_repo(repo):
    value = str(repo).rstrip('/')
    match = re.match(r'git@([^:]+):(.+)', value)
    if match:
        value = f'https://{match[1]}/{match[2]}'
    return value.removesuffix('.git')


def resolve_snapshot(repo, revision, checkout_dir):
    if revision and revision.startswith('-'):
        raise ValueError('Revision cannot be a Git option')
    local = Path(repo).expanduser()
    if local.exists():
        source = local.resolve()
        if git(source, 'rev-parse', '--show-toplevel') != str(source):
            raise ValueError('Local input must be a Git repository root')
        try:
            origin = git(source, 'remote', 'get-url', 'origin')
        except subprocess.CalledProcessError:
            origin = source.as_uri()
        dirty = bool(git(source, 'status', '--porcelain'))
    else:
        origin, dirty = repo, False
        source = checkout_dir / ('mirror-' + digest(canonical_repo(repo))[:16])
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--mirror', '--no-recurse-submodules', '--', repo, str(source)], check=True)
        else:
            subprocess.run(['git', '-C', str(source), 'fetch', '--prune', 'origin'], check=True)
    commit = git(source, 'rev-parse', '--verify', (revision or 'HEAD') + '^{commit}')
    identity = canonical_repo(origin)
    repo_id = 'repo-' + digest(identity)[:16]
    destination = checkout_dir / repo_id / commit
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', '--no-recurse-submodules', '--', str(source), str(destination)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            git(destination, 'cat-file', '-e', commit + '^{tree}')
        except subprocess.CalledProcessError:
            subprocess.run(['git', '-C', str(destination), 'fetch', '--no-tags', str(source), commit], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--no-recurse-submodules', '--detach', commit], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if git(destination, 'rev-parse', 'HEAD') != commit or git(destination, 'status', '--porcelain'):
        raise ValueError('Cached pinned checkout was modified; use a fresh checkout directory')
    gitlinks = []
    for record in git(destination, 'ls-tree', '-rz', commit).split('\0'):
        if record:
            header, path = record.split('\t', 1)
            mode, _, child_commit = header.split()
            if mode == '160000':
                gitlinks.append({'path': path, 'commit': child_commit, 'reason': 'Submodules are not scanned'})
    return {'repo_id': repo_id, 'repo': identity, 'commit': commit,
            'checkout': str(destination), 'ignored_worktree_changes': dirty, 'skipped_submodules': gitlinks}
