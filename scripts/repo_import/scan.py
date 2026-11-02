"""Conservative entry extraction and repository-local dependency receipts."""
import ast
import fnmatch
import re
from pathlib import Path
from kernel_entries import kernel_entries, CUDA_EXTENSIONS
from .checkout import git
from .store import digest, file_hash, contained

EXTENSIONS = CUDA_EXTENSIONS | {'.py'}
EXTRACTOR_VERSION = 'repo-entries-v1'  # Stable encoding of individual entry receipts.
SCAN_CONTRACT = 'repo-scan-v5'  # Balanced CUDA signatures include braced default arguments.
ENTRY_KINDS = {'cuda_global', 'cute_kernel', 'triton_jit'}


def dependency_paths(path, root, tracked):
    text = path.read_text(errors='replace')
    found, unresolved = set(), []
    tokens = re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.MULTILINE)
    if path.suffix == '.py':
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = (node.module or '').replace('.', '/')
                    base = path.parent
                    for _ in range(max(0, node.level - 1)):
                        base = base.parent
                    candidates = ([base / (module + '.py'), base / module / '__init__.py'] if node.level else
                                  [root / (module + '.py'), root / module / '__init__.py'])
                    if not module:
                        candidates = [base / (alias.name + '.py') for alias in node.names]
                    matches = [p for p in candidates if p.resolve().is_relative_to(root.resolve()) and p.relative_to(root).as_posix() in tracked]
                    found.update(matches)
                    if not matches:
                        unresolved.append('import:' + '.' * node.level + (node.module or ''))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.replace('.', '/')
                        options = [root / (module + '.py'), root / module / '__init__.py']
                        matches = [p for p in options if p.relative_to(root).as_posix() in tracked]
                        found.update(matches)
                        if not matches:
                            unresolved.append('import:' + alias.name)
        except SyntaxError:
            unresolved.append('python-parse-error')
    for token in tokens:
        options = [path.parent / token, root / token, root / 'include' / token]
        matches = [p for p in options if p.resolve().is_relative_to(root.resolve()) and p.relative_to(root).as_posix() in tracked]
        found.update(matches)
        if not matches:
            unresolved.append('include:' + token)
    return found, unresolved


def scan_snapshot(snapshot, config):
    enabled = config.get('entry_kinds', sorted(ENTRY_KINDS))
    if not isinstance(enabled, list) or any(kind not in ENTRY_KINDS for kind in enabled):
        raise ValueError('scan.entry_kinds must be a list of supported entry kinds')
    root = Path(snapshot['checkout'])
    tracked = set(git(root, 'ls-files', '-z').split('\0')) - {''}
    skipped = {item['path'] for item in snapshot.get('skipped_submodules', [])}
    tracked = {path for path in tracked if not any(path == child or path.startswith(child + '/') for child in skipped)}
    records, failures = [], []
    # The checkout is pinned for the whole scan. Shared dependency modules and
    # their hashes can be reused within this scan, never across source snapshots.
    dependency_cache, hash_cache = {}, {}
    def source_hash(path):
        if path not in hash_cache:
            hash_cache[path] = file_hash(path)
        return hash_cache[path]
    for relative in sorted(tracked):
        path = contained(root, relative)
        if path.is_symlink() or not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in config.get('include', ['*'])):
            continue
        if any(fnmatch.fnmatch(relative, pattern) for pattern in config.get('exclude', [])):
            continue
        text = path.read_text(errors='replace')
        if path.suffix == '.py':
            try:
                ast.parse(text)
            except SyntaxError as error:
                failures.append({'path': relative, 'error': str(error)})
                continue
        entries = [entry for entry in kernel_entries(path, text) if entry.kind in enabled]
        if not entries:
            continue
        visited, queue, unresolved = set(), [path], set()
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            if current not in dependency_cache:
                dependency_cache[current] = dependency_paths(current, root, tracked)
            dependencies, unknown = dependency_cache[current]
            unresolved.update(unknown)
            queue.extend(dependencies - visited)
        receipts = [{'path': p.relative_to(root).as_posix(), 'sha256': source_hash(p)} for p in sorted(visited)]
        # Full defining file is bound too: helpers, templates, signatures and macros
        # in the same translation unit cannot silently reuse a body-only decision.
        fingerprint = {'source_sha256': source_hash(path), 'dependencies_sha256': digest(receipts),
                       'extraction_version': EXTRACTOR_VERSION}
        for entry in entries:
            candidate = {'origin': {k: snapshot[k] for k in ('repo_id', 'repo', 'commit')},
                         'path': relative, 'entry': entry.name, 'kind': entry.kind, 'line': entry.line,
                         'fingerprints': {**fingerprint, 'entry_sha256': entry.body_sha256},
                         'dependencies': receipts, 'unresolved_dependencies': sorted(unresolved)}
            candidate['candidate_id'] = 'candidate-' + digest(candidate)[:24]
            candidate['content_digest'] = digest({k: candidate[k] for k in ('path', 'entry', 'kind', 'fingerprints', 'unresolved_dependencies')})
            records.append(candidate)
    return {'candidates': records, 'failures': failures}
