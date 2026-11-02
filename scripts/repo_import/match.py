"""Conservative overlap hints across registry, wiki/PR evidence and a batch."""
import re
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from kernel_entries import kernel_entries, CUDA_EXTENSIONS
from .store import digest, file_hash, read
from .checkout import canonical_repo


def registry(root):
    return read(root / 'data/repo-import/registry.yaml', {'version': 1, 'implementations': {}})


@lru_cache(maxsize=512)
def code_entries(path, sha256):
    source = Path(path)
    return [entry.manifest_record() for entry in kernel_entries(source, source.read_text(errors='replace'))]


def existing_index(root):
    from _yaml_compat import yaml
    root = root.resolve()
    result = []
    for directory, kind in [('wiki/kernels', 'kernel'), ('sources/prs', 'pr')]:
        for path in sorted((root / directory).rglob('*.md')):
            match = re.match(r'^---\s*\n(.*?)\n---', path.read_text(), re.DOTALL)
            if not match:
                continue
            metadata = yaml.safe_load(match.group(1))
            assets = []
            asset_dir = metadata.get('artifact_dir')
            if not asset_dir and kind == 'pr':
                relative = path.relative_to(root / 'sources/prs').with_suffix('')
                asset_dir = 'artifacts/prs/' + relative.as_posix()
            if asset_dir:
                from .store import contained
                base = contained(root, asset_dir)
                for source in sorted(base.rglob('*')):
                    if source.is_file() and not source.is_symlink() and source.suffix in CUDA_EXTENSIONS | {'.py'}:
                        sha = file_hash(source)
                        assets.append({'path': source.relative_to(root).as_posix(), 'sha256': sha,
                                       'entries': code_entries(str(source), sha)})
            result.append({'page_id': metadata['id'], 'path': path.relative_to(root).as_posix(),
                           'sha256': file_hash(path), 'title': metadata.get('title', ''), 'kind': kind,
                           'repo': metadata.get('repo', ''), 'sources': metadata.get('sources', []),
                           'kernel_types': metadata.get('kernel_types', []),
                           'languages': metadata.get('languages', []), 'changed_paths': metadata.get('changed_paths', []),
                           'code_assets': assets})
    for page in result:
        if page['kind'] == 'pr':
            page['kernel_pages'] = [p['page_id'] for p in result if p['kind'] == 'kernel' and page['page_id'] in p['sources']]
    return result


def repo_name(repo):
    return canonical_repo(repo).rstrip('/').split('/')[-1].lower()


def meaningful_source_stem(path):
    stem = re.sub(r'[^a-z0-9]+', ' ', Path(path).stem.lower()).strip()
    return stem not in {'kernel', 'kernels', 'utils', 'main', 'input', 'cute', 'default'}


def find_matches(candidate, registered):
    matches = []
    for identity, implementation in registered.get('implementations', {}).items():
        for version_id, version in implementation.get('versions', {}).items():
            old = version.get('candidate')
            if old is None:
                continue
            relation = None
            if candidate['candidate_id'] == old['candidate_id']:
                relation = 'same_snapshot'
            elif (candidate['fingerprints'] == old['fingerprints'] and candidate['entry'] == old['entry']
                  and not candidate['unresolved_dependencies'] and not old['unresolved_dependencies']):
                relation = 'exact_content'
            elif (candidate['origin']['repo_id'] == old['origin']['repo_id'] and
                  (candidate['path'], candidate['entry']) == (old['path'], old['entry'])):
                relation = 'possible_new_version'
            elif candidate['fingerprints']['entry_sha256'] == old['fingerprints']['entry_sha256']:
                relation = 'similar_body'
            if relation:
                matches.append({'implementation_id': identity, 'version_id': version_id,
                                'page_id': implementation['page_id'], 'relation': relation})
        if not any(match['implementation_id'] == identity for match in matches) and any(
                candidate['content_digest'] == record['content_digest'] for record in implementation.get('records', [])):
            matches.append({'implementation_id': identity, 'version_id': implementation['current_version'],
                            'page_id': implementation['page_id'], 'relation': 'previously_reviewed_content'})
    return matches


def overlap_index(candidates, registered, pages, owners=None, discoveries=None):
    """Hints are review blockers, not claims of semantic equivalence."""
    result = {c['candidate_id']: [] for c in candidates}
    page_map = {p['page_id']: p for p in pages}
    registered_pages = {item['page_id'] for item in registered['implementations'].values()}
    managed_pages = {item['page_id'] for item in registered['implementations'].values()
                     if item.get('managed', True)}
    def append(candidate, hint):
        hint['match_id'] = 'overlap-' + digest({'candidate': candidate['candidate_id'], **hint})[:24]
        result[candidate['candidate_id']].append(hint)
    buckets = defaultdict(set)
    def tuning_base(name):
        return re.sub(r'_(?:tile|block)?\d+$', '', name)
    def batch_keys(c):
        name = c['entry']
        base = tuning_base(name)
        keys = [('body', c['fingerprints']['entry_sha256']), ('file', c['path'])]
        if name in {'kernel', 'main'}:
            if meaningful_source_stem(c['path']):
                keys.append(('generic_entry_file', (name, Path(c['path']).name)))
        else:
            keys.append(('entry', name))
        if base not in {'kernel', 'main'} or base != name:
            keys.append(('tuned', base))
        return keys
    for index, item in enumerate(candidates):
        for key in batch_keys(item):
            buckets[key].add(index)
    for candidate in candidates:
        for match in find_matches(candidate, registered):
            page = page_map.get(match['page_id'])
            if page:
                append(candidate, {**match, 'kind': 'registry', 'evidence': {
                    'path': page['path'], 'sha256': page['sha256'], 'locator': page['page_id']}})
        stem = Path(candidate['path']).stem.lower()
        meaningful_stem = meaningful_source_stem(candidate['path'])
        for page in pages:
            if page['page_id'] in registered_pages and (any(hint.get('page_id') == page['page_id'] for hint in result[candidate['candidate_id']]) or
                    page['page_id'] in managed_pages):
                continue
            same_repo = bool(page['repo']) and repo_name(page['repo']) == repo_name(candidate['origin']['repo'])
            matching = [asset for asset in page['code_assets'] if
                        asset['sha256'] == candidate['fingerprints']['source_sha256'] or
                        any(entry['body_sha256'] == candidate['fingerprints']['entry_sha256'] for entry in asset['entries']) or
                        (same_repo and (asset['path'] == candidate['path'] or
                         asset['path'].partition('/key-files/')[2] == candidate['path'] or
                         (meaningful_stem and Path(asset['path']).name == Path(candidate['path']).name)))]
            name = repo_name(candidate['origin']['repo'])
            normalized_stem = re.sub(r'[^a-z0-9]+', ' ', stem).strip()
            normalized_title = re.sub(r'[^a-z0-9]+', ' ', page['title'].lower()).strip()
            title_hint = meaningful_stem and normalized_stem in normalized_title
            path_hint = same_repo and any(
                path == candidate['path'] or (meaningful_stem and Path(path).name == Path(candidate['path']).name)
                for path in page.get('changed_paths', []))
            source_hint = len(name) >= 4 and (name in page['title'].lower() or name in ' '.join(page['sources']).lower())
            operation_context = re.sub(r'[^a-z0-9]+', ' ', candidate['path'].lower() + ' ' + candidate['entry'].lower())
            related_operation = any(
                re.sub(r'[^a-z0-9]+', ' ', kind.lower()).strip() in operation_context
                for kind in page.get('kernel_types', []) if kind and kind != 'fused-kernel')
            if matching or (same_repo and (path_hint or title_hint)) or (page['kind'] == 'kernel' and (
                    title_hint or (source_hint and related_operation))):
                evidence = matching[0] if matching else page
                append(candidate, {'kind': page['kind'], 'page_id': page['page_id'],
                                   'kernel_pages': page.get('kernel_pages', []),
                                   'relation': 'possible_existing_implementation',
                                   'evidence': {'path': evidence['path'], 'sha256': evidence['sha256'], 'locator': page['page_id']}})
        for discovery in (discoveries or {}).get('entries', []):
            if 'target_candidate_id' in discovery:
                source, target = discovery['candidate_id'], discovery['target_candidate_id']
                if candidate['candidate_id'] not in {source, target}:
                    continue
                other = target if candidate['candidate_id'] == source else source
                append(candidate, {'kind': 'batch', 'candidate_id': other,
                                   'relation': 'possible_batch_overlap',
                                   'evidence': discovery['matched_evidence' if other == target else 'input_evidence'][0],
                                   'discovered_overlap': discovery})
                continue
            if discovery['candidate_id'] != candidate['candidate_id']:
                continue
            page = page_map[discovery['page_id']]
            # Independent requests remain receipt-bound even if a heuristic also matches.
            append(candidate, {'kind': page['kind'], 'page_id': page['page_id'],
                               'kernel_pages': page.get('kernel_pages', []),
                               'relation': 'possible_existing_implementation',
                               'evidence': discovery['matched_evidence'][0],
                               'discovered_overlap': discovery})
        # A multi-implementation PR's source links are navigation, not identity.
        # Expose every linked canonical page for a source-bound comparison.
        for pr_hint in [hint for hint in result[candidate['candidate_id']] if hint['kind'] == 'pr']:
            for page_id in pr_hint.get('kernel_pages', []):
                page = page_map.get(page_id)
                if page is None or any(hint.get('page_id') == page_id for hint in result[candidate['candidate_id']]):
                    continue
                append(candidate, {'kind': 'kernel', 'page_id': page_id,
                                   'relation': 'possible_existing_implementation',
                                   'evidence': {'path': page['path'], 'sha256': page['sha256'], 'locator': page_id}})
        neighbors = set().union(*(buckets[key] for key in batch_keys(candidate)))
        for index in sorted(neighbors):
            other = candidates[index]
            if (other['candidate_id'] == candidate['candidate_id'] or
                    (owners is not None and owners[other['candidate_id']] == owners[candidate['candidate_id']])):
                continue
            same_body = other['fingerprints']['entry_sha256'] == candidate['fingerprints']['entry_sha256']
            same_entry = (other['entry'] == candidate['entry'] and
                          ((meaningful_stem and Path(other['path']).name == Path(candidate['path']).name) or
                           other['entry'] not in {'kernel', 'main'}))
            same_file = other['path'] == candidate['path']
            base, other_base = tuning_base(candidate['entry']), tuning_base(other['entry'])
            tuned_name = base == other_base and (base not in {'kernel', 'main'} or
                         (candidate['entry'] != base and other['entry'] != other_base))
            if same_body or same_entry or same_file or tuned_name:
                append(candidate, {'kind': 'batch', 'candidate_id': other['candidate_id'],
                                   'relation': 'possible_batch_overlap', 'evidence': {
                                       'path': other['path'], 'sha256': other['fingerprints']['source_sha256'],
                                       'locator': f"{other['entry']} at line {other['line']}"}})
    return result


def context_digest(index, registered):
    return digest({'pages': index, 'registry': registered})
