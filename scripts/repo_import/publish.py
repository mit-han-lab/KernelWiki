"""Stable rendering, offline validation and recoverable publication."""
import copy
import datetime
import json
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from _yaml_compat import yaml
from .store import atomic_write, contained, digest, file_hash, read, write
from .match import registry
from kernel_entries import kernel_snippet


def markdown(metadata, body):
    return '---\n' + yaml.safe_dump(metadata, sort_keys=False).strip() + '\n---\n\n' + body.rstrip() + '\n'


def compact_version(version):
    candidate = version['candidate']
    return {'content_digest': candidate['content_digest'], 'origin': candidate['origin'],
            'path': candidate['path'], 'entry': candidate['entry'],
            'relation': version.get('relation'), 'reason': version['decision'].get('selection_reason', version['decision']['rationale'])}


def add_record(implementation, record):
    records = implementation.setdefault('records', [])
    for existing in records:
        if all(existing[key] == record[key] for key in ('origin', 'path', 'entry', 'content_digest')):
            existing.update(record)
            return
    records.append(record)


def checked_text(value):
    return str(value).replace('`', '').replace('\n', ' ')


def member_section(implementation):
    """Render compact provenance without claiming unioned entry capabilities."""
    records = implementation.get('records', [])
    if not records:
        return ''
    def cell(value):
        return checked_text(value).replace('|', '&#124;')
    rows = ['<!-- repo-import-members:start -->', '## Related source entries', '',
            'Each entry retains its own parameter and dtype limits. These sources do not imply that the representative entry supports every variant.', '',
            '| Repository / commit | Path | Entry | Relationship | Difference / selection reason |',
            '|---|---|---|---|---|']
    for record in records:
        origin = record['origin']
        rows.append('| ' + ' | '.join(cell(value) for value in (
            str(origin.get('repo', origin.get('repo_id', ''))) + ' / ' + origin.get('commit', ''),
            record['path'], record['entry'], record['relation'], record['reason'])) + ' |')
    return '\n'.join(rows + ['<!-- repo-import-members:end -->'])


def compact_manifest(run, request, document, selected):
    """Formal publication evidence is not a replacement for the full audit pack."""
    admitted = {c['candidate_id'] for c, _ in selected}
    explicit = [d for d in document.get('decisions', [])
                if d.get('decision') == 'exclude' or d.get('candidate_id') in admitted]
    formal = {key: document[key] for key in ('version', 'context_digest', 'selection_digest') if key in document}
    formal['decisions'] = explicit
    formal['run_reference'] = {
        'run_id': request['run_id'], 'path': str(run.resolve()),
        'repo': request['snapshot']['repo'], 'commit': request['snapshot']['commit'],
        'request_digest': digest(request), 'decisions_digest': digest(document),
        'inventory_digest': request.get('inventory_digest', digest(read(run / 'inventory.yaml'))),
        'selection_digest': digest(request['selection']) if 'selection' in request else None}
    formal['members'] = [{key: row[key] for key in (
        'candidate_id', 'content_digest', 'path', 'entry', 'representative', 'relation',
        'difference_summary', 'rationale', 'evidence') if key in row}
        for row in request.get('selection', {}).get('entries', [])
        if row['role'] == 'member' and row['candidate_id'] in admitted]
    return formal


def build_plan(root, run, request, document, selected):
    registered = copy.deepcopy(request['registry'])
    snapshot = request['snapshot']
    source_id = 'repo-source-' + digest({k: snapshot[k] for k in ('repo', 'commit')})[:20]
    source_path = f'sources/repos/{source_id}.md'
    date = request['created_at'][:10]
    source = {'id': source_id, 'title': f"Repository snapshot: {snapshot['repo']}",
              'repo': snapshot['repo'], 'url': snapshot['repo'], 'commit': snapshot['commit'],
              'source_category': 'upstream-code', 'architectures': [], 'tags': [], 'retrieved_at': date}
    outputs = {}
    if selected:
        source_body = f"# {source['title']}\n\nFixed commit: `{snapshot['commit']}`.\n"
        for name in ('LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING'):
            notice = Path(snapshot['checkout']) / name
            if notice.is_file():
                relative = f"references/import-licenses/{snapshot['repo_id']}/{snapshot['commit']}/{name}"
                outputs[relative] = notice.read_text()
                source_body += f"\n[Upstream license notice](../../{relative}).\n"
                break
        outputs[source_path] = markdown(source, source_body)
    selected = sorted(selected, key=lambda item: item[1]['action'] == 'merge_candidate')
    canonical = {}
    for candidate, decision in selected:
        review_sources = set(decision.get('sources', []))
        for hint in request.get('matches', {}).get(candidate['candidate_id'], []):
            if hint['kind'] == 'pr' and any(r['match_id'] == hint['match_id'] and r['outcome'] == 'same_implementation'
                                          for r in decision.get('overlap_resolutions', [])):
                review_sources.add(hint['page_id'])
        identity = decision.get('target_implementation_id') or ('implementation-' + digest(candidate['candidate_id'])[:20])
        existing_page_target = decision['action'] in {'attach_source', 'retain_page'} and not decision.get('target_implementation_id')
        if existing_page_target:
            identity = 'implementation-' + digest({'existing_page': decision['target_page_id']})[:20]
        if decision['action'] == 'merge_candidate':
            identity = canonical[decision['target_candidate_id']]
        canonical[candidate['candidate_id']] = identity
        implementation = registered['implementations'].setdefault(identity, {
            'page_id': 'kernel-imported-' + identity.removeprefix('implementation-'),
            'versions': {}, 'origins': []})
        if existing_page_target and not implementation.get('current_version'):
            target = next(p for p in request['existing_pages'] if p['page_id'] == decision['target_page_id'])
            implementation['page_id'] = target['page_id']
            implementation['page_path'] = target['path']
        page_id = implementation['page_id']
        page_path = implementation.setdefault('page_path', f'wiki/kernels/imported/{page_id}.md')
        if existing_page_target and not implementation.get('current_version'):
            existing_text = contained(root, page_path).read_text()
            existing_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', existing_text, re.DOTALL)
            existing_metadata = yaml.safe_load(existing_match.group(1))
            existing_page_version = 'version-' + digest({'page': page_id, 'body': existing_match.group(2)})[:24]
            implementation.update(managed=False, current_version=existing_page_version,
                                  versions={existing_page_version: {'page_evidence': {'path': page_path}, 'metadata': existing_metadata}})
        # Same content may occur at different commits; retain every source origin.
        origin = {**candidate['origin'], 'path': candidate['path'], 'entry': candidate['entry']}
        if origin not in implementation['origins']:
            implementation['origins'].append(origin)
        version_id = 'version-' + digest({'implementation': identity, 'content': candidate['content_digest']})[:24]
        target_version = decision.get('target_version_id')
        if existing_page_target or decision['action'] in {'record_duplicate', 'retain_page', 'merge_candidate'}:
            version_id = target_version
            add_record(implementation, {
                    'origin': candidate['origin'], 'path': candidate['path'], 'entry': candidate['entry'],
                    'content_digest': candidate['content_digest'], 'relation': decision['relation'],
                    'representative': implementation['current_version'],
                    'reason': decision.get('selection_reason', decision['rationale']),
                    'evidence': decision.get('evidence', []),
                    **({'target_evidence': decision['target_evidence'],
                        'equivalence_basis': decision['equivalence_basis']}
                       if any('anchor' in e for e in decision.get('target_evidence', [])) else {})})
        else:
            version = {'candidate': candidate, 'metadata': decision['metadata'], 'decision': decision,
                       'source_id': source_id}
            if target_version and target_version != version_id:
                version['relation'] = {'kind': decision['relation'], 'target': target_version}
            history = implementation.setdefault('history', {})
            if version_id in history:
                original_relation = history.pop(version_id).get('relation')
                if original_relation:
                    version['relation'] = original_relation
                else:
                    version.pop('relation', None)
            history.update({key: compact_version(value) for key, value in implementation['versions'].items() if key != version_id})
            implementation['versions'] = {version_id: version}
            implementation['current_version'] = version_id
        if decision['action'] in {'record_duplicate', 'attach_source', 'retain_page', 'merge_candidate'}:
            # Sources extend attribution, never replace existing prose or metrics.
            text = outputs.get(page_path) or contained(root, page_path).read_text()
            match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
            metadata = yaml.safe_load(match.group(1))
            metadata['sources'] = sorted(set(metadata['sources']) | {source_id} | review_sources)
            metadata['import_record'] = identity
            if decision['action'] == 'attach_source':
                implementation['managed'] = False
            implementation['page_reproducibility'] = metadata['reproducibility']
            outputs[page_path] = markdown(metadata, match.group(2))
            representative = implementation['versions'][implementation['current_version']]
            if 'page_evidence' in representative:
                representative['page_evidence']['sha256'] = hashlib.sha256(outputs[page_path].encode()).hexdigest()
                representative['metadata'] = metadata
            continue
        metadata = {**decision['metadata'], 'id': page_id, 'type': 'kernel', 'repo': snapshot['repo'],
                    'related': decision.get('related', []), 'sources': sorted(review_sources | {source_id}),
                    'performance_claims': copy.deepcopy(decision.get('performance_claims', [])), 'import_record': identity}
        if decision.get('evidence_basis'):
            metadata['evidence_basis'] = decision['evidence_basis']
        if decision.get('blackwell_relevance'):
            metadata['blackwell_relevance'] = decision['blackwell_relevance']
        for claim in metadata['performance_claims']:
            claim.setdefault('source_id', source_id)
        code = kernel_snippet(contained(Path(snapshot['checkout']), candidate['path']), candidate['entry'], candidate['line'])
        notices = [line for line in contained(Path(snapshot['checkout']), candidate['path']).read_text().splitlines()[:20]
                   if 'copyright' in line.lower() or 'SPDX-License-Identifier' in line]
        if notices:
            code = '\n'.join(notices) + '\n' + code
        fence = '`' * max(3, max((len(m.group()) + 1 for m in re.finditer(r'`+', code)), default=3))
        language = 'python' if candidate['path'].endswith('.py') else 'cpp'
        # Only explicitly admitted entry definitions appear in the page.
        body = (f"# {checked_text(metadata['title'])}\n\n"
                f"Source: `{checked_text(candidate['path'])}`; entry `{checked_text(candidate['entry'])}`; "
                f"commit `{snapshot['commit']}`.\n\n"
                f"## Implementation\n\n{fence}{language}\n{code.rstrip()}\n{fence}\n\n"
                f"## Dependencies\n\n"
                + '\n'.join(f"- `{checked_text(d['path'])}` (`{d['sha256']}`)." for d in candidate['dependencies'])
                + f"\n\nUnresolved: {checked_text(', '.join(candidate['unresolved_dependencies'])) or 'none'}.\n\n"
                + f"## Review\n\n{decision['rationale']}\n\n"
                f"Confidence basis: {decision['confidence_basis']}\n\n"
                f"## Reproducibility evidence\n\n"
                + '\n'.join(f"- {key}: `{checked_text(value.get('path', ''))}`, {checked_text(value.get('locator', ''))} ({value.get('basis')})."
                            for key, value in decision.get('reproducibility_evidence', {}).items()) + '\n')
        if decision['action'] == 'extend_page':
            old_text = outputs.get(page_path) or contained(root, page_path).read_text()
            if implementation.get('managed') is False or f'import_record: {identity}' not in old_text:
                raise ValueError('Cannot replace an existing kernel page; supply a separate reviewed page edit')
        outputs[page_path] = markdown(metadata, body)
    # Append only importer-owned provenance, preserving all existing page prose.
    for implementation in registered['implementations'].values():
        page_path = implementation['page_path']
        if page_path not in outputs:
            continue
        text = re.sub(r'\n*<!-- repo-import-members:start -->.*?<!-- repo-import-members:end -->\n*',
                      '\n', outputs[page_path], flags=re.DOTALL).rstrip()
        section = member_section(implementation)
        outputs[page_path] = text + ('\n\n' + section if section else '') + '\n'
        implementation['page_receipt'] = {'path': page_path,
            'sha256': hashlib.sha256(outputs[page_path].encode()).hexdigest()}
        representative = implementation['versions'][implementation['current_version']]
        if 'page_evidence' in representative:
            representative['page_evidence']['sha256'] = hashlib.sha256(outputs[page_path].encode()).hexdigest()
    manifest_path = f"data/repo-import/decisions/{snapshot['repo_id']}/{request['run_id']}.yaml"
    formal = compact_manifest(run, request, document, selected)
    outputs[manifest_path] = yaml.safe_dump(formal, sort_keys=False)
    outputs['data/repo-import/registry.yaml'] = yaml.safe_dump(registered, sort_keys=False)
    files = []
    for relative, text in sorted(outputs.items()):
        staged = contained(run / 'staging', relative)
        atomic_write(staged, text)
        files.append({'path': relative, 'before': file_hash(contained(root, relative)), 'after': file_hash(staged)})
    plan = {'version': 1, 'run_id': request['run_id'], 'files': files,
            'request_digest': digest(request), 'decisions_digest': digest(document)}
    write(run / 'publication-plan.yaml', plan)
    return plan


def registry_errors(root):
    errors = []
    data = registry(root)
    if data.get('version') != 1 or not isinstance(data.get('implementations'), dict):
        return ['Invalid repository import registry']
    page_ids = set()
    for identity, item in data['implementations'].items():
        try:
            if item['page_id'] in page_ids:
                raise ValueError('Multiple implementations claim one canonical page')
            page_ids.add(item['page_id'])
            path = contained(root, item['page_path'])
            text = path.read_text()
            match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
            fm = yaml.safe_load(match.group(1))
            if fm['id'] != item['page_id'] or fm.get('import_record') != identity:
                raise ValueError('Registry/page identity mismatch')
            versions = item['versions']
            if len(versions) != 1:
                raise ValueError('Exactly one full representative version is required')
            if item['current_version'] not in versions:
                raise ValueError('Unknown current version')
            if item.get('managed', True) and fm.get('reproducibility') != versions[item['current_version']]['metadata']['reproducibility']:
                raise ValueError('Page reproducibility differs from its selected version')
            for version_id, version in versions.items():
                relationships = {**item.get('history', {}), **versions}
                visited, cursor = set(), version_id
                while cursor:
                    if cursor in visited or cursor not in relationships:
                        raise ValueError('Version relation is cyclic or dangling')
                    visited.add(cursor)
                    cursor = (relationships[cursor].get('relation') or {}).get('target')
                if 'page_evidence' in version:
                    if version['page_evidence']['path'] != item['page_path'] or version['page_evidence']['sha256'] != file_hash(path):
                        raise ValueError('Existing-page representative receipt changed')
                    if fm.get('reproducibility') != version['metadata']['reproducibility']:
                        raise ValueError('Existing-page representative reproducibility mismatch')
                    continue
                candidate = version['candidate']
                actual = digest({k: candidate[k] for k in ('path', 'entry', 'kind', 'fingerprints', 'unresolved_dependencies')})
                if actual != candidate['content_digest']:
                    raise ValueError('Stored candidate fingerprint mismatch')
                if version['decision']['content_digest'] != candidate['content_digest']:
                    raise ValueError('Stored decision binding mismatch')
        except (KeyError, ValueError, OSError, AttributeError, TypeError) as error:
            errors.append(f'{identity}: {error}')
    return errors


def validate_plan(root, run, plan):
    import tempfile
    temporary = Path(tempfile.mkdtemp(prefix='kernelwiki-import-')) / 'wiki'
    excluded = {'.git', '.cache', '__pycache__', '.worklog', '.venv'}
    config = read(run / 'request.yaml').get('config', {})
    storage_roots = {run.parent.resolve()}
    for key in ('state_dir', 'checkout_dir'):
        if config.get(key):
            path = Path(config[key]).expanduser()
            storage_roots.add((path if path.is_absolute() else root / path).resolve())
    def ignore(directory, names):
        return [name for name in names if name in excluded or (Path(directory) / name).resolve() in storage_roots]
    shutil.copytree(root, temporary, ignore=ignore)
    for record in plan['files']:
        source = contained(run / 'staging', record['path'])
        if file_hash(source) != record['after']:
            raise ValueError('Staged content changed; rebuild required')
        destination = contained(temporary, record['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    results = []
    for script in ('generate-indices.py', 'validate.py', 'repo_size_check.py'):
        result = subprocess.run([sys.executable, str(temporary / 'scripts' / script)], cwd=temporary,
                                capture_output=True, text=True)
        results.append({'script': script, 'exit_code': result.returncode, 'output': result.stdout + result.stderr})
        write(run / 'validation.yaml', results)
        if result.returncode:
            shutil.rmtree(temporary.parent)
            raise ValueError(f'{script} failed; see {run / "validation.yaml"}')
    # Index outputs participate in the same publication transaction.
    for path in sorted((temporary / 'queries').glob('*.md')):
        relative = path.relative_to(temporary).as_posix()
        if file_hash(path) == file_hash(contained(root, relative)):
            continue
        destination = contained(run / 'staging', relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        plan['files'].append({'path': relative, 'before': file_hash(contained(root, relative)), 'after': file_hash(path)})
    plan['validated_digest'] = digest(plan['files'])
    write(run / 'publication-plan.yaml', plan)
    shutil.rmtree(temporary.parent)


def recover_transaction(root, run):
    journal = read(run / 'transaction.yaml')
    if journal.get('status') != 'applying':
        return
    for record in reversed(journal['files']):
        destination = contained(root, record['path'])
        current = file_hash(destination)
        if current not in {record['before'], record['after']}:
            raise ValueError('Recovery would overwrite a concurrent edit: ' + record['path'])
        if record['before'] is None:
            destination.unlink(missing_ok=True)
        else:
            backup = contained(run / 'backup', record['path'])
            atomic_write(destination, backup.read_text())
    write(run / 'transaction.yaml', {**journal, 'status': 'rolled_back'})


def apply_plan(root, run, plan):
    lock = root / '.cache/repo-import/publish.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_transaction(root, run)
        if plan.get('validated_digest') != digest(plan['files']):
            raise ValueError('Plan has not passed validation')
        for record in plan['files']:
            if file_hash(contained(root, record['path'])) != record['before']:
                raise ValueError('Concurrent target edit: ' + record['path'])
            if file_hash(contained(run / 'staging', record['path'])) != record['after']:
                raise ValueError('Staged content changed')
            if record['before'] is not None:
                backup = contained(run / 'backup', record['path'])
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contained(root, record['path']), backup)
        write(run / 'transaction.yaml', {'status': 'applying', 'files': plan['files']})
        try:
            for record in plan['files']:
                destination = contained(root, record['path'])
                if file_hash(destination) != record['before']:
                    raise ValueError('Concurrent edit during publication')
                atomic_write(destination, contained(run / 'staging', record['path']).read_text())
        except Exception:
            recover_transaction(root, run)
            raise
        write(run / 'transaction.yaml', {'status': 'committed', 'files': plan['files']})
