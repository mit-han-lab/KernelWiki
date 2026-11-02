"""Portable, read-only review records; never publish or enable an import preset."""
import configparser
import copy
import gzip
import hashlib
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlparse

from _yaml_compat import yaml
from .store import contained, digest, file_hash, read, atomic_write

ARCHIVE_FORMAT = 'kernelwiki-repo-review-archive-v1'
PACK_FORMAT = 'kernelwiki-repo-review-pack-v1'
LOCAL_PATH = re.compile(r'/(?:Users|home|private|tmp|var/folders)/[^\s\x22\x27<>),\]}]+')
CACHE_PATH = re.compile(r'(?<![\w/])\.cache/repo-import/[^\s\x22\x27<>),\]}]+')
CODE_SUFFIXES = {'.cu', '.cuh', '.h', '.hpp', '.cpp', '.cc', '.c', '.py', '.ptx', '.cubin'}


def compact(value):
    """Share equal containers using standard YAML anchors, with no data loss."""
    memo = {}

    def visit(item):
        if isinstance(item, dict):
            pairs = [(key, visit(child)) for key, child in item.items()]
            key = ('dict', tuple((name, signature) for name, (signature, _) in pairs))
            result = {name: child for name, (_, child) in pairs}
        elif isinstance(item, list):
            pairs = [visit(child) for child in item]
            key = ('list', tuple(signature for signature, _ in pairs))
            result = [child for _, child in pairs]
        else:
            return (type(item).__name__, item), item
        if key not in memo:
            memo[key] = result
        return key, memo[key]

    return visit(value)[1]


def write_record(path, value):
    dumper = getattr(yaml, 'CSafeDumper', yaml.SafeDumper)
    text = yaml.dump(compact(value), Dumper=dumper, sort_keys=False,
                     allow_unicode=True, width=100)
    if LOCAL_PATH.search(text):
        raise ValueError('Machine-local path remains in archived record')
    atomic_write(path, text)


def public_repo(url):
    url = url.removesuffix('.git')
    if not url.startswith('https://github.com/') or urlparse(url).query or urlparse(url).fragment:
        raise ValueError('Archive requires an HTTPS GitHub source')
    if len(urlparse(url).path.strip('/').split('/')) != 2:
        raise ValueError('Expected a repository URL')
    return url


class Relocator:
    def __init__(self, root, batch, records, source_catalogs=()):
        self.root, self.batch = Path(root).resolve(), Path(batch).resolve()
        self.origins, self.checkouts, self.commit_origins = {}, {}, {}
        self.catalog, self.notes, self.source_addresses = {}, {}, {}
        for record in records:
            snapshot = read(Path(record['run']) / 'snapshot.yaml')
            repo = public_repo(snapshot['repo'])
            if not re.fullmatch(r'[0-9a-f]{40}', snapshot['commit']):
                raise ValueError('Archive requires a fixed full commit')
            self.origins[snapshot['repo_id']] = repo
            self.checkouts[str(Path(snapshot['checkout']).resolve())] = (repo, snapshot['commit'])
            self.commit_origins[snapshot['commit']] = repo
        # This optional provenance manifest is a recorded public object-store index.
        # Read Git configuration only; never fetch or inspect an unlisted source.
        def source_origins(value, inherited=None):
            if isinstance(value, dict):
                repo = value.get('repo', inherited)
                if all(value.get(key) for key in ('package_url', 'package_sha256', 'member_path', 'local_path', 'sha256')):
                    url, checksum = value['package_url'], value['package_sha256']
                    if not url.startswith('https://files.pythonhosted.org/') or not re.fullmatch(r'[0-9a-f]{64}', checksum):
                        raise ValueError('Unsupported or unpinned package evidence')
                    uri = 'package+' + url + '#sha256=' + checksum + '&path=' + quote(value['member_path'], safe='/')
                    self.source_addresses[str((self.root / value['local_path']).resolve())] = (uri, value['sha256'])
                raw_url = re.fullmatch(r'https://raw.githubusercontent.com/([^/]+/[^/]+)/([0-9a-f]{40})/(.+)', value.get('url', ''))
                if raw_url:
                    repo = 'https://github.com/' + raw_url[1]
                    self.commit_origins[raw_url[2]] = repo
                    local = value.get('local_path')
                    if local and value.get('sha256'):
                        self.source_addresses[str((self.root / local).resolve())] = (
                            self.git_uri(repo, raw_url[2], raw_url[3]), value['sha256'])
                if repo and repo.startswith('https://github.com/'):
                    repo = public_repo(repo)
                    for key in ('commit', 'base_commit'):
                        commit = value.get(key, '')
                        if isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit):
                            previous = self.commit_origins.get(commit)
                            if previous and previous != repo:
                                raise ValueError('Conflicting historical source origins')
                            self.commit_origins[commit] = repo
                for child in value.values():
                    source_origins(child, repo)
            elif isinstance(value, list):
                for child in value:
                    source_origins(child, inherited)

        for index in source_catalogs:
            document = read(Path(index))
            source_origins(document)
            for directory in (document.get('public_object_directories', []) if isinstance(document, dict) else []):
                directory = Path(directory)
                match = re.fullmatch(r'([0-9a-f]{40})\.git', directory.parent.name if directory.name == 'objects' else directory.name)
                gitdir = directory.parent if directory.name == 'objects' else directory
                if not match or not (gitdir / 'config').is_file():
                    continue
                config = configparser.ConfigParser()
                config.read(gitdir / 'config')
                urls = []
                for section in config.sections():
                    if section.startswith('remote '):
                        urls.append(config.get(section, 'url', fallback=''))
                        urls.extend(re.findall(r'https://github\.com/[^\s\x22]+', section))
                for url in urls:
                    if url.startswith('https://github.com/') and public_repo(url) in set(self.origins.values()):
                        self.commit_origins[match[1]] = public_repo(url)

    @staticmethod
    def git_uri(repo, commit, path):
        return 'git+' + repo + '@' + commit + '/' + quote(path, safe='/')

    def locate(self, value, required=False):
        path = self.root / value if value.startswith('.cache/repo-import/') else Path(value)
        path = path.resolve()
        normalized = str(path)
        if normalized in self.source_addresses:
            uri, checksum = self.source_addresses[normalized]
            if file_hash(path) != checksum:
                raise ValueError('Stale public source catalog receipt')
            return uri
        for prefix, (repo, commit) in sorted(self.checkouts.items(), key=lambda p: -len(p[0])):
            if normalized.startswith(prefix + '/'):
                return self.git_uri(repo, commit, normalized[len(prefix) + 1:])
        match = re.search(r'/checkouts/(repo-[0-9a-f]+)/([0-9a-f]{40})/(.+)$', normalized)
        if match and match[1] in self.origins:
            return self.git_uri(self.origins[match[1]], match[2], match[3])
        match = re.search(r'/([0-9a-f]{40})/(.+)$', normalized)
        if match and match[1] in self.commit_origins:
            return self.git_uri(self.commit_origins[match[1]], match[1], match[2])
        if path.is_relative_to(self.root):
            relative = path.relative_to(self.root)
            if relative.parts and relative.parts[0] in {'wiki', 'sources', 'artifacts', 'data', 'references', 'scripts'}:
                return 'wiki://' + relative.as_posix()
        if required:
            raise ValueError('Cannot establish portable public source origin: ' + path.name)
        if path.is_relative_to(self.batch):
            return 'history://' + path.relative_to(self.batch).as_posix()
        return 'local-reference://' + hashlib.sha256(value.encode()).hexdigest()[:24]

    def text(self, value):
        def replace(match):
            raw = match[0]
            return self.notes.get(raw, self.locate(raw))
        return CACHE_PATH.sub(replace, LOCAL_PATH.sub(replace, value))

    def receipt(self, receipt):
        """Bind an absolute current receipt before relocating its address."""
        raw_path = receipt['path']
        path = self.root / raw_path if raw_path.startswith('.cache/repo-import/') else Path(raw_path)
        if not path.is_absolute():
            return
        path = path.resolve()
        checksum = receipt['sha256']
        if file_hash(path) != checksum:
            raise ValueError('Stale source receipt: ' + path.name)
        if path.is_relative_to(self.root) and path.relative_to(self.root).parts[0] in {'wiki', 'sources', 'artifacts'}:
            self.notes[str(path)] = self.locate(str(path), required=True)
        elif path.suffix.lower() in CODE_SUFFIXES:
            uri = self.locate(str(path), required=True)
            self.notes[str(path)] = uri
            self.catalog.setdefault(uri, {'kind': 'public-package-source' if uri.startswith('package+') else 'public-git-source', 'sha256': checksum})
            if self.catalog[uri]['sha256'] != checksum:
                raise ValueError('Conflicting source hashes')
        else:
            uri = 'archive://evidence/' + checksum + '.yaml'
            self.notes[str(path)] = uri
            if uri not in self.catalog:
                # Preserve documentary evidence. Full upstream implementation files
                # are referenced above by public Git origin instead of being copied.
                self.catalog[uri] = {'kind': 'review-evidence', 'original_sha256': checksum,
                                     'name': path.name, '_text': path.read_text()}
        self.notes[raw_path] = self.notes[str(path)]

    def register(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.receipt(value)
            for child in value.values():
                self.register(child)
        elif isinstance(value, list):
            for child in value:
                self.register(child)

    def transform(self, value):
        if isinstance(value, dict):
            return {key: self.transform(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self.transform(child) for child in value]
        return self.text(value) if isinstance(value, str) else value

    def evidence(self, destination):
        for uri, record in self.catalog.items():
            if record['kind'] != 'review-evidence':
                continue
            payload = {'version': 1, 'original_sha256': record['original_sha256'],
                       'name': record['name'], 'content': self.text(record.pop('_text')),
                       'relocation': 'Only local path tokens were relocated; original and archive byte hashes differ.'}
            relative = uri.removeprefix('archive://')
            write_record(destination / relative, payload)
            record.update(archive_path=relative, archive_sha256=file_hash(destination / relative))


def group_identity(repo, candidate_id, decision):
    return (decision.get('target_implementation_id') or decision.get('target_page_id') or
            'group-' + digest({'repo': repo, 'candidate': candidate_id})[:24])


def state_check(inventory, selection, document, valid, summary, repo):
    """Check coverage and verdicts independently, without calling build/apply."""
    raw = {candidate['candidate_id']: candidate for candidate in inventory['candidates']}
    rows = {row['candidate_id']: row for row in selection['entries']}
    verdicts = {row['candidate_id']: row for row in document['decisions']}
    if len(raw) != len(inventory['candidates']) or len(rows) != len(selection['entries']) or set(raw) != set(rows):
        raise ValueError('Archive has incomplete or duplicate selection coverage')
    representatives = {key for key, row in rows.items() if row['role'] == 'representative'}
    if len(verdicts) != len(document['decisions']) or set(verdicts) != representatives:
        raise ValueError('Every representative must have exactly one review verdict')
    for key, row in rows.items():
        if row['content_digest'] != raw[key]['content_digest']:
            raise ValueError('Stale candidate binding')
        if row['role'] not in {'representative', 'member', 'helper', 'pending'}:
            raise ValueError('Unknown selection status')
        if row['role'] == 'member' and (row.get('representative') not in representatives or not row.get('difference_summary')):
            raise ValueError('Member lacks a direct representative or comparison')
        if row['role'] == 'pending' and not row.get('rationale', '').strip():
            raise ValueError('Unfinished selection cannot be archived as a resolved hold')
    for key, verdict in verdicts.items():
        if verdict['content_digest'] != raw[key]['content_digest']:
            raise ValueError('Stale review binding')
        if verdict['decision'] not in {'include', 'exclude', 'defer'}:
            raise ValueError('Unknown review status')
        if not verdict.get('rationale', '').strip() or verdict['rationale'] == 'Pending external review':
            raise ValueError('Unfinished review cannot be relabelled as a genuine defer')
    includes = {key for key, row in verdicts.items() if row['decision'] == 'include'}
    represented, members = set(), set()
    group_ids = set()
    for group in valid['implementations']:
        if group['implementation_id'] in group_ids:
            raise ValueError('Duplicate effective group')
        group_ids.add(group['implementation_id'])
        group_reps = set()
        for candidate in group['representatives']:
            key = candidate['candidate_id']
            if key not in includes or key in represented:
                raise ValueError('Effective list admits a held, excluded or duplicate representative')
            if group_identity(repo, key, verdicts[key]) != group['implementation_id']:
                raise ValueError('Wrong canonical group')
            if candidate['content_digest'] != raw[key]['content_digest'] or candidate['metadata'] != verdicts[key]['metadata']:
                raise ValueError('Effective list changed source binding or confidence')
            represented.add(key)
            group_reps.add(key)
        for candidate in group['members']:
            key = candidate['candidate_id']
            if key in members or rows[key]['role'] != 'member' or rows[key]['representative'] not in group_reps:
                raise ValueError('Effective member belongs to an unresolved or different group')
            if candidate['content_digest'] != raw[key]['content_digest']:
                raise ValueError('Stale effective member')
            members.add(key)
    expected_members = {key for key, row in rows.items() if row['role'] == 'member' and row['representative'] in includes}
    if represented != includes or members != expected_members:
        raise ValueError('Effective list drops an admitted source')
    roles = Counter(row['role'] for row in rows.values())
    decisions = Counter(row['decision'] for row in verdicts.values())
    expected = {'definitions': len(raw), 'representatives': roles['representative'],
                'members': roles['member'], 'helpers': roles['helper'],
                'selection_pending': roles['pending'], 'reviewed_include': decisions['include'],
                'reviewed_exclude': decisions['exclude'], 'review_defer': decisions['defer'],
                'effective_kernels': len(group_ids), 'included_definitions': len(includes) + len(members),
                'pending_definitions': sum(row['role'] == 'pending' or
                    (row['role'] == 'representative' and verdicts[key]['decision'] == 'defer') or
                    (row['role'] == 'member' and verdicts[row['representative']]['decision'] == 'defer')
                    for key, row in rows.items()),
                'selection_defer': roles['pending'],
                'review_unfinished': 0, 'selection_unfinished': 0, 'blocked_include': 0}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError('Archive status/count mismatch: ' + key)
    return expected


def export_reviews(root, batch, destination, source_catalogs=()):
    root, batch, destination = Path(root).resolve(), Path(batch).resolve(), Path(destination).resolve()
    allowed = root / 'examples/repo-import/reviews'
    if not destination.is_relative_to(allowed) or destination == allowed:
        raise ValueError('Review archives must be an isolated batch under examples/repo-import/reviews')
    if destination.exists():
        raise ValueError('Refuse to overwrite an existing review archive')
    acceptance = read(batch / 'final-review-acceptance.yaml')
    records = acceptance['repositories']
    relocator = Relocator(root, batch, records, source_catalogs)
    index = {'format': ARCHIVE_FORMAT, 'archive_only': True, 'admission_enabled': False,
             'scan_observed_at': read(batch / 'manifest.yaml').get('created_at'),
             'review_completed_at': acceptance['completed_at'],
             'source_acceptance_sha256': file_hash(batch / 'final-review-acceptance.yaml'),
             'repositories': [], 'files': {}}
    pending = []
    try:
        for record in records:
            name, run = record['name'], Path(record['run'])
            slug = re.sub(r'[^a-z0-9-]+', '-', name.lower())
            source = {key: read(run / (key + '.yaml')) for key in
                      ('snapshot', 'inventory', 'selection', 'decisions', 'matches', 'discovered-overlaps')}
            summary = copy.deepcopy(record['summary'])
            valid_path = Path(record['valid_kernels']['path'])
            if file_hash(valid_path) != record['valid_kernels']['sha256']:
                raise ValueError('Validated list changed: ' + name)
            valid = read(valid_path)
            if digest(source['decisions']) != summary['decisions_digest']:
                raise ValueError('Decisions changed after acceptance: ' + name)
            if source['selection']['inventory_digest'] != digest(source['inventory']):
                raise ValueError('Selection scan binding changed: ' + name)
            state_check(source['inventory'], source['selection'], source['decisions'], valid, summary, record['repo'])
            for key in ('selection', 'decisions', 'discovered-overlaps'):
                relocator.register(source[key])
            fingerprints = {key: {'file_sha256': file_hash(run / (key + '.yaml')), 'record_digest': digest(value)}
                            for key, value in source.items() if (run / (key + '.yaml')).exists()}
            snapshot = {key: value for key, value in source['snapshot'].items()
                        if key not in {'checkout', 'ignored_worktree_changes'}}
            snapshot['scan_config'] = summary['scan_config']
            source['snapshot'] = snapshot
            summary.pop('run_reference', None)
            source['summary'] = summary
            valid['summary'] = summary
            source['valid-kernels'] = valid
            source['decisions']['archive_only'] = True
            for key, value in source.items():
                write_record(destination / slug / (key + '.yaml'), relocator.transform(value))
            guide = run / 'review-guide.md'
            if guide.exists():
                guide_name = 'prompts/' + file_hash(guide) + '.md'
                text = relocator.text(guide.read_text())
                atomic_write(destination / guide_name, text)
            else:
                raise ValueError('Saved review prompt is missing: ' + name)
            # Only the Wiki context and actual matching inputs are kept, not another
            # copy of the full candidate inventory embedded in review-request.yaml.
            request = read(run / 'review-request.yaml')
            if digest(request) != summary['request_digest'] or request['matches'] != source['matches']:
                raise ValueError('Prepared review inputs changed after acceptance: ' + name)
            context = {'context_digest': request['context_digest'], 'policy_receipts': request.get('policy_receipts', {}),
                       'existing_pages': request['existing_pages'], 'registry_digest': digest(request['registry'])}
            context_name = 'contexts/' + digest(context) + '.yaml'
            write_record(destination / context_name, relocator.transform(context))
            repo_info = {'name': name, 'directory': slug, 'repo': record['repo'], 'commit': record['commit'],
                         'prompt': guide_name, 'context': context_name, 'source_records': fingerprints,
                         'summary': summary}
            index['repositories'].append(repo_info)
            raw = {row['candidate_id']: row for row in source['inventory']['candidates']}
            for stage, rows in [('review', source['decisions']['decisions']), ('selection', source['selection']['entries'])]:
                for row in rows:
                    status = row.get('decision') if stage == 'review' else row.get('role')
                    if status != ('defer' if stage == 'review' else 'pending'):
                        continue
                    candidate = raw[row['candidate_id']]
                    pending.append({'repo': name, 'stage': stage, 'status': status,
                                    'candidate_id': row['candidate_id'], 'content_digest': row['content_digest'],
                                    'path': candidate['path'], 'entry': candidate['entry'],
                                    'rationale': row['rationale'], 'evidence': row.get('evidence', [])})
        relocator.evidence(destination)
        write_record(destination / 'evidence-index.yaml', {'version': 1, 'records': relocator.catalog})
        write_record(destination / 'pending.yaml', {'version': 1, 'archive_only': True,
                                                   'entries': relocator.transform(pending)})
        index['totals'] = dict(sum((Counter({key: repo['summary'][key] for key in
                           ('effective_kernels', 'reviewed_include', 'reviewed_exclude', 'review_defer', 'selection_pending')})
                           for repo in index['repositories']), Counter()))
        index['history_scope'] = 'Current decisions, selection, matching inputs, prompts and directly cited documentary evidence. Iterative raw agent transcripts are retained outside this archive; history:// references are documentary navigation, never admission evidence.'
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                index['files'][path.relative_to(destination).as_posix()] = file_hash(path)
        write_record(destination / 'index.yaml', index)
        return verify_archive(destination)
    except Exception:
        # A partial folder is deliberately retained for inspection, with no index
        # claiming success. No previous run or existing archive is modified.
        raise


def unpack_records(directory, destination):
    """Read our file-only tar format without trusting member paths or links."""
    manifest = read(directory / 'archive.yaml')
    package = directory / 'records.tar.gz'
    if (manifest.get('format') != PACK_FORMAT or manifest.get('archive_only') is not True
            or manifest.get('admission_enabled') is not False
            or file_hash(package) != manifest.get('sha256')):
        raise ValueError('Changed or invalid compressed review archive')
    seen, total = set(), 0
    with tarfile.open(package, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise ValueError('Review archive contains a link or duplicate/non-file member')
            path = contained(destination, member.name)
            seen.add(member.name)
            total += member.size
            if len(seen) > 10000 or total > 256 * 1024 * 1024:
                raise ValueError('Review archive exceeds extraction bounds')
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('wb') as target:
                # Bound each read; extraction never invokes a shell or writes links.
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    if len(seen) != manifest.get('files') or total != manifest.get('uncompressed_bytes'):
        raise ValueError('Compressed archive inventory mismatch')
    return manifest


def pack_archive(source, destination):
    """Create a compact copy; retain the original review directory untouched."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Compressed output must be a new directory outside its input')
    result = verify_archive(source)
    if (source / 'archive.yaml').exists():
        raise ValueError('Input is already compressed')
    destination.mkdir(parents=True)
    paths = sorted(path for path in source.rglob('*') if path.is_file() and path.name != 'README.md')
    package = destination / 'records.tar.gz'
    with package.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw,
            mtime=0, compresslevel=9) as compressed, tarfile.open(fileobj=compressed,
            mode='w|', format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            member = archive.gettarinfo(str(path), arcname=path.relative_to(source).as_posix())
            member.uid = member.gid = member.mtime = 0
            member.uname = member.gname = ''
            member.mode = 0o644
            with path.open('rb') as stream:
                archive.addfile(member, stream)
    write_record(destination / 'archive.yaml', dict(format=PACK_FORMAT, archive_only=True,
        admission_enabled=False, sha256=file_hash(package), files=len(paths),
        uncompressed_bytes=sum(path.stat().st_size for path in paths), result=result))
    if verify_archive(destination) != result:
        raise ValueError('Compressed results differ from original')
    return result


def verify_archive(directory):
    directory = Path(directory).resolve()
    if (directory / 'archive.yaml').exists():
        if {path.name for path in directory.iterdir()} - {'archive.yaml', 'records.tar.gz', 'README.md'}:
            raise ValueError('Unexpected files next to compressed review archive')
        with tempfile.TemporaryDirectory(prefix='kernelwiki-review-verify-') as temporary:
            unpacked = Path(temporary)
            manifest = unpack_records(directory, unpacked)
            result = verify_archive(unpacked)
            if result != manifest['result']:
                raise ValueError('Compressed result summary mismatch')
            return result
    index = read(directory / 'index.yaml')
    if index.get('format') != ARCHIVE_FORMAT or index.get('archive_only') is not True or index.get('admission_enabled') is not False:
        raise ValueError('Not a read-only review archive')
    catalog = read(directory / 'evidence-index.yaml')['records']

    def check_receipts(value):
        if isinstance(value, dict):
            uri = value.get('path', '')
            if isinstance(uri, str) and uri.startswith(('archive://', 'git+', 'package+')) and 'sha256' in value:
                record = catalog.get(uri)
                if not record or value['sha256'] != record.get('original_sha256', record.get('sha256')):
                    raise ValueError('Missing or mismatched portable evidence receipt')
            for child in value.values():
                check_receipts(child)
        elif isinstance(value, list):
            for child in value:
                check_receipts(child)

    expected_files = {path.relative_to(directory).as_posix() for path in directory.rglob('*')
                      if path.is_file() and path.name not in {'index.yaml', 'README.md'}}
    if set(index['files']) != expected_files:
        raise ValueError('Archive file manifest has missing or unexpected records')
    for name, checksum in index['files'].items():
        path = contained(directory, name)
        if path.is_symlink() or file_hash(path) != checksum:
            raise ValueError('Changed or missing archive file: ' + name)
        if LOCAL_PATH.search(path.read_text()):
            raise ValueError('Archive still depends on a local machine path')
    pending = read(directory / 'pending.yaml')['entries']
    expected_pending = []
    totals = Counter()
    for info in index['repositories']:
        base = contained(directory, info['directory'])
        inventory, selection, decisions, valid, summary = [read(base / (key + '.yaml')) for key in
                                                         ('inventory', 'selection', 'decisions', 'valid-kernels', 'summary')]
        if decisions.get('archive_only') is not True:
            raise ValueError('Archived decisions must not be an active preset')
        state_check(inventory, selection, decisions, valid, summary, info['repo'])
        snapshot = read(base / 'snapshot.yaml')
        if summary != info['summary'] or valid['summary'] != summary:
            raise ValueError('Index summary differs from archived result')
        if snapshot['repo'] != info['repo'] or snapshot['commit'] != info['commit']:
            raise ValueError('Snapshot differs from indexed source')
        if selection['inventory_digest'] != digest(inventory):
            raise ValueError('Archived selection scan binding changed')
        for value in (selection, decisions, read(base / 'discovered-overlaps.yaml')):
            check_receipts(value)
        totals.update({key: summary[key] for key in index['totals']})
        for stage, rows, key, held in [('review', decisions['decisions'], 'decision', 'defer'),
                                       ('selection', selection['entries'], 'role', 'pending')]:
            for row in rows:
                if row[key] == held:
                    candidate = next(item for item in inventory['candidates'] if item['candidate_id'] == row['candidate_id'])
                    expected_pending.append(dict(repo=info['name'], stage=stage, status=held,
                        candidate_id=row['candidate_id'], content_digest=row['content_digest'],
                        path=candidate['path'], entry=candidate['entry'], rationale=row['rationale'],
                        evidence=row.get('evidence', [])))
    if sorted(map(digest, expected_pending)) != sorted(map(digest, pending)) or dict(totals) != index['totals']:
        raise ValueError('Held candidate statuses or counts were changed or lost')
    for uri, record in catalog.items():
        if record['kind'] == 'review-evidence':
            path = contained(directory, record['archive_path'])
            if uri != 'archive://' + record['archive_path'] or record['archive_path'] not in index['files']:
                raise ValueError('Evidence is not in the archive manifest')
            if file_hash(path) != record['archive_sha256'] or read(path)['original_sha256'] != record['original_sha256']:
                raise ValueError('Documentary evidence binding mismatch')
        elif record['kind'] == 'public-package-source':
            if not re.fullmatch(r'package\+https://files\.pythonhosted\.org/[^#]+#sha256=[0-9a-f]{64}&path=.+', uri):
                raise ValueError('Unpinned public package evidence')
        elif record['kind'] != 'public-git-source' or not re.fullmatch(r'git\+https://github\.com/[^/]+/[^@]+@[0-9a-f]{40}/.+', uri):
            raise ValueError('Unpinned or unsupported source evidence')
    return {'status': 'passed', 'repositories': len(index['repositories']), **dict(totals),
            'pending_records': len(pending), 'files': len(index['files']),
            'verification_scope': 'Offline archive integrity and status preservation; no new source/GPU validation or publication.'}
