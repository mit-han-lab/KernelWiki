"""Snapshot differences and conservative, content-bound selection reuse.

A difference is an observation, never an instruction to delete a published page.
Rename matches are navigation hints and never enter the automatic reuse map.
"""
import copy
import fcntl
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

from .checkout import canonical_repo
from .scan import EXTRACTOR_VERSION
from .selection import validate_selection
from .store import contained, digest, file_hash, read, write, valid_receipt


def _identity(candidate):
    return tuple(candidate[key] for key in ('path', 'entry', 'kind'))


def _contract(inventory, request):
    scan = copy.deepcopy(request['config']['scan'])
    scan.setdefault('include', ['*'])
    scan.setdefault('exclude', [])
    scan.setdefault('entry_kinds', ['cuda_global', 'cute_kernel', 'triton_jit'])
    for key in ('include', 'exclude', 'entry_kinds'):
        scan[key] = sorted(set(scan[key]))
    versions = sorted({c['fingerprints']['extraction_version'] for c in inventory['candidates']})
    return {'scan': scan, 'extractors': versions or [EXTRACTOR_VERSION],
            'coverage': request.get('extractor_contract', 'repo-scan-v1')}


def compare_snapshots(old_snapshot, old_inventory, old_request,
                      new_snapshot, new_inventory, new_request):
    """Compare observations without interpreting semantic equivalence."""
    if canonical_repo(old_snapshot['repo']) != canonical_repo(new_snapshot['repo']):
        raise ValueError('Update baseline belongs to a different repository')
    old_contract = _contract(old_inventory, old_request)
    new_contract = _contract(new_inventory, new_request)
    reasons = []
    if old_contract['scan'] != new_contract['scan']:
        reasons.append('scan_configuration_changed')
    if (old_contract['extractors'] != new_contract['extractors'] or
            old_contract['coverage'] != new_contract['coverage']):
        reasons.append('extractor_contract_changed')
    if old_inventory.get('failures') or new_inventory.get('failures'):
        reasons.append('scan_has_parse_failures')
    def skipped_pins(snapshot):
        return sorted((item['path'], item['commit']) for item in snapshot.get('skipped_submodules', []))
    if skipped_pins(old_snapshot) != skipped_pins(new_snapshot):
        reasons.append('skipped_dependency_pins_changed')
    buckets = []
    for inventory in (old_inventory, new_inventory):
        index = defaultdict(list)
        for candidate in inventory['candidates']:
            index[_identity(candidate)].append(candidate)
        buckets.append(index)
    old, new = buckets
    result = {'version': 1, 'repo': canonical_repo(new_snapshot['repo']),
              'baseline_commit': old_snapshot['commit'], 'observed_commit': new_snapshot['commit'],
              'baseline_inventory_digest': digest(old_inventory), 'inventory_digest': digest(new_inventory),
              'deletion_allowed': not reasons, 'incompatible_scan': reasons,
              'unchanged': [], 'changed': [], 'added': [], 'missing': [],
              'possible_renames': [], 'ambiguous_identities': [], 'id_map': {}}
    for key in sorted(set(old) | set(new)):
        previous, current = old.get(key, []), new.get(key, [])
        if len(previous) > 1 or len(current) > 1:
            # Identical complete source files preserve lexical definition locations.
            # This disambiguates class methods/overloads only when no source text
            # moved; body hashes alone cannot establish an overloaded identity.
            old_locations = {(c['line'], c['fingerprints']['source_sha256']): c for c in previous}
            new_locations = {(c['line'], c['fingerprints']['source_sha256']): c for c in current}
            exact_locations = (len(old_locations) == len(previous) == len(current) == len(new_locations)
                               and set(old_locations) == set(new_locations))
            if exact_locations:
                for location, before in sorted(old_locations.items()):
                    after = new_locations[location]
                    pair = {'previous': before['candidate_id'], 'current': after['candidate_id'],
                            'path': after['path'], 'entry': after['entry'], 'kind': after['kind']}
                    if before['content_digest'] == after['content_digest'] and before['dependencies'] == after['dependencies']:
                        result['unchanged'].append(pair)
                        result['id_map'][before['candidate_id']] = after['candidate_id']
                    else:
                        result['changed'].append(pair)
                continue
            result['ambiguous_identities'].append({
                'path': key[0], 'entry': key[1], 'kind': key[2],
                'previous': [c['candidate_id'] for c in previous],
                'current': [c['candidate_id'] for c in current]})
            continue
        if len(previous) == len(current) == 1:
            before, after = previous[0], current[0]
            pair = {'previous': before['candidate_id'], 'current': after['candidate_id'],
                    'path': after['path'], 'entry': after['entry'], 'kind': after['kind']}
            if before['content_digest'] == after['content_digest'] and before['dependencies'] == after['dependencies']:
                result['unchanged'].append(pair)
                result['id_map'][before['candidate_id']] = after['candidate_id']
            else:
                result['changed'].append(pair)
        else:
            result['missing'].extend(c['candidate_id'] for c in previous)
            result['added'].extend(c['candidate_id'] for c in current)
    if result['ambiguous_identities']:
        reasons.append('ambiguous_candidate_identity')
        result['deletion_allowed'] = False
    old_ids = {c['candidate_id']: c for c in old_inventory['candidates']}
    new_ids = {c['candidate_id']: c for c in new_inventory['candidates']}
    body_index = defaultdict(list)
    for identity in result['added']:
        candidate = new_ids[identity]
        body_index[(candidate['kind'], candidate['fingerprints']['entry_sha256'])].append(identity)
    for identity in result['missing']:
        candidate = old_ids[identity]
        matches = body_index.get((candidate['kind'], candidate['fingerprints']['entry_sha256']), [])
        if matches:
            result['possible_renames'].append({'previous': identity, 'candidates': matches,
                                               'basis': 'same_kind_and_entry_hash_only'})
    return result


def remap_selection_group(selection, representative_id, id_map):
    """Rebind only a complete original group; callers still check all receipts."""
    rows = [row for row in selection.get('entries', [])
            if row['candidate_id'] == representative_id or
            (row.get('role') == 'member' and row.get('representative') == representative_id)]
    if not rows or any(row['candidate_id'] not in id_map for row in rows):
        return None
    rebound = copy.deepcopy(rows)
    for row in rebound:
        row['candidate_id'] = id_map[row['candidate_id']]
        if row.get('role') == 'member':
            row['representative'] = id_map[row['representative']]
    return rebound


def remap_review_candidate(verdict, id_map):
    """Syntactic rebinding only: validate new context, groups and overlaps later."""
    if verdict.get('candidate_id') not in id_map:
        return None
    result = copy.deepcopy(verdict)
    result['candidate_id'] = id_map[result['candidate_id']]
    if result.get('target_candidate_id'):
        if result['target_candidate_id'] not in id_map:
            return None
        result['target_candidate_id'] = id_map[result['target_candidate_id']]
    # Match IDs bind old candidates/context and cannot authorize new comparisons.
    result.pop('overlap_resolutions', None)
    return result


def _receipts_valid(rows, checkout):
    try:
        return all(valid_receipt(contained(checkout, receipt['path']), receipt)
                   for row in rows for receipt in row.get('evidence', []))
    except (ValueError, KeyError, OSError):
        return False


def prepare_update(root, run, baseline, *, fresh_review=False):
    """Persist a comparison and rebind safe selection groups in the new run.

    The baseline and the wiki corpus are read-only. Publication state is never
    advanced by this function, even when all observed content is unchanged.
    """
    run, baseline = Path(run).resolve(), Path(baseline).resolve()
    if run == baseline:
        raise ValueError('Update needs a distinct immutable baseline run')
    if (run / 'transaction.yaml').exists() or (run / 'review-request.yaml').exists():
        raise ValueError('Prepare update before detailed review or publication')
    baseline_receipt = _verify_scan_receipt(baseline)
    current_receipt = _verify_scan_receipt(run)
    old_snapshot, snapshot = read(baseline / 'snapshot.yaml'), read(run / 'snapshot.yaml')
    old_inventory, inventory = read(baseline / 'inventory.yaml'), read(run / 'inventory.yaml')
    report = compare_snapshots(old_snapshot, old_inventory, read(baseline / 'request.yaml'),
                               snapshot, inventory, read(run / 'request.yaml'))
    if not baseline_receipt or not current_receipt:
        report['incompatible_scan'].append('legacy_scan_receipt_missing')
        report['deletion_allowed'] = False
    report.update(baseline_run=str(baseline), observed_run=str(run), fresh_review=fresh_review,
                  publication_advanced=False, reused_selection=[], invalidated_selection=[])
    selection = read(run / 'selection.yaml')
    previous = read(baseline / 'selection.yaml')
    if not selection:
        raise ValueError('Update requires a prepared selection template')
    # Avoid replacing externally edited work or silently rebasing a different scan.
    if selection.get('inventory_digest') != digest(inventory) or any(
            row.get('role') != 'pending' for row in selection.get('entries', [])):
        raise ValueError('Update requires an untouched current selection template')
    safe = not fresh_review and not report['incompatible_scan'] and bool(previous)
    if safe:
        try:
            validate_selection(previous, old_inventory, Path(old_snapshot['checkout']))
        except (ValueError, OSError, KeyError):
            safe = False
            report['reuse_blocked'] = 'baseline_selection_or_evidence_invalid'
    if safe:
        replacements = {}
        for row in previous['entries']:
            if row.get('role') not in {'representative', 'helper'}:
                continue
            rebound = remap_selection_group(previous, row['candidate_id'], report['id_map'])
            if rebound is not None and _receipts_valid(rebound, Path(snapshot['checkout'])):
                replacements.update({entry['candidate_id']: entry for entry in rebound})
            else:
                report['invalidated_selection'].append(row['candidate_id'])
        selection['entries'] = [replacements.get(row['candidate_id'], row) for row in selection['entries']]
        validate_selection(selection, inventory, Path(snapshot['checkout']))
        report['reused_selection'] = sorted(replacements)
        # Store the pristine template too; source runs and original receipts stay immutable.
        write(run / 'update-selection-template.yaml', read(run / 'selection.yaml'))
        write(run / 'selection.yaml', selection)
    from .discovery import retain_update_discoveries
    retain_update_discoveries(root, run, baseline, report, inventory)
    write(run / 'update.yaml', report)
    return report


def _verify_scan_receipt(run):
    """Legacy absence is explicit; existing receipts must bind all scan inputs."""
    run = Path(run)
    receipt = read(run / 'scan-receipt.yaml')
    if not receipt:
        return False
    if 'extractor_contract' not in receipt:
        return False  # Legacy scans cannot establish the current coverage contract.
    inventory = read(run / 'inventory.yaml')
    snapshot = read(run / 'snapshot.yaml')
    request = read(run / 'request.yaml')
    expected = {'inventory_digest': digest(inventory), 'snapshot_digest': digest(snapshot),
                'scan_config_digest': digest(request['config']['scan'])}
    if 'extractor_contract' in receipt:
        expected['extractor_contract'] = request.get('extractor_contract')
    if receipt.get('version') != 1 or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Scan receipt does not match inventory, snapshot or configuration')
    return True


def observed_state_path(root, run):
    """Return the per-repository index under configured persistent state storage."""
    run, root = Path(run), Path(root)
    snapshot = read(run / 'snapshot.yaml')
    config = read(run / 'request.yaml')['config']
    state_dir = Path(config['state_dir']).expanduser()
    state_dir = state_dir.resolve() if state_dir.is_absolute() else (root / state_dir).resolve()
    identity = 'repo-' + digest(canonical_repo(snapshot['repo']))[:16]
    return state_dir / 'observations' / (identity + '.yaml')


@contextmanager
def _locked_state(path, repo):
    """Serialize observation and publication writes for one repository."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = read(path, {'version': 1, 'repo': repo, 'scans': {}})
        if state.get('repo') != repo or state.get('version') != 1:
            raise ValueError('Observation index belongs to a different repository or schema')
        yield state
        write(path, state)


def record_observation(root, run):
    """Record a scan, never publishing it or replacing a good scan with a failure."""
    run = Path(run).resolve()
    verified = _verify_scan_receipt(run)
    snapshot, inventory = read(run / 'snapshot.yaml'), read(run / 'inventory.yaml')
    result = read(run / 'result.yaml')
    if not snapshot or 'candidates' not in inventory:
        raise ValueError('Observation requires a completed scan inventory and snapshot')
    complete = verified and not inventory.get('failures') and result.get('status') not in {'failed', 'preparing'}
    observation = {'run': str(run), 'commit': snapshot['commit'], 'inventory_digest': digest(inventory),
                   'receipt_verified': verified, 'complete': complete,
                   'parse_failures': len(inventory.get('failures', [])),
                   'status': result.get('status', 'scanned')}
    with _locked_state(observed_state_path(root, run), canonical_repo(snapshot['repo'])) as state:
        state.setdefault('scans', {})[str(run)] = observation
        state['last_observed_scan'] = str(run)
        if complete:
            state['last_complete_scan'] = str(run)
    return state


def record_publication(root, run):
    """Record a committed apply without changing newer scan observations."""
    run = Path(run).resolve()
    if read(run / 'transaction.yaml').get('status') != 'committed':
        raise ValueError('Publication state requires a committed apply transaction')
    if not _verify_scan_receipt(run):
        raise ValueError('Publication state requires a verified scan receipt')
    snapshot, result = read(run / 'snapshot.yaml'), read(run / 'result.yaml')
    inventory = read(run / 'inventory.yaml')
    with _locked_state(observed_state_path(root, run), canonical_repo(snapshot['repo'])) as state:
        state['last_applied_run'] = str(run)
        state.setdefault('published_runs', {})[str(run)] = {
            'commit': snapshot['commit'], 'included': result.get('included', 0),
            'pending': len(result.get('pending', []))}
        if not result.get('pending') and not inventory.get('failures'):
            state['last_published_run'] = str(run)
    return state
