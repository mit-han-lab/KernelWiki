"""Source-bound reviewer comparison requests, never equivalence decisions."""
import copy
from pathlib import Path

from .store import contained, file_hash, read, write, valid_receipt


def load_discoveries(run):
    path = Path(run) / 'discovered-overlaps.yaml'
    return read(path) if path.exists() else None


def definition_receipt(candidate):
    return {'path': candidate['path'], 'sha256': candidate['fingerprints']['source_sha256'],
            'locator': f"{candidate['entry']} at line {candidate['line']}"}


def page_receipt(page):
    return {'path': page['path'], 'sha256': page['sha256'], 'locator': page['page_id']}


def discovery_key(row):
    if 'target_candidate_id' in row:
        return ('batch', *sorted((row['candidate_id'], row['target_candidate_id'])))
    return ('page', row['candidate_id'], row.get('page_id'))


def definition_receipts(candidate, checkout):
    expected = definition_receipt(candidate)
    receipts = [expected]
    from .match import code_entries
    definitions = code_entries(str(contained(checkout, candidate['path'])), expected['sha256'])
    if sum(entry['name'] == candidate['entry'] for entry in definitions) == 1:
        receipts.append({**expected, 'locator': candidate['entry']})
    return receipts


def validate_discoveries(document, candidates, pages, root, checkout):
    """Require exact known source/target receipts and verify the actual files."""
    if document is None:
        return
    if not isinstance(document, dict) or document.get('version') != 1 or not isinstance(document.get('entries'), list):
        raise ValueError('Invalid discovered-overlaps format; require version: 1 and entries')
    inputs = {c['candidate_id']: c for c in candidates}
    targets = {p['page_id']: p for p in pages}
    seen = set()
    for row in document['entries']:
        if not isinstance(row, dict):
            raise ValueError('Invalid discovered overlap entry')
        batch = 'target_candidate_id' in row
        if batch and 'page_id' in row:
            raise ValueError('Discovered overlap requires exactly one target kind')
        key = (row.get('candidate_id'), row.get('target_candidate_id') if batch else row.get('page_id'))
        if (not all(isinstance(value, str) for value in key) or
                key[0] not in inputs or key[1] not in (inputs if batch else targets) or
                (batch and key[0] == key[1])):
            raise ValueError('Unknown discovered overlap candidate/page or candidate outside review coverage')
        pair = discovery_key(row)
        if pair in seen:
            raise ValueError('Duplicate discovered overlap pair')
        seen.add(pair)
        if not isinstance(row.get('rationale'), str) or not row['rationale'].strip():
            raise ValueError('Discovered overlap requires a rationale')
        candidate = inputs[key[0]]
        input_receipts = definition_receipts(candidate, checkout)
        page = {} if batch else targets[key[1]]
        target_receipts = (definition_receipts(inputs[key[1]], checkout) if batch else [page_receipt(page)])
        for asset in page.get('code_assets', []):
            target_receipts.append({'path': asset['path'], 'sha256': asset['sha256'], 'locator': page['page_id']})
            for entry in asset['entries']:
                if sum(e['name'] == entry['name'] for e in asset['entries']) != 1:
                    continue
                target_receipts.append({'path': asset['path'], 'sha256': asset['sha256'], 'locator': entry['name']})
        for field, expected, base in [('input_evidence', input_receipts, checkout),
                                      ('matched_evidence', target_receipts, checkout if batch else root)]:
            receipts = row.get(field)
            if not isinstance(receipts, list) or not receipts:
                raise ValueError('Discovered overlap requires source and matched evidence')
            for receipt in receipts:
                if receipt not in expected:
                    raise ValueError('Discovered overlap must cite exact known definition/page/asset receipts and locators')
                if not valid_receipt(contained(base, receipt['path']), receipt):
                    raise ValueError('Stale discovered overlap evidence hash')


def check_discovery_input(run, request):
    if load_discoveries(run) != request.get('discovered_overlaps'):
        raise ValueError('Discovered overlaps changed; prepare current review materials again')


def retain_update_discoveries(root, run, baseline, report, inventory):
    """Carry navigation onto all observed successors; never infer identity."""
    from .match import existing_index
    previous_request = read(baseline / 'review-request.yaml')
    previous = load_discoveries(baseline)
    if previous_request:
        check_discovery_input(baseline, previous_request)
    baseline_update = read(baseline / 'update.yaml')
    unresolved = baseline_update.get('unresolved_discovered_overlaps', [])
    if previous is None and not unresolved:
        return
    previous = previous or {'version': 1, 'entries': []}
    if load_discoveries(run) is not None:
        raise ValueError('Update requires an untouched discovered-overlaps input')
    old_inventory = read(baseline / 'inventory.yaml')
    old_candidates = {c['candidate_id']: c for c in old_inventory['candidates']}
    # Prepared historical receipts were validated at preparation. Unprepared
    # discoveries must still establish their source and current target receipts.
    old_pages = previous_request.get('existing_pages') if previous_request else existing_index(root)
    if not previous_request:
        validate_discoveries(previous, old_inventory['candidates'], old_pages, root,
                             Path(read(baseline / 'snapshot.yaml')['checkout']))
    successors = {}
    for pair in report['unchanged'] + report['changed']:
        successors.setdefault(pair['previous'], set()).add(pair['current'])
    for group in report['ambiguous_identities']:
        for identity in group['previous']:
            successors.setdefault(identity, set()).update(group['current'])
    for rename in report['possible_renames']:
        successors.setdefault(rename['previous'], set()).update(rename['candidates'])
    pages = {p['page_id']: p for p in existing_index(root)}
    current = {c['candidate_id']: c for c in inventory['candidates']}
    rows, seen, history, pending = [], set(), [], []
    previous_rows = {discovery_key(r): r for r in unresolved + previous['entries']}
    for row in previous_rows.values():
        if row['candidate_id'] not in old_candidates:
            raise ValueError('Unknown baseline discovered overlap candidate')
        targets = sorted(successors.get(row['candidate_id'], set()))
        if 'target_candidate_id' in row:
            other = row['target_candidate_id']
            if other not in old_candidates:
                raise ValueError('Unknown baseline discovered overlap target candidate')
            target_successors = sorted(successors.get(other, set()))
            history.append({'previous': row, 'successors': targets, 'target_successors': target_successors,
                            'status': 'retained_navigation_only' if targets and target_successors else 'no_observed_successor'})
            for identity in targets:
                for target in target_successors:
                    renewed = {**copy.deepcopy(row), 'candidate_id': identity, 'target_candidate_id': target}
                    key = discovery_key(renewed)
                    if identity == target or key in seen:
                        continue
                    seen.add(key)
                    unchanged = (report['id_map'].get(row['candidate_id']) == identity and
                                 report['id_map'].get(other) == target and not report['incompatible_scan'])
                    if unchanged:
                        try:
                            validate_discoveries({'version': 1, 'entries': [renewed]}, inventory['candidates'],
                                                 list(pages.values()), root,
                                                 Path(read(run / 'snapshot.yaml')['checkout']))
                        except (ValueError, OSError, KeyError):
                            unchanged = False
                    if not unchanged:
                        renewed.update(rationale='Retained comparison request from update baseline; navigation only. ' + row['rationale'],
                                       input_evidence=[definition_receipt(current[identity])],
                                       matched_evidence=[definition_receipt(current[target])])
                    rows.append(renewed)
            continue
        status = ('missing_current_page' if row['page_id'] not in pages else
                  'no_observed_successor' if not targets else 'retained_navigation_only')
        history.append({'previous': row, 'successors': targets, 'status': status})
        if status == 'missing_current_page':
            pending.extend({**row, 'candidate_id': identity,
                            'input_evidence': [definition_receipt(current[identity])]} for identity in targets)
        if status != 'retained_navigation_only':
            continue
        for identity in targets:
            key = ('page', identity, row['page_id'])
            if key in seen:
                continue
            seen.add(key)
            if report['id_map'].get(row['candidate_id']) == identity and not report['incompatible_scan']:
                retained = copy.deepcopy(row)
                retained['candidate_id'] = identity
                try:
                    validate_discoveries({'version': 1, 'entries': [retained]}, inventory['candidates'],
                                         list(pages.values()), root,
                                         Path(read(run / 'snapshot.yaml')['checkout']))
                except (ValueError, OSError, KeyError):
                    pass  # Changed target receipts need a newly reviewed comparison.
                else:
                    rows.append(retained)
                    continue
            rows.append({'candidate_id': identity, 'page_id': row['page_id'],
                         'rationale': 'Retained comparison request from update baseline; navigation only. ' + row['rationale'],
                         'input_evidence': [definition_receipt(current[identity])],
                         'matched_evidence': [page_receipt(pages[row['page_id']])]})
    document = {'version': 1, 'entries': rows}
    validate_discoveries(document, inventory['candidates'], list(pages.values()), root,
                         Path(read(run / 'snapshot.yaml')['checkout']))
    write(run / 'discovered-overlaps.yaml', document)
    report['discovered_overlap_history'] = history
    report['unresolved_discovered_overlaps'] = pending
    report['baseline_discovered_overlap_history'] = (
        baseline_update.get('baseline_discovered_overlap_history', []) +
        baseline_update.get('discovered_overlap_history', []))
