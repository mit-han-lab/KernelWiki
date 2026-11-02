"""Deterministic file materials for independent external review stages."""
import datetime
from pathlib import Path
from .discovery import load_discoveries, validate_discoveries
from .store import read, write, digest, atomic_write, file_hash
from .decisions import reuse_distinct_comparisons, reuse_record, validate_decisions
from .match import registry, existing_index, context_digest
from .selection import propose_selection, representative_ids, review_matches, validate_selection


def policy_receipts(root):
    return {name: file_hash(root / name) for name in
            ('CLAUDE.md', 'references/schema.md', 'data/tags.yaml', 'data/schemas.yaml')
            if (root / name).is_file()}


def review_context(root, pages, registered):
    policy = policy_receipts(root)
    base = context_digest(pages, registered)
    return (digest({'wiki_context': base, 'policy': policy}) if policy else base), policy


def verify_inventory(run):
    """Validate immutable stored scan inputs without editing a later stage."""
    inventory = read(run / 'inventory.yaml')
    snapshot = read(run / 'snapshot.yaml')
    saved = read(run / 'request.yaml')
    receipt = read(run / 'scan-receipt.yaml')
    if not inventory or not snapshot or not saved:
        raise ValueError('Prepare a complete scan before generating materials')
    expected = {'inventory_digest': digest(inventory), 'snapshot_digest': digest(snapshot),
                'scan_config_digest': digest(saved['config']['scan'])}
    if 'extractor_contract' in receipt:
        expected['extractor_contract'] = saved.get('extractor_contract')
    if receipt and (receipt.get('version') != 1 or any(receipt.get(key) != value for key, value in expected.items())):
        raise ValueError('Stored scan inputs changed; prepare a new run')
    return snapshot, inventory


def prepare_selection(root, run):
    """Generate a pending draft once; never overwrite external selection work."""
    snapshot, inventory = verify_inventory(run)
    if (run / 'selection.yaml').exists():
        if read(run / 'selection.yaml').get('inventory_digest') != digest(inventory):
            raise ValueError('Existing selection belongs to different scan inputs')
        return read(run / 'selection.yaml')
    selection = propose_selection(inventory, Path(snapshot['checkout']))
    write(run / 'selection.yaml', selection)
    guide = root / 'references/repo-import-review.md'
    if guide.exists():
        atomic_write(run / 'review-guide.md', guide.read_text())
    atomic_write(run / 'review.md', '# Repository selection\n\nRead review-guide.md and inventory.yaml. Save external selection.yaml; then prepare-review.\nHelpers need purpose/caller evidence, not full fact ledgers. Suggestions are not decisions.\n')
    write(run / 'result.yaml', {**read(run / 'result.yaml'), 'status': 'selection_pending',
                               'definitions': len(inventory['candidates']),
                               'parse_failures': len(inventory['failures'])})
    return selection


def prepare_review(root, run, snapshot, inventory, selection=None, *, fresh_review=None):
    run_id = run.name
    saved = read(run / 'request.yaml')
    config = saved['config']
    fresh_review = saved.get('fresh_review', False) if fresh_review is None else fresh_review
    registered, pages = registry(root), existing_index(root)
    current_context, policy = review_context(root, pages, registered)
    request = {'version': 1, 'run_id': run_id, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'snapshot': snapshot, 'scan_config': config['scan'], **inventory,
               'registry': registered, 'existing_pages': pages,
               'context_digest': current_context, 'policy_receipts': policy}
    if selection is not None:
        request['selection'] = selection
        request['inventory_digest'] = digest(read(run / 'inventory.yaml'))
    discoveries = load_discoveries(run)
    validate_discoveries(discoveries, inventory['candidates'], pages, root, Path(snapshot['checkout']))
    if discoveries is not None:
        request['discovered_overlaps'] = discoveries
    matches = review_matches(request)
    request['matches'] = matches
    write(run / 'review-request.yaml', request)
    write(run / 'matches.yaml', matches)
    guide = root / 'references/repo-import-review.md'
    if guide.exists():
        atomic_write(run / 'review-guide.md', guide.read_text())
    decisions = []
    prior_exclusions = {}
    prior_includes = {}
    prior_dir = root / 'data/repo-import/decisions' / snapshot['repo_id']
    if prior_dir.exists() and not fresh_review:
        for path in sorted(prior_dir.glob('*.yaml')):
            for decision in read(path).get('decisions', []):
                if decision.get('decision') == 'exclude':
                    prior_exclusions[decision['content_digest']] = decision
                elif decision.get('decision') == 'include':
                    prior_includes[decision['content_digest']] = decision
    for candidate in inventory['candidates']:
        if selection is not None and candidate['candidate_id'] not in representative_ids(selection):
            continue
        # A previously admitted unchanged same-repository version can be reused.
        reused = None
        for identity, implementation in ({} if fresh_review else registered['implementations']).items():
            for version_id, version in implementation['versions'].items():
                old = version.get('candidate')
                if old is None:
                    continue
                if (old['origin']['repo_id'] == candidate['origin']['repo_id'] and
                        old['content_digest'] == candidate['content_digest']):
                    reused = {**version['decision'], 'candidate_id': candidate['candidate_id'],
                              'relation': 'exact_duplicate', 'action': 'record_duplicate',
                              'target_implementation_id': identity, 'target_page_id': implementation['page_id'],
                              'target_version_id': version_id}
                    reused['overlap_resolutions'] = [
                        {'match_id': hint['match_id'], 'outcome': 'same_implementation',
                         'representative': implementation['page_id'],
                         'rationale': 'Unchanged previously admitted content; retain its canonical page.',
                         'evidence': [hint['evidence']]}
                        for hint in matches[candidate['candidate_id']]
                        if hint['kind'] == 'registry' and hint['implementation_id'] == identity]
                    reused['overlap_resolutions'] += reuse_distinct_comparisons(version['decision'], matches[candidate['candidate_id']], request)
                    break
        if not reused and candidate['content_digest'] in prior_exclusions:
            reused = {**prior_exclusions[candidate['content_digest']], 'candidate_id': candidate['candidate_id']}
        if not reused:
            reused = reuse_record(candidate, prior_includes.get(candidate['content_digest'], {}),
                                  request, matches[candidate['candidate_id']])
        if reused:
            try:
                _, pending = validate_decisions({'version': 1, 'context_digest': request['context_digest'],
                                    'decisions': [reused],
                                    **({'selection_digest': digest(selection)} if selection is not None else {})}, request, root, Path(snapshot['checkout']))
                if candidate['candidate_id'] in pending:
                    reused = None
            except ValueError:
                reused = None
        if reused:
            decisions.append(reused)
        else:
            decisions.append({'candidate_id': candidate['candidate_id'], 'content_digest': candidate['content_digest'],
                              'decision': 'defer', 'rationale': 'Pending external review',
                              'evidence': [{'path': candidate['path'], 'sha256': candidate['fingerprints']['source_sha256'],
                                            'locator': candidate['entry']}]})
    write(run / 'decisions.yaml', {'version': 1, 'context_digest': request['context_digest'], 'decisions': decisions,
                                  **({'selection_digest': digest(selection)} if selection is not None else {})})
    if selection is not None:
        rows = selection['entries']
        targets = []
        for candidate in inventory['candidates']:
            if candidate['candidate_id'] in representative_ids(selection):
                targets.append({**{k: candidate[k] for k in ('candidate_id', 'content_digest', 'path', 'entry', 'kind', 'line')},
                                'members': [r for r in rows if r['role'] == 'member' and r['representative'] == candidate['candidate_id']]})
        write(run / 'review-targets.yaml', {'version': 1, 'selection_digest': digest(selection),
                                          'context_digest': request['context_digest'], 'representatives': targets})
    atomic_write(run / 'review.md', '# External kernel review\n\nRead review-guide.md (or references/repo-import-review.md in KernelWiki).\n'
                 'Start with review-targets.yaml when present, then matches.yaml and the pinned checkout.\n'
                 'review-request.yaml is the complete machine validation context.\n'
                 'Edit decisions.yaml; do not change the source snapshot.\n'
                 'Save original agent input/output under review-attachments/ if available.\n')
    write(run / 'result.yaml', {'run_id': run_id, 'status': 'pending', 'candidates': len(inventory['candidates']),
                               'parse_failures': len(inventory['failures']),
                               'representatives': len(decisions)})


def reuse_update_reviews(root, run):
    """Rebind complete unchanged groups and comparisons, then validate anew."""
    import copy
    from .updates import remap_selection_group, remap_review_candidate
    update = read(run / 'update.yaml')
    saved = read(run / 'request.yaml')
    if not update or saved.get('fresh_review') or update.get('incompatible_scan'):
        return []
    baseline = Path(update['baseline_run'])
    old = read(baseline / 'review-request.yaml')
    request = read(run / 'review-request.yaml')
    if (not old or old.get('context_digest') != request.get('context_digest') or
            old.get('policy_receipts') != request.get('policy_receipts')):
        return []
    id_map = update['id_map']
    document = read(run / 'decisions.yaml')
    matches = review_matches(request)
    previous_hints = old.get('matches', {})
    old_selection = old.get('selection', {})
    current_selection = request.get('selection', {})

    def comparison_key(hint, rebind=False):
        value = {k: copy.deepcopy(v) for k, v in hint.items() if k != 'match_id'}
        if rebind and value.get('kind') == 'batch':
            target = id_map.get(value['candidate_id'])
            if target is None:
                return None
            value['candidate_id'] = target
        if rebind and value.get('discovered_overlap'):
            discovery = value['discovered_overlap']
            target = id_map.get(discovery['candidate_id'])
            if target is None:
                return None
            discovery['candidate_id'] = target
            if 'target_candidate_id' in discovery:
                target = id_map.get(discovery['target_candidate_id'])
                if target is None:
                    return None
                discovery['target_candidate_id'] = target
        return digest(value)

    reused = []
    old_verdicts = {d['candidate_id']: d for d in read(baseline / 'decisions.yaml').get('decisions', [])}
    for index, template in enumerate(document['decisions']):
        old_id = next((key for key, value in id_map.items() if value == template['candidate_id']), None)
        verdict = old_verdicts.get(old_id)
        if not verdict or verdict.get('decision') == 'defer':
            continue
        rebound_group = remap_selection_group(old_selection, old_id, id_map)
        current_group = [row for row in current_selection.get('entries', [])
                         if row['candidate_id'] == template['candidate_id'] or
                         (row.get('role') == 'member' and row.get('representative') == template['candidate_id'])]
        if rebound_group is None or sorted(rebound_group, key=lambda r:r['candidate_id']) != sorted(current_group, key=lambda r:r['candidate_id']):
            continue
        renewed = remap_review_candidate(verdict, id_map)
        if renewed is None:
            continue
        old_hints = {hint['match_id']: hint for hint in previous_hints.get(old_id, [])}
        new_hints = {comparison_key(hint): hint for hint in matches[template['candidate_id']]}
        resolutions = []
        for resolution in verdict.get('overlap_resolutions', []):
            old_hint = old_hints.get(resolution['match_id'])
            replacement = new_hints.get(comparison_key(old_hint, True)) if old_hint else None
            if replacement is None:
                continue
            row = copy.deepcopy(resolution)
            row['match_id'] = replacement['match_id']
            if row.get('representative') in id_map:
                row['representative'] = id_map[row['representative']]
            resolutions.append(row)
        renewed['overlap_resolutions'] = resolutions
        trial = copy.deepcopy(document)
        trial['decisions'][index] = renewed
        try:
            _, pending = validate_decisions(trial, request, root, Path(request['snapshot']['checkout']), _matches=matches)
        except (ValueError, KeyError, OSError):
            continue
        if renewed['candidate_id'] not in pending:
            document = trial
            reused.append(renewed['candidate_id'])
    write(run / 'decisions.yaml', document)
    write(run / 'update-review-reuse.yaml', {'version': 1, 'reused': reused,
                                          'context_digest': request['context_digest'],
                                          'selection_digest': digest(current_selection)})
    return reused


def export_audit_pack(root, run, destination):
    """Explicit portable review input, separate from compact publication output."""
    from import_repo import check_snapshot, check_context
    verify_inventory(run)
    request = read(run / 'review-request.yaml')
    if not request:
        raise ValueError('Generate detailed review materials before exporting an audit pack')
    check_snapshot(request)
    check_context(root, request, run)
    document = read(run / 'decisions.yaml')
    validate_decisions(document, request, root, Path(request['snapshot']['checkout']))
    pack = {**document, 'snapshot': request['snapshot'],
            'scan_config': request['scan_config'], 'audit_pack_version': 1}
    if 'discovered_overlaps' in request:
        pack['discovered_overlaps'] = request['discovered_overlaps']
    if 'selection' in request:
        if read(run / 'selection.yaml') != request['selection']:
            raise ValueError('Selection changed; prepare current review materials')
        pack['selection'] = request['selection']
    destination = Path(destination).resolve()
    if any(destination.is_relative_to((root / name).resolve()) for name in ('wiki', 'sources', 'data', 'queries')):
        raise ValueError('Full audit packs belong outside formal corpus outputs')
    if destination.is_relative_to(Path(request['snapshot']['checkout']).resolve()):
        raise ValueError('Cannot write review packs into the pinned checkout')
    write(destination, pack)
    return pack
