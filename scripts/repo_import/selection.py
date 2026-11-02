"""File-based selection of launchable representatives before detailed review.

Suggestions carry no authority. Every omitted helper and merged member requires
an explicit content-bound external decision. Unknowns remain pending.
"""
import ast
import copy
from collections import defaultdict
from pathlib import Path

from .store import contained, digest, file_hash, read, write, atomic_write, valid_receipt
from .match import overlap_index

KINDS = {'cuda_global', 'cute_kernel', 'triton_jit'}
ROLES = {'representative', 'member', 'helper', 'pending'}


def call_hints(checkout, candidates):
    """Same-file Python calls are navigation hints, never absence-of-launch proof."""
    result = defaultdict(list)
    paths = sorted({c['path'] for c in candidates if c['path'].endswith('.py')})
    for path in paths:
        try:
            tree = ast.parse(contained(checkout, path).read_text())
        except (SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            launch = isinstance(func, ast.Subscript)
            if launch:
                func = func.value
            if isinstance(func, ast.Name):
                result[(path, func.id)].append({'line': node.lineno,
                                                'kind': 'grid_launch' if launch else 'direct_call'})
    return result


def propose_selection(inventory, checkout):
    candidates = inventory['candidates']
    calls = call_hints(checkout, candidates)
    buckets = defaultdict(list)
    for c in candidates:
        # Small navigation buckets, not transitive semantic groups or verdicts.
        buckets[(c['path'], c['kind'])].append(c['candidate_id'])
    return {'version': 1, 'inventory_digest': digest(inventory),
            'suggestions': [{'basis': 'same_file_and_kind', 'candidates': members}
                            for members in buckets.values() if len(members) > 1],
            'entries': [{'candidate_id': c['candidate_id'], 'content_digest': c['content_digest'],
                         'path': c['path'], 'entry': c['entry'], 'kind': c['kind'],
                         'role': 'pending', 'rationale': '', 'evidence': [],
                         'call_hints': calls.get((c['path'], c['entry']), [])} for c in candidates]}


def representative_ids(selection):
    return {row['candidate_id'] for row in selection['entries'] if row['role'] == 'representative'}


def validate_selection(selection, inventory, checkout):
    if selection.get('version') != 1 or selection.get('inventory_digest') != digest(inventory):
        raise ValueError('Selection has stale inventory binding')
    candidates = {c['candidate_id']: c for c in inventory['candidates']}
    rows = {}
    for row in selection.get('entries', []):
        identity = row.get('candidate_id')
        if identity not in candidates or identity in rows:
            raise ValueError('Unknown or duplicate selection member')
        rows[identity] = row
        if row.get('content_digest') != candidates[identity]['content_digest']:
            raise ValueError('Stale selection member content')
        if row.get('role') not in ROLES:
            raise ValueError('Invalid selection role')
        if row['role'] == 'pending':
            continue
        if not isinstance(row.get('rationale'), str) or not row['rationale'].strip() or not row.get('evidence'):
            raise ValueError('Selection requires a rationale and source evidence')
        source = candidates[identity]
        if not any(e.get('path') == source['path'] and
                   e.get('sha256') == source['fingerprints']['source_sha256'] for e in row['evidence']):
            raise ValueError('Selection must cite the member definition')
        for receipt in row['evidence']:
            if not valid_receipt(contained(checkout, receipt.get('path', '')), receipt):
                raise ValueError('Invalid selection evidence receipt')
        if row['role'] == 'helper' and not row.get('caller_basis'):
            raise ValueError('Helper exclusion requires caller/launch analysis')
    if set(rows) != set(candidates):
        raise ValueError('Selection must cover every scanned definition')
    for identity, row in rows.items():
        if row['role'] != 'member':
            continue
        target = rows.get(row.get('representative'), {})
        if target.get('role') != 'representative' or row['representative'] == identity:
            raise ValueError('Members must target a representative directly')
        if row.get('relation') not in {'exact_duplicate', 'variant', 'new_version'} or not row.get('difference_summary'):
            raise ValueError('Members require a relation and an explicit difference summary')
        target_source = candidates[row['representative']]
        if not any(e.get('path') == target_source['path'] and
                   e.get('sha256') == target_source['fingerprints']['source_sha256'] for e in row['evidence']):
            raise ValueError('Member comparison must cite the representative source')
    return [c for c in inventory['candidates'] if rows[c['candidate_id']]['role'] in {'representative', 'member'}]


def review_matches(request):
    """Project ALL member hints onto representatives; never lose external matches."""
    selection = request.get('selection')
    if selection is None:
        return overlap_index(request['candidates'], request['registry'], request['existing_pages'],
                             discoveries=request.get('discovered_overlaps'))
    owners = {r['candidate_id']: r.get('representative') if r['role'] == 'member' else r['candidate_id']
              for r in selection['entries'] if r['role'] in {'representative', 'member'}}
    raw = overlap_index(request['candidates'], request['registry'], request['existing_pages'],
                        owners=owners, discoveries=request.get('discovered_overlaps'))
    result = {c['candidate_id']: [] for c in request['candidates']}
    seen = defaultdict(set)
    for identity, hints in raw.items():
        owner = owners[identity]
        for hint in hints:
            hint = copy.deepcopy(hint)
            if hint['kind'] == 'batch':
                target = owners[hint['candidate_id']]
                if target == owner:
                    continue  # Explicit source-bound member relation replaces this pair.
                hint['candidate_id'] = target
            hint.pop('match_id', None)
            key = digest(hint)
            if key not in seen[owner]:
                seen[owner].add(key)
                hint['match_id'] = 'overlap-' + digest({'candidate': owner, **hint})[:24]
                result[owner].append(hint)
    return result


def expand_members(document, request):
    """Compact relationship records do not inherit a representative's metrics."""
    if 'selection' not in request:
        return document
    document = copy.deepcopy(document)
    decisions = {d['candidate_id']: d for d in document.get('decisions', [])}
    for row in request['selection']['entries']:
        if row['role'] == 'representative' and decisions.get(row['candidate_id'], {}).get('action') == 'merge_candidate':
            raise ValueError('Selected representatives cannot merge; revise selection')
        if row['role'] != 'member':
            continue
        identity = row['candidate_id']
        if identity in decisions:
            raise ValueError('Member needs no separate verdict; split selection to review independently')
        representative = decisions.get(row['representative'], {})
        if representative.get('decision') == 'exclude':
            decisions[identity] = {
                'candidate_id': identity, 'content_digest': row['content_digest'],
                'decision': 'exclude', 'rationale': 'Excluded with its reviewed implementation group: ' + representative['rationale'],
                'evidence': row['evidence']}
            document.setdefault('decisions', []).append(decisions[identity])
            continue
        if representative.get('decision') != 'include':
            continue  # Unresolved groups remain pending; exclusion never admits members.
        if representative.get('action') == 'merge_candidate':
            raise ValueError('Selected representatives cannot merge; revise selection')
        metadata = copy.deepcopy(representative.get('metadata', {}))
        metadata.update(reproducibility='snippet', confidence='inferred')
        decisions[identity] = {
            'candidate_id': identity, 'content_digest': row['content_digest'], 'decision': 'include',
            'action': 'merge_candidate', 'relation': row['relation'],
            'target_candidate_id': row['representative'], 'rationale': row['rationale'],
            'selection_reason': row['difference_summary'], 'evidence': row['evidence'],
            'metadata': metadata, 'confidence_basis': 'Source relationship reviewed during selection; no execution or performance inheritance.'}
        document.setdefault('decisions', []).append(decisions[identity])
    return document


def select_run(root, run, prepare_review=None):
    from .materials import verify_inventory
    verify_inventory(run)
    if prepare_review is None:
        from .materials import prepare_review
    saved = read(run / 'request.yaml')
    if (run / 'transaction.yaml').exists():
        raise ValueError('Cannot revise selection after publication; prepare a new run')
    snapshot, inventory = read(run / 'snapshot.yaml'), read(run / 'inventory.yaml')
    # Re-scan before accepting externally edited selection, including receipts.
    from .scan import scan_snapshot
    from .checkout import git
    checkout = Path(snapshot['checkout'])
    if git(checkout, 'rev-parse', 'HEAD') != snapshot['commit'] or git(checkout, 'status', '--porcelain'):
        raise ValueError('Pinned checkout was modified')
    if scan_snapshot(snapshot, saved['config']['scan']) != inventory:
        raise ValueError('Snapshot receipts changed')
    selection = read(run / 'selection.yaml')
    active = validate_selection(selection, inventory, checkout)
    # Keep old detailed review material available when a reviewer splits/merges.
    old = old_document = None
    if (run / 'review-request.yaml').exists():
        old_document = read(run / 'decisions.yaml')
        old = read(run / 'review-request.yaml')
        archive = run / 'review-attachments' / ('selection-' + digest(old)[:24])
        for name in ('review-request.yaml', 'decisions.yaml', 'matches.yaml', 'selection.yaml',
                     'review-targets.yaml'):
            value = old.get('selection') if name == 'selection.yaml' else read(run / name)
            if value is not None:
                write(archive / name, value)
        # Preserve the actual prompts read in this round before preparation
        # replaces them with the current guide. Prompt-only updates do not
        # invalidate verdicts, but must remain visible in the audit history.
        for name in ('review-guide.md', 'review.md'):
            if (run / name).exists():
                atomic_write(archive / name, (run / name).read_text())
    prepare_review(root, run, snapshot, {'candidates': active, 'failures': inventory['failures']}, selection)
    if old is not None and old.get('context_digest') == read(run / 'review-request.yaml')['context_digest']:
        reuse_unchanged_reviews(root, run, old, old_document)
    pending = [r['candidate_id'] for r in selection['entries'] if r['role'] == 'pending']
    result = read(run / 'result.yaml')
    write(run / 'result.yaml', {**result, 'definitions': len(inventory['candidates']), 'selection_pending': pending,
                               'helpers': sum(r['role'] == 'helper' for r in selection['entries']),
                               'members': sum(r['role'] == 'member' for r in selection['entries'])})


def reuse_unchanged_reviews(root, run, old, old_document):
    """Reuse only unchanged groups AND still-valid comparisons, never blind verdicts."""
    from .decisions import validate_decisions
    request = read(run / 'review-request.yaml')
    document = read(run / 'decisions.yaml')
    def groups(selection):
        result = defaultdict(list)
        for row in selection.get('entries', []):
            owner = row['representative'] if row['role'] == 'member' else row['candidate_id']
            result[owner].append(row)
        return result

    old_groups, current_groups = groups(old.get('selection', {})), groups(request['selection'])
    previous = {d['candidate_id']: d for d in old_document.get('decisions', [])}
    matches = None
    reusable = []
    for i, template in enumerate(document['decisions']):
        identity = template['candidate_id']
        verdict = previous.get(identity)
        if (verdict is None or verdict.get('decision') == 'defer' or
                old_groups[identity] != current_groups[identity]):
            continue
        if matches is None:
            matches = review_matches(request)
        if verdict.get('decision') == 'include' and (
                {r.get('match_id') for r in verdict.get('overlap_resolutions', [])} !=
                {h['match_id'] for h in matches[identity]}):
            continue  # New or removed comparisons require a renewed decision.
        reusable.append((i, verdict))

    if not reusable:
        return
    checkout = Path(request['snapshot']['checkout'])
    # The common case is a valid set of unchanged reviews. Validate it together
    # once, including member expansion and reciprocal/page-target relationships.
    # Revalidating the growing full document per verdict repeatedly hashes the
    # same sources and makes selection refresh quadratic in completed reviews.
    trial = copy.deepcopy(document)
    for i, verdict in reusable:
        trial['decisions'][i] = verdict
    try:
        _, pending = validate_decisions(trial, request, root, checkout, _matches=matches)
    except (ValueError, OSError, KeyError):
        pass
    else:
        if not {verdict['candidate_id'] for _, verdict in reusable} & set(pending):
            write(run / 'decisions.yaml', trial)
            return

    # Preserve independent valid reviews when one old record is malformed or
    # conflicts with another target. Every accepted trial still passes the full
    # validator; the batch path never weakens validation or evidence checks.
    for i, verdict in reusable:
        identity = verdict['candidate_id']
        trial = copy.deepcopy(document)
        trial['decisions'][i] = verdict
        try:
            selected, pending = validate_decisions(trial, request, root, checkout,
                                                  _matches=matches)
        except (ValueError, OSError, KeyError):
            continue
        if identity not in pending:
            document = trial
    write(run / 'decisions.yaml', document)
