"""Validated, list-only exports; never build or apply publication artifacts."""
from collections import Counter
from pathlib import Path

from .decisions import validate_decisions
from .selection import validate_selection
from .store import atomic_write, digest, read, write


def report_run(root, run, out_dir):
    """Export effective implementation groups, retaining unfinished work honestly."""
    # Use the same source and wiki guards as build, without invoking build.
    from import_repo import check_context, check_snapshot
    root, run, out_dir = Path(root), Path(run), Path(out_dir)
    from .materials import verify_inventory
    verify_inventory(run)
    destination = out_dir.resolve()
    if destination == root.resolve() or any(destination.is_relative_to((root / name).resolve())
                                           for name in ('wiki', 'sources', 'data', 'queries')):
        raise ValueError('List-only reports must be outside corpus directories')
    request = read(run / 'review-request.yaml')
    if not request:
        raise ValueError('Complete selection and generate review materials before reporting')
    if destination.is_relative_to(Path(request['snapshot']['checkout']).resolve()):
        raise ValueError('List-only reports must not modify the pinned checkout')
    selection = request.get('selection')
    if selection is not None:
        if read(run / 'selection.yaml') != selection:
            raise ValueError('Selection changed; refresh review materials before reporting')
        validate_selection(selection, read(run / 'inventory.yaml'), Path(request['snapshot']['checkout']))
    check_snapshot(request)
    check_context(root, request, run)
    document = read(run / 'decisions.yaml')
    selected, pending = validate_decisions(document, request, root, Path(request['snapshot']['checkout']))
    candidates = {c['candidate_id']: c for c in request['candidates']}
    includes = {c['candidate_id']: (c, d) for c, d in selected}
    groups = {}
    for candidate, decision in selected:
        if decision['action'] == 'merge_candidate':
            continue
        identity = (decision.get('target_implementation_id') or decision.get('target_page_id') or
                    'group-' + digest({'repo': request['snapshot']['repo'], 'candidate': candidate['candidate_id']})[:24])
        group = groups.setdefault(identity, {'implementation_id': identity, 'representatives': [], 'members': []})
        group['representatives'].append({
            'candidate_id': candidate['candidate_id'], 'origin': candidate['origin'],
            'path': candidate['path'], 'entry': candidate['entry'], 'content_digest': candidate['content_digest'],
            'action': decision['action'], 'relation': decision['relation'], 'metadata': decision['metadata'],
            'rationale': decision['rationale'], 'decision_locator': 'decisions.yaml#' + candidate['candidate_id']})
    for candidate, decision in selected:
        if decision['action'] != 'merge_candidate':
            continue
        target, target_decision = includes[decision['target_candidate_id']]
        identity = (target_decision.get('target_implementation_id') or target_decision.get('target_page_id') or
                    'group-' + digest({'repo': request['snapshot']['repo'], 'candidate': target['candidate_id']})[:24])
        groups[identity]['members'].append({
            'candidate_id': candidate['candidate_id'], 'origin': candidate['origin'],
            'path': candidate['path'], 'entry': candidate['entry'], 'content_digest': candidate['content_digest'],
            'relation': decision['relation'], 'difference_summary': decision.get('selection_reason', decision['rationale'])})
    verdicts = document.get('decisions', [])
    unfinished = {d['candidate_id'] for d in verdicts if d.get('decision') == 'defer' and
                  d.get('rationale') == 'Pending external review'}
    seen = {d['candidate_id'] for d in verdicts}
    reviewable = ({r['candidate_id'] for r in selection['entries'] if r['role'] == 'representative'}
                  if selection is not None else set(candidates))
    unfinished |= reviewable - seen
    roles = Counter(r['role'] for r in selection['entries']) if selection is not None else Counter()
    summary = {
        'version': 1, 'run_id': request['run_id'], 'repo': request['snapshot']['repo'],
        'commit': request['snapshot']['commit'], 'scan_config': request['scan_config'],
        'definitions': len(read(run / 'inventory.yaml').get('candidates', request['candidates'])),
        'parse_failures': len(request.get('failures', [])),
        'helpers': roles['helper'], 'representatives': roles['representative'] if selection is not None else len(candidates),
        'members': roles['member'], 'selection_pending': roles['pending'],
        'selection_unfinished': sum(r['role'] == 'pending' and not r.get('rationale', '').strip()
                                    for r in (selection or {}).get('entries', [])),
        'selection_defer': sum(r['role'] == 'pending' and bool(r.get('rationale', '').strip())
                               for r in (selection or {}).get('entries', [])),
        'reviewed_include': sum(d.get('decision') == 'include' for d in verdicts),
        'reviewed_exclude': sum(d.get('decision') == 'exclude' for d in verdicts),
        'review_defer': sum(d.get('decision') == 'defer' and d['candidate_id'] not in unfinished for d in verdicts),
        'review_unfinished': len(unfinished),
        'blocked_include': sum(d.get('decision') == 'include' and d['candidate_id'] not in includes for d in verdicts),
        'effective_kernels': len(groups), 'included_definitions': len(selected),
        'pending_definitions': len(pending),
        'request_digest': digest(request), 'decisions_digest': digest(document),
        'run_reference': str(run.resolve()),
        'counting_note': 'Per-repository effective implementation groups; counts across repositories are not a global unique total.'}
    ordered = [groups[key] for key in sorted(groups)]
    write(out_dir / 'summary.yaml', summary)
    write(out_dir / 'valid-kernels.yaml', {'version': 1, 'summary': summary, 'implementations': ordered})
    lines = ['# Validated kernel implementations', '', f"Repository: {summary['repo']}", '',
             f"Commit: `{summary['commit']}`", '',
             f"Effective implementations: **{len(groups)}**. This is a per-repository count, not a global unique total.", '',
             '| Implementation | Path | Entry | Action | Confidence / reproducibility |', '|---|---|---|---|---|']
    def cell(value):
        return str(value).replace('|', '&#124;').replace('\n', ' ').replace('`', '')
    for group in ordered:
        for representative in group['representatives']:
            metadata = representative['metadata']
            lines.append('| ' + ' | '.join(cell(value) for value in (
                group['implementation_id'], representative['path'], representative['entry'], representative['action'],
                metadata['confidence'] + ' / ' + metadata['reproducibility'])) + ' |')
        for member in group['members']:
            lines.append('| ' + ' | '.join(cell(value) for value in (
                group['implementation_id'], member['path'], member['entry'], member['relation'],
                'Member limits remain specific to this entry')) + ' |')
    atomic_write(out_dir / 'valid-kernels.md', '\n'.join(lines) + '\n')
    return summary
