#!/usr/bin/env python3
"""Prepare repo candidates; consume external decisions; publish validated pages."""
import argparse
import datetime
import json
import subprocess
import sys
import uuid
from pathlib import Path

from repo_import.checkout import git, resolve_snapshot
from repo_import.config import load_config, load_preset, merge
from repo_import.decisions import rebind_manifest, validate_decisions
from repo_import.materials import (prepare_review, prepare_selection, verify_inventory,
                                   review_context, reuse_update_reviews, export_audit_pack)
from repo_import.match import context_digest, existing_index, registry
from repo_import.publish import apply_plan, build_plan, validate_plan, recover_transaction
from repo_import.scan import scan_snapshot, SCAN_CONTRACT
from repo_import.selection import (propose_selection, validate_selection, representative_ids,
                                   review_matches, select_run)
from repo_import.store import contained, digest, file_hash, read, write, atomic_write

ROOT = Path(__file__).resolve().parents[1]


def storage(root, configured):
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def prepare(root, repo, revision, config, *, scan_only=False, fresh_review=False):
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    run = storage(root, config['state_dir']) / run_id
    run.mkdir(parents=True)
    write(run / 'result.yaml', {'run_id': run_id, 'status': 'preparing'})
    try:
        snapshot = resolve_snapshot(repo, revision, storage(root, config['checkout_dir']))
        inventory = scan_snapshot(snapshot, config['scan'])
        write(run / 'request.yaml', {'repo': repo, 'revision': revision, 'config': config,
                                     'fresh_review': fresh_review, 'extractor_contract': SCAN_CONTRACT})
        write(run / 'snapshot.yaml', snapshot)
        write(run / 'inventory.yaml', inventory)
        write(run / 'scan-receipt.yaml', {'version': 1, 'inventory_digest': digest(inventory),
                                        'snapshot_digest': digest(snapshot),
                                        'scan_config_digest': digest(config['scan']), 'extractor_contract': SCAN_CONTRACT})
        write(run / 'result.yaml', {'run_id': run_id, 'status': 'scanned',
                                   'definitions': len(inventory['candidates']),
                                   'parse_failures': len(inventory['failures'])})
        from repo_import.updates import record_observation
        record_observation(root, run)
        if scan_only:
            return run
        if config.get('selection', {}).get('enabled', True):
            prepare_selection(root, run)
        else:
            prepare_review(root, run, snapshot, inventory)
        return run
    except Exception as error:
        write(run / 'result.yaml', {'run_id': run_id, 'status': 'failed', 'error': str(error)})
        raise




def check_snapshot(request):
    snapshot = request['snapshot']
    checkout = Path(snapshot['checkout'])
    if git(checkout, 'rev-parse', 'HEAD') != snapshot['commit'] or git(checkout, 'status', '--porcelain'):
        raise ValueError('Pinned checkout was modified; prepare a new run')
    current = scan_snapshot(snapshot, request['scan_config'])
    if 'selection' in request:
        active = validate_selection(request['selection'], current, checkout)
        if request.get('inventory_digest') != digest(current):
            raise ValueError('Selection inventory changed')
        current = {'candidates': active, 'failures': current['failures']}
    if current != {k: request[k] for k in ('candidates', 'failures')}:
        raise ValueError('Snapshot receipts changed')


def check_context(root, request, run=None):
    if run is not None:
        from repo_import.discovery import check_discovery_input
        check_discovery_input(run, request)
    pages, registered = existing_index(root), registry(root)
    current_context, policy = review_context(root, pages, registered)
    expected_context = current_context if 'policy_receipts' in request else context_digest(pages, registered)
    if (request['context_digest'] != expected_context or
            ('policy_receipts' in request and request['policy_receipts'] != policy) or
            request['existing_pages'] != pages or request['registry'] != registered or
            request['matches'] != review_matches(request)):
        raise ValueError('Wiki or matching materials changed; prepare again')


def build(root, run):
    verify_inventory(run)
    request = read(run / 'review-request.yaml')
    if not request:
        raise ValueError('Complete selection and run select before build')
    if 'selection' in request and read(run / 'selection.yaml') != request['selection']:
        raise ValueError('Selection changed; run select again before build')
    check_snapshot(request)
    check_context(root, request, run)
    document = read(run / 'decisions.yaml')
    selected, pending = validate_decisions(document, request, root, Path(request['snapshot']['checkout']))
    plan = build_plan(root, run, request, document, selected)
    write(run / 'result.yaml', {**read(run / 'result.yaml'), 'run_id': request['run_id'], 'status': 'built',
                               'included': len(selected), 'pending': pending,
                               'included_representatives': sum(d['action'] != 'merge_candidate' for _, d in selected),
                               'included_members': sum(d['action'] == 'merge_candidate' for _, d in selected)})
    return plan


def apply(root, run):
    verify_inventory(run)
    # Restore interrupted writes before comparing the saved wiki context.
    if read(run / 'transaction.yaml').get('status') == 'applying':
        import fcntl
        lock = root / '.cache/repo-import/publish.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recover_transaction(root, run)
    if read(run / 'transaction.yaml').get('status') == 'committed':
        from repo_import.discovery import check_discovery_input
        check_discovery_input(run, read(run / 'review-request.yaml'))
        write(run / 'result.yaml', {**read(run / 'result.yaml'), 'status': 'applied'})
        record_published_state(root, run)
        print('Already applied; no changes')
        return
    request = read(run / 'review-request.yaml')
    if 'selection' in request and read(run / 'selection.yaml') != request['selection']:
        raise ValueError('Selection changed after build')
    check_snapshot(request)
    plan = read(run / 'publication-plan.yaml')
    if plan.get('run_id') != request['run_id']:
        raise ValueError('Build this run before applying it')
    if plan.get('request_digest') != digest(request) or plan.get('decisions_digest') != digest(read(run / 'decisions.yaml')):
        raise ValueError('Review materials changed after build; rebuild required')
    check_context(root, request, run)
    if not plan.get('validated_digest'):
        validate_plan(root, run, plan)
    apply_plan(root, run, plan)
    write(run / 'result.yaml', {**read(run / 'result.yaml'), 'status': 'applied'})
    record_published_state(root, run)


def record_published_state(root, run):
    """Only a committed apply can advance publication, never scan/report/build."""
    from repo_import.updates import record_publication
    return record_publication(root, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wiki-root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'scan', 'update'):
        prepare_parser = sub.add_parser(command)
        prepare_parser.add_argument('--repo', required=True)
        prepare_parser.add_argument('--revision')
        prepare_parser.add_argument('--fresh-review', action='store_true',
                                    help='Do not reuse previously saved admission/exclusion verdicts')
        if command == 'update':
            prepare_parser.add_argument('--baseline', required=True, type=Path)
    for command in ('prepare-selection', 'validate-selection', 'prepare-review', 'select', 'build', 'apply', 'status', 'report', 'export-audit'):
        child = sub.add_parser(command)
        child.add_argument('--run', required=True, help='Run ID or full persistent run directory')
        if command in {'report', 'export-audit'}:
            child.add_argument('--out', type=Path)
    preset_parser = sub.add_parser('import')
    preset_parser.add_argument('--preset', required=True)
    args = parser.parse_args()
    root = args.wiki_root.resolve()
    config = load_config(root, args.config)
    run = None
    try:
        if args.command in {'prepare', 'scan', 'update'}:
            run = prepare(root, args.repo, args.revision, config,
                          scan_only=args.command == 'scan', fresh_review=args.fresh_review)
            if args.command == 'update':
                from repo_import.updates import prepare_update
                prepare_update(root, run, args.baseline.resolve(), fresh_review=args.fresh_review)
        elif args.command == 'import':
            preset = load_preset(root, args.preset, config)
            run = prepare(root, preset['repo'], preset['revision'], merge(config, preset.get('config', {})))
            manifest = read(contained(root, preset['decision_manifest']))
            if 'discovered_overlaps' in manifest:
                write(run / 'discovered-overlaps.yaml', manifest['discovered_overlaps'])
            if 'selection' in manifest:
                write(run / 'selection.yaml', manifest['selection'])
                select_run(root, run, prepare_review)
            elif ('discovered_overlaps' in manifest and not
                  read(run / 'request.yaml')['config'].get('selection', {}).get('enabled', True)):
                prepare_review(root, run, read(run / 'snapshot.yaml'), read(run / 'inventory.yaml'))
            request = read(run / 'review-request.yaml')
            if not request:
                raise ValueError('Preset requires content-matching saved selection')
            document = rebind_manifest(manifest, request)
            write(run / 'decisions.yaml', document)
            selected, pending = validate_decisions(document, request, root, Path(request['snapshot']['checkout']))
            if pending:
                raise ValueError(f'{len(pending)} candidates need external review; materials: {run}')
            build(root, run)
            apply(root, run)
        else:
            path = Path(args.run).expanduser()
            run = path.resolve() if path.is_dir() else contained(storage(root, config['state_dir']), args.run)
            if args.command in {'select', 'prepare-review'}:
                select_run(root, run, prepare_review)
                if (run / 'update.yaml').exists() and not read(run / 'request.yaml').get('fresh_review'):
                    reuse_update_reviews(root, run)
            elif args.command == 'prepare-selection':
                prepare_selection(root, run)
            elif args.command == 'validate-selection':
                snapshot, inventory = verify_inventory(run)
                active = validate_selection(read(run / 'selection.yaml'), inventory, Path(snapshot['checkout']))
                write(run / 'selection-validation.yaml', {'status': 'passed',
                      'inventory_digest': digest(inventory), 'selection_digest': digest(read(run / 'selection.yaml')),
                      'active_definitions': len(active)})
            elif args.command == 'build':
                build(root, run)
            elif args.command == 'apply':
                apply(root, run)
            elif args.command == 'report':
                from repo_import.report import report_run
                print(json.dumps(report_run(root, run, (args.out or run / 'report').resolve()), indent=2))
            elif args.command == 'export-audit':
                export_audit_pack(root, run, args.out or run / 'audit-pack.yaml')
            else:
                print(json.dumps(read(run / 'result.yaml'), indent=2))
        if args.command != 'status':
            print(f'Run directory: {run}')
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        if run and run.exists():
            write(run / 'last-error.yaml', {'command': args.command, 'error': str(error)})
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
