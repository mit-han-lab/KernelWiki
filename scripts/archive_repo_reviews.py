#!/usr/bin/env python3
"""Export or verify read-only review archives without importing kernels."""
import argparse
import json
import tarfile
from pathlib import Path

from repo_import.review_archive import export_reviews, pack_archive, verify_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--batch', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    export.add_argument('--public-source-catalog', type=Path, action='append', default=[],
                        help='Saved public Git object-store catalog for historical source receipts; never fetches')
    verify = commands.add_parser('verify')
    verify.add_argument('--archive', type=Path, required=True)
    pack = commands.add_parser('pack', help='Compress a verified archive into a new directory')
    pack.add_argument('--archive', type=Path, required=True)
    pack.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'export':
            result = export_reviews(Path(__file__).resolve().parents[1], args.batch, args.out,
                                    args.public_source_catalog)
        elif args.command == 'pack':
            result = pack_archive(args.archive, args.out)
        else:
            result = verify_archive(args.archive)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError, tarfile.TarError) as error:
        parser.exit(1, 'ERROR: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
