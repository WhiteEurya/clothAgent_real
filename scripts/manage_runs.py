#!/usr/bin/env python3
"""List and remove self-contained runs; deletion defaults to preview."""
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.run_storage import find_run, iter_runs, runs_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    listing = sub.add_parser('list')
    listing.add_argument('--size', action='store_true')
    deletion = sub.add_parser('delete')
    deletion.add_argument('run_id')
    deletion.add_argument('--confirm', help='Repeat exact ID to permanently delete; otherwise preview only')
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.command == 'check':
        target = runs_root(root, prepare=True)
        print(f'Run storage: {target}; free {shutil.disk_usage(target).free / 2**30:.1f} GiB')
    elif args.command == 'list':
        for run in sorted(iter_runs(root)):
            size = sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and not p.is_symlink()) if args.size else None
            print(f'{run.name}\t{size / 2**30:.3f} GiB\t{run}' if size is not None else f'{run.name}\t{run}')
    else:
        run = find_run(root, args.run_id)
        if run is None or not (run / 'run_metadata.json').is_file():
            parser.error('No complete run identity found for this ID')
        print(f'Permanent deletion target: {run}')
        if args.confirm != args.run_id:
            print('Preview only. Stop the run first; repeat with --confirm <exact run ID> to delete.')
            return
        shutil.rmtree(run)
        print('Deleted')


if __name__ == '__main__':
    main()
