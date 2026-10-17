#!/usr/bin/env python3
"""Verify original run evidence or the small demo copies, without robot calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='restored experiment repository')
    parser.add_argument('--demo', type=Path, default=ROOT/'examples/graph_demo')
    parser.add_argument('--demo-only', action='store_true', help='verify small copies; no release download needed')
    args = parser.parse_args()
    records = json.loads((args.demo/'actual_cases.json').read_text())['files']
    failures, checked, skipped = [], 0, 0
    for row in records:
        if args.demo_only and 'demo_file' not in row:
            skipped += 1
            continue
        path = args.demo/row['demo_file'] if args.demo_only else args.root/row['source']
        checked += 1
        if not path.is_file():
            failures.append(dict(path=str(path), reason='missing'))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append(dict(path=str(path), reason='checksum mismatch'))
    print(json.dumps(dict(status='PASS' if not failures else 'FAIL', checked=checked,
                          omitted_release_arrays=skipped, failures=failures), ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
