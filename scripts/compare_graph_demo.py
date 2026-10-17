#!/usr/bin/env python3
"""Compare a restored/dry-run graph with the published demonstration."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_snapshot(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if 'snapshot' in value:
        target = Path(value['snapshot'])
        if not target.is_absolute():
            target = path.parent/target
        if not target.exists():
            target = path.parent/target.name
        if digest(target) != value['sha256']:
            raise ValueError('snapshot pointer checksum mismatch')
        path = target
        value = json.loads(path.read_text())
    sidecar = Path(str(path)+'.sha256.json')
    if not sidecar.exists() or json.loads(sidecar.read_text())['sha256'] != digest(path):
        raise ValueError('frozen snapshot checksum missing or mismatched')
    return value


def canonical(value):
    """Only normalize the machine's workspace prefix and snapshot counter."""
    def normalize(v):
        if isinstance(v, dict):
            return {k: normalize(item) for k, item in v.items()}
        if isinstance(v, list):
            return [normalize(item) for item in v]
        if isinstance(v, str) and v.startswith('/') and '\n' not in v and '/runs/' in v:
            return 'runs/'+v.split('/runs/', 1)[1]
        return v
    return normalize({k: v for k, v in value.items() if k != 'version'})


def compare(candidate, expected):
    actual, reference = canonical(candidate), canonical(expected)
    different = sorted(k for k in actual.keys() | reference.keys()
                       if k not in actual or k not in reference or actual[k] != reference[k])
    return dict(status='PASS' if not different else 'FAIL', differing_sections=different,
                nodes=len(candidate['nodes']), actions=len(candidate['actions']),
                transitions=len(candidate['transitions']), snapshot_version=candidate['version'],
                ignored_differences=['snapshot version', 'absolute workspace prefix before /runs/'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='growth JSON or latest.json pointer')
    parser.add_argument('--reference', type=Path, default=ROOT/'examples/graph_demo/snapshot.json')
    args = parser.parse_args()
    result = compare(read_snapshot(args.snapshot), read_snapshot(args.reference))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
