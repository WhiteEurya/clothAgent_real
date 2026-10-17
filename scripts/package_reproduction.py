#!/usr/bin/env python3
"""Archive experiment evidence without rewriting frozen paths or hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import zstandard

CURRENT_GRAPH_RUNS = (
    'policy_replay_20261017T022854003479Z',
    'graph_collect_20261017_112855',
    'graph_recover_20261017_130922',
    'graph_recover_20261017_133136',
    'graph_recover_20261017_134332',
)
OMITTED_SUFFIXES = {'.mp4', '.avi', '.webm', '.svg', '.dot', '.html'}


class Parts:
    def __init__(self, output, limit=512 * 1024**2):
        self.output, self.limit = output, limit
        self.current = None
        self.rows = []
        self.size = 0

    def write(self, data):
        total = len(data)
        while data:
            if self.current is None:
                self.name = f'evidence.tar.zst.part{len(self.rows):03d}'
                self.current = (self.output / self.name).open('xb')
                self.digest = hashlib.sha256()
                self.size = 0
            piece = data[:self.limit-self.size]
            self.current.write(piece)
            self.digest.update(piece)
            self.size += len(piece)
            data = data[len(piece):]
            if self.size == self.limit:
                self.finish()
        return total

    def flush(self):
        if self.current:
            self.current.flush()

    def finish(self):
        if self.current:
            self.current.close()
            self.rows.append(dict(name=self.name, size=self.size, sha256=self.digest.hexdigest()))
            print('PACKED', self.name, self.size, flush=True)
            self.current = None


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [(args.root/'runs'/name, 'workspace/runs/'+name) for name in CURRENT_GRAPH_RUNS]
    sources.append((args.root/'config/dual_arm.local.json', 'workspace/config/dual_arm.local.json'))
    excluded = []
    files = []
    for source, prefix in sources:
        if not source.exists():
            continue
        for path in sorted(source.rglob('*')) if source.is_dir() else [source]:
            if path.is_file() and not any(p in ('__pycache__', '.pytest_cache', '.git') for p in path.parts):
                if path.suffix in OMITTED_SUFFIXES or 'live_graph' in path.parts:
                    excluded.append(dict(path=str(path.relative_to(args.root)), size=path.stat().st_size,
                                         reason='optional video or regenerable visualization'))
                    continue
                if path.is_symlink():
                    raise ValueError(f'symlink requires explicit review: {path}')
                name = prefix + ('/' + str(path.relative_to(source)) if source.is_dir() else '')
                files.append((path, name))
    total = sum(p.stat().st_size for p, _ in files)
    print('INVENTORY', len(files), 'files;', total, 'bytes', flush=True)
    parts = Parts(args.output)
    count = 0
    with (args.output/'files.jsonl').open('w') as inventory:
        with zstandard.ZstdCompressor(level=3, threads=4).stream_writer(parts, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for path, name in files:
                    info = archive.gettarinfo(str(path), arcname=name)
                    with path.open('rb') as source:
                        reader = HashReader(source)
                        archive.addfile(info, reader)
                    inventory.write(json.dumps(dict(path=name, size=info.size,
                                                    sha256=reader.digest.hexdigest()))+'\n')
                    count += 1
                    if count % 5000 == 0:
                        print('FILES', count, '/', len(files), flush=True)
    parts.finish()
    manifest = dict(schema_version=1, files=count, uncompressed_bytes=total,
                    archive_parts=parts.rows, included_runs=list(CURRENT_GRAPH_RUNS),
                    omitted_files=excluded, scope='current graph reproduction chain only',
                    original_workspace=str(args.root))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
