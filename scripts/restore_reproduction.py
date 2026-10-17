#!/usr/bin/env python3
"""Download and verify the experiment release; restore without changing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


class JoinedParts:
    def __init__(self, paths):
        self.paths, self.stream = iter(paths), None

    def read(self, size=-1):
        if size < 0:
            raise ValueError('stream must use bounded reads')
        pieces = []
        while size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            chunk = self.stream.read(size)
            if not chunk:
                self.stream.close()
                self.stream = None
                continue
            pieces.append(chunk)
            size -= len(chunk)
        return b''.join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ssd-root', type=Path, required=True,
                        help='host directory mounted at /mnt/newssd in the container')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.root/'reproduction/manifest.json').read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    for asset in manifest['archive_parts'] + [manifest['inventory']]:
        path = args.cache/asset['name']
        if not path.exists():
            print('Downloading', asset['name'], flush=True)
            pending = path.with_suffix(path.suffix+'.download')
            with urllib.request.urlopen(manifest['download_base']+'/'+asset['name']) as source, pending.open('wb') as target:
                shutil.copyfileobj(source, target, 1024**2)
            pending.replace(path)
        if path.stat().st_size != asset['size'] or sha256(path) != asset['sha256']:
            raise ValueError(f'asset checksum mismatch: {path}')
        print('Verified', asset['name'], flush=True)
    if args.verify_only:
        return
    import zstandard
    expected = {row['path']: row for row in map(json.loads, (args.cache/'files.jsonl').read_text().splitlines())}
    joined = JoinedParts([args.cache/a['name'] for a in manifest['archive_parts']])
    restored = 0
    with zstandard.ZstdDecompressor().stream_reader(joined) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name not in expected:
                    raise ValueError(f'unexpected archive member: {member.name}')
                if name.parts[0] == 'workspace':
                    base = args.root.resolve()
                elif name.parts[0] == 'ssd_runs':
                    base = (args.ssd_root/'sja/clothAgent_real/runs').resolve()
                else:
                    raise ValueError(f'unknown archive root: {name}')
                target = base.joinpath(*name.parts[1:])
                if not target.resolve().is_relative_to(base):
                    raise ValueError(f'archive target escapes root: {target}')
                row = expected.pop(member.name)
                if target.exists():
                    if sha256(target) != row['sha256']:
                        raise ValueError(f'refusing to overwrite different local data: {target}')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pending = target.with_name(target.name+'.restore')
                    with archive.extractfile(member) as source, pending.open('xb') as output:
                        shutil.copyfileobj(source, output, 1024**2)
                    if pending.stat().st_size != row['size'] or sha256(pending) != row['sha256']:
                        pending.unlink()
                        raise ValueError(f'evidence checksum mismatch: {target}')
                    pending.replace(target)
                restored += 1
    if expected:
        raise ValueError(f'archive missing {len(expected)} files')
    print('Restored and verified', restored, 'files')


if __name__ == '__main__':
    main()
