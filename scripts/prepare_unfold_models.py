"""Download pinned segmentation assets, or verify existing files; no hardware IO."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reproduction/dual_wrist_unfold_20261114/models.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for model in json.loads(MANIFEST.read_text()):
        directory = ROOT / model['directory']
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(model['repo_id'], revision=model['revision'],
                              local_dir=str(directory), allow_patterns=list(model['files']))
        for name, expected in model['files'].items():
            path = directory / name
            if not path.is_file() or digest(path) != expected:
                raise SystemExit(f'Model file missing or different from recorded experiment: {path}')
        print(f"Verified {model['repo_id']} ({len(model['files'])} files)")


if __name__ == '__main__':
    main()
