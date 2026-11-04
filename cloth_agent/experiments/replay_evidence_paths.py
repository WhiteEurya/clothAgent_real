"""Resolve restored release paths without rewriting frozen evidence bytes."""
import os
from pathlib import Path


def resolve_evidence_path(path):
    path = Path(path)
    root = os.environ.get('CLOTH_REPLAY_EVIDENCE_ROOT')
    if not root or not path.is_absolute():
        return path
    root = Path(root).resolve()
    for old, new in (('/home/sja/clothAgent_real', root / 'workspace'),
                     ('/mnt/newssd', root / 'ssd')):
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        target = new.joinpath(relative)
        if not target.resolve().is_relative_to(new.resolve()):
            raise ValueError('release evidence path escapes restored root')
        return target
    return path
