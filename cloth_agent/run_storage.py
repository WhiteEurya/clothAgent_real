"""Central run storage; configured SSD must be mounted before writing."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def runs_root(project_root: Path, *, prepare: bool = False) -> Path:
    config = project_root / 'config' / 'run_storage.json'
    settings = json.loads(config.read_text()) if config.is_file() else {}
    mount = settings.get('required_mount')
    if mount and not Path(mount).is_mount():
        raise RuntimeError(f'Result disk is not mounted: {mount}')
    root = Path(settings.get('runs_root', str(project_root / 'runs'))).expanduser().resolve()
    if prepare:
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            raise PermissionError(f'Result directory is not writable: {root}')
    return root


def storage_roots(project_root: Path) -> tuple[Path, ...]:
    return tuple(dict.fromkeys((runs_root(project_root), (project_root / 'runs').resolve())))


def validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_id must be one simple directory name')


def iter_runs(project_root: Path):
    seen = set()
    for root in storage_roots(project_root):
        for pattern in ('*/run_metadata.json', '*/*/run_metadata.json', '*/workspace', '*/*/workspace'):
            for metadata in sorted(root.glob(pattern)):
                run = metadata.parent.resolve()
                if root in run.parents and run not in seen:
                    seen.add(run)
                    yield run


def find_run(project_root: Path, run_id: str) -> Path | None:
    validate_run_id(run_id)
    matches = set()
    for root in storage_roots(project_root):
        for p in (root / run_id, *(date / run_id for date in root.glob('????-??-??') if date.is_dir())):
            if p.exists():
                resolved = p.resolve()
                if root not in resolved.parents:
                    raise PermissionError(f'Run escapes storage: {p}')
                matches.add(resolved)
    if len(matches) > 1:
        raise ValueError(f'Ambiguous run ID; use --run-dir: {run_id}')
    return next(iter(matches), None)


def new_run_path(project_root: Path, run_id: str) -> Path:
    validate_run_id(run_id)
    if find_run(project_root, run_id) is not None:
        raise FileExistsError(f'Run already exists: {run_id}')
    root = runs_root(project_root, prepare=True)
    # Unconfigured projects keep their historical layout (including test fixtures).
    if (project_root / 'config' / 'run_storage.json').is_file():
        root = root / datetime.now(timezone.utc).strftime('%Y-%m-%d')
    path = root / run_id
    if runs_root(project_root) not in path.resolve().parents:
        raise PermissionError('Run escapes storage')
    return path


def result_files(project_root: Path, pattern: str):
    """Find artifacts in both dated and historical run layouts."""
    seen = set()
    for root in storage_roots(project_root):
        for prefix in ('*/', '????-??-??/*/'):
            for path in root.glob(prefix + pattern):
                if path.resolve() not in seen:
                    seen.add(path.resolve())
                    yield path


def auxiliary_dir(project_root: Path, category: str) -> Path:
    """Separate camera/calibration output from experimental runs."""
    validate_run_id(category)
    root = runs_root(project_root, prepare=True)
    return root.parent / 'tools' / category / datetime.now(timezone.utc).strftime('%Y-%m-%d')
