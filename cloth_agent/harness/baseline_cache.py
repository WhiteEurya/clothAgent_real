"""Verified reuse of completed baselines; cached timing is historical, never a new sample."""
from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
from pathlib import Path

from .common import digest, read_json, write_json


def cache_contract(frozen, baseline_hash, configuration, *, repeats, max_ops, replay_timeout, call_timeout):
    root = Path(__file__).resolve().parent
    files = list(root.rglob('*.py')) + [root.parent/'image_tools_mcp.py', root.parent/'planner_backend.py']
    code = {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {'schema_version': 1, 'evidence_hash': frozen['evidence_hash'], 'baseline_hash': baseline_hash,
            'model_configuration': configuration, 'repeats': repeats, 'max_ops': max_ops,
            'replay_timeout': replay_timeout, 'call_timeout': call_timeout, 'runtime_hash': digest(code)}


def tree_hashes(directory):
    hashes = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink(): raise ValueError('Cache cannot contain symlinks')
        if path.is_file() and path.name != 'cache_manifest.json':
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_baseline(directory, contract, output):
    entry = Path(directory)/digest(contract)
    if not entry.exists(): return [], {'status': 'MISS', 'key': entry.name}
    try:
        manifest = read_json(entry/'cache_manifest.json')
        if manifest['contract'] != contract or manifest['files'] != tree_hashes(entry):
            raise ValueError('Cache identity or artifact hashes changed')
        rows = read_json(entry/'rows.json')
        if len(rows) != contract['repeats'] or any(r['status'] != 'READY' or
                r['root_evidence_hash'] != contract['evidence_hash'] or
                r['harness_hash'] != contract['baseline_hash'] for r in rows):
            raise ValueError('Cache is not a complete matching baseline')
        # Copy immutable audit artifacts, not model responses into the candidate input.
        shutil.copytree(entry/'artifacts', Path(output)/'cached_baseline')
        for row in rows:
            row['baseline_cache'] = {'reused': True, 'source_run': manifest['source_run'], 'key': entry.name,
                                    'timing': 'Original measured duration; not a fresh model invocation'}
        return rows, {'status': 'HIT', 'key': entry.name, 'source_run': manifest['source_run'],
                      'historical_elapsed_s': sum(r['metrics']['elapsed_s'] for r in rows)}
    except (ValueError, KeyError, OSError) as exc:
        return [], {'status': 'INVALID', 'key': entry.name, 'reason': str(exc)}


def store_baseline(directory, contract, rows, output):
    if len(rows) != contract['repeats'] or any(r['status'] != 'READY' for r in rows): return False
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    entry = directory/digest(contract)
    if entry.exists(): return False
    temp = Path(tempfile.mkdtemp(prefix='.writing-', dir=directory))
    try:
        write_json(temp/'rows.json', copy.deepcopy(rows))
        shutil.copytree(Path(output)/'evidence', temp/'artifacts/evidence')
        for row in rows:
            rid = row['rollout_id']
            shutil.copytree(Path(output)/'replays'/rid, temp/'artifacts/replays'/rid)
        write_json(temp/'cache_manifest.json', {'contract': contract, 'files': tree_hashes(temp),
                   'source_run': str(Path(output).resolve())})
        try: temp.rename(entry)
        except FileExistsError: return False
        return True
    finally:
        if temp.exists(): shutil.rmtree(temp)


def load_baseline_run(source, contract, output):
    """Explicitly reuse a historical run across runtime updates, with matching inputs/settings."""
    from .candidate_patch import bundle_hash
    from .reasoning_contract import verify_evidence
    source = Path(source)
    report = read_json(source/'report.json')
    frozen = read_json(source/'evidence/evidence.json')
    verify_evidence(frozen, source/'evidence')
    if frozen['evidence_hash'] != contract['evidence_hash']:
        raise ValueError('Historical baseline evidence differs')
    if bundle_hash(read_json(source/'baseline_bundle.json')) != contract['baseline_hash']:
        raise ValueError('Historical baseline bundle differs')
    if report['model_configuration'] != contract['model_configuration']:
        raise ValueError('Historical baseline model configuration differs')
    for key in ('repeats','max_ops','replay_timeout','call_timeout'):
        if report['settings'][key] != contract[key]:
            raise ValueError(f'Historical baseline setting differs: {key}')
    rows = [copy.deepcopy(r) for r in report['rollouts'] if r['harness_hash']==contract['baseline_hash']]
    if len(rows)!=contract['repeats'] or any(r['status']!='READY' for r in rows):
        raise ValueError('Historical run has no complete READY baseline')
    artifacts = source/'cached_baseline' if rows[0].get('baseline_cache',{}).get('reused') else source
    for row in rows:
        path = artifacts/'replays'/row['rollout_id']
        saved = read_json(path/'result.json')
        if saved['action'] != row['action'] or saved['metrics'] != row['metrics']:
            raise ValueError('Historical baseline result differs from report')
        prepared = read_json(path/'prepared/evidence.json')
        verify_evidence(prepared, path/'prepared')
        if prepared['evidence_hash'] != row['derived_evidence_hash']:
            raise ValueError('Historical prepared evidence differs')
    for row in rows:
        rid = row['rollout_id']
        shutil.copytree(artifacts/'replays'/rid, Path(output)/'cached_baseline/replays'/rid)
        row['baseline_cache']={'reused':True,'source_run':str(source.resolve()),
            'timing':'Explicit historical measurement; runtime may differ'}
    return rows, {'status':'HIT','mode':'EXPLICIT_HISTORICAL_RUN','source_run':str(source.resolve()),
                  'historical_elapsed_s':sum(r['metrics']['elapsed_s'] for r in rows)}
