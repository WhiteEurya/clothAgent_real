#!/usr/bin/env python3
"""Offline dependency/resource checks. Never connects hardware or model services."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    checks = {}
    for name in ('numpy', 'PIL', 'scipy', 'cv2', 'jsonschema', 'yaml', 'requests',
                 'httpx', 'viser', 'yourdfpy', 'xarm', 'pyrealsense2', 'fcl',
                 'jax', 'pyroki', 'ompl'):
        try:
            importlib.import_module(name)
            checks[name] = {'ok': True}
        except Exception as exc:
            checks[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    for name in ('cloth_agent.experiments.minimal_one_shot_6iter',
                 'cloth_agent.experiments.automatic_top5_batch',
                 'cloth_agent.ggo.live', 'cloth_agent.cloth_harness.runtime.live'):
        try:
            importlib.import_module(name)
            checks[name] = {'ok': True}
        except Exception as exc:
            checks[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    try:
        from cloth_agent.dual_arm.observation import Observation
        from cloth_agent.run_storage import runs_root
        reference = ROOT / 'data/reference/deployment_example'
        obs = Observation.load(reference / 'observation')
        seed = json.loads((reference / 'ggo_annotations.json').read_text())
        digest = hashlib.sha256(obs.image.read_bytes()).hexdigest()
        if digest != seed['reference_rgb_sha256']:
            raise ValueError('Bundled reference / annotation hash mismatch')
        checks['reference'] = {'ok': True}
        checks['run_storage'] = {'ok': True, 'path': str(runs_root(ROOT))}
    except Exception as exc:
        checks['resources'] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps({'python': sys.executable, 'checks': checks,
                      'model_cli': {n: shutil.which(n) for n in ('codex', 'claude')},
                      'hardware_connected': False, 'model_calls': 0,
                      'note': 'Passing checks does not validate hardware calibration or model access.'},
                     ensure_ascii=False, indent=2))
    return 0 if all(v['ok'] for v in checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
