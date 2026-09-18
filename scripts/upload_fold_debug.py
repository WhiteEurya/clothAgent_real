#!/usr/bin/env python3
"""Package a fold run's lightweight diagnostics and upload them temporarily."""
from __future__ import annotations
import argparse, json, subprocess, tempfile, tarfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.run_storage import iter_runs

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--project-root', type=Path, default=Path('.'))
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--expiry-hours', type=int, default=1)
    a=p.parse_args(argv); root=a.project_root.resolve()
    run=a.run_dir.resolve() if a.run_dir else max(iter_runs(root), key=lambda x:x.stat().st_mtime)
    with tempfile.TemporaryDirectory(prefix='cloth_debug_') as td:
        staging=Path(td)/run.name; staging.mkdir()
        # Preserve actionable records and generated overlays; exclude large/raw data.
        for src in [run/'run_metadata.json', run/'summary.json']:
            if src.is_file(): (staging/src.name).write_bytes(src.read_bytes())
        for src in run.glob('results/fold_exploration/**/*.json'):
            rel=src.relative_to(run); dst=staging/rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(src.read_bytes())
        for src in run.glob('results/fold_exploration/**/*'):
            if src.is_file() and src.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:
                rel=src.relative_to(run); dst=staging/rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(src.read_bytes())
        archive=Path(td)/(run.name+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as tf: tf.add(staging, arcname=run.name)
        result=subprocess.run(['curl','-fsS','-X','POST','https://tempfile.org/api/upload/local','-F',f'files=@{archive}','-F',f'expiryHours={a.expiry_hours}'], text=True, capture_output=True, check=False)
        if result.returncode or not result.stdout.strip():
            print('DEBUG UPLOAD FAILED: '+(result.stderr.strip() or result.stdout.strip())); return 1
        try: payload=json.loads(result.stdout)
        except json.JSONDecodeError: print('DEBUG UPLOAD FAILED: invalid relay response'); return 1
        files=payload.get('files',[]) if isinstance(payload,dict) else []
        fid=files[0].get('id') if files and isinstance(files[0],dict) else payload.get('id') if isinstance(payload,dict) else None
        if not isinstance(fid,str) or not fid: print('DEBUG UPLOAD FAILED: relay response has no file id'); return 1
        print(f'DEBUG BUNDLE: {run}')
        print(f'DOWNLOAD URL: https://tempfile.org/{fid}/download')
    return 0
if __name__=='__main__': raise SystemExit(main())
