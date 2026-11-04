#!/usr/bin/env python3
"""Apply a completed ZED/xArm7 physical-validation record safely.

The record must contain explicit operator_result=true and an existing evidence
file for every physical gate.  This command never connects to hardware and
never accepts a partial record.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'open_tools_and_attachments_verified',
    'collision_geometry_verified',
    'controller_watchdog_verified',
    'joint_segment_tracking_verified',
    'servo_commissioned',
    'live_execution_observation',
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--config', type=Path,
                        default=ROOT/'config/dual_arm.zed_dual_xarm7_20261103.json')
    parser.add_argument('--output', type=Path,
                        help='new config path; defaults to updating --config with a timestamped backup')
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    config = json.loads(args.config.read_text())
    if record.get('schema_version') != 1:
        parser.error('unsupported validation record schema')
    if record.get('execution_allowed') is not False:
        parser.error('validation record must start with execution_allowed=false')
    if record.get('calibration_id') != config.get('calibration_id'):
        parser.error('validation record calibration_id does not match config')
    rows = {row.get('id'): row for row in record.get('checks', [])}
    missing = sorted(REQUIRED - rows.keys())
    if missing:
        parser.error('record is missing checks: ' + ', '.join(missing))
    failures = []
    for key in sorted(REQUIRED):
        row = rows[key]
        if row.get('operator_result') is not True:
            failures.append(f'{key}: operator_result must be true')
        evidence = row.get('evidence')
        if not isinstance(evidence, str) or not evidence.strip():
            failures.append(f'{key}: evidence path is required')
        else:
            path = Path(evidence).expanduser()
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            if not path.is_file():
                failures.append(f'{key}: evidence file does not exist: {path}')
    if failures:
        parser.error('validation not complete:\n  ' + '\n  '.join(failures))
    # Keep the existing safety model and provenance; only promote gates after
    # the complete record has passed the checks above.
    config['collision_geometry_verified'] = True
    config['servo_commissioned'] = True
    safety = config.setdefault('safety', {})
    for key in ('open_tools_and_attachments_verified',
                'controller_watchdog_verified',
                'joint_segment_tracking_verified'):
        safety[key] = True
    safety['validation_id'] = record.get('validation_id') or (
        'zed_dual_xarm7_physical_validation_' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    config['configuration_review']['physical_validation_record'] = str(args.record)
    config['configuration_review']['physical_validation_applied_at'] = datetime.now(timezone.utc).isoformat()
    config['configuration_review']['status'] = 'physical_validation_record_applied'
    config['geometry_provenance']['collision_geometry_verified'] = True
    destination = args.output or args.config
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == args.config.resolve():
        backup = destination.with_name(destination.name + '.' +
                                       datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.bak')
        shutil.copy2(destination, backup)
        print(f'backup={backup}')
    destination.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
