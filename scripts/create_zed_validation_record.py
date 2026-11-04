#!/usr/bin/env python3
"""Create a dated checklist for the remaining physical ZED/xArm7 gates.

This command is offline: it reads configuration/evidence and writes a record
with blank operator-result fields. It never connects to or commands hardware.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT/'config/dual_arm.zed_dual_xarm7_20261103.json')
    parser.add_argument('--identity-report', type=Path,
                        default=ROOT/'outputs/startup_robot_state_001.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    cfg = json.loads(args.config.read_text())
    identity = json.loads(args.identity_report.read_text()) if args.identity_report.is_file() else None
    checks = [
        dict(id='open_tools_and_attachments_verified', operator_result=None,
             required_evidence='Photos/measurements of both open grippers, attachments, cables and fixed ZED support.'),
        dict(id='collision_geometry_verified', operator_result=None,
             required_evidence='Supervised low-speed clearance record for both arms, table, inactive arm and camera support.'),
        dict(id='controller_watchdog_verified', operator_result=None,
             required_evidence='Manufacturer-approved communication-loss stop test and controller log.'),
        dict(id='joint_segment_tracking_verified', operator_result=None,
             required_evidence='Empty low-speed explicit-joint path with commanded/feedback samples and max tracking error.'),
        dict(id='servo_commissioned', operator_result=None,
             required_evidence='Successful empty open-gripper +3 mm lift/return record after the preceding checks.'),
        dict(id='live_execution_observation', operator_result=None,
             required_evidence='Fresh ZED observation with current joint snapshot captured by the formal runtime.'),
    ]
    record = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'execution_allowed': False,
        'physical_commands_sent_by_this_command': 0,
        'config': str(args.config),
        'calibration_id': cfg.get('calibration_id'),
        'identity_report': str(args.identity_report),
        'identity_snapshot': identity.get('arms') if identity else None,
        'current_flags': {
            k: cfg.get(k) for k in ('calibration_status', 'collision_geometry_verified', 'servo_commissioned')
        },
        'safety_flags': {k: cfg.get('safety', {}).get(k) for k in (
            'status', 'validation_id', 'open_tools_and_attachments_verified',
            'controller_watchdog_verified', 'joint_segment_tracking_verified')},
        'checks': checks,
        'operator_notes': '',
        'instructions': 'Fill only operator_result/evidence paths after supervised physical tests; do not edit execution gates directly.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
