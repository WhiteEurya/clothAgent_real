#!/usr/bin/env python3
"""Offline readiness inventory; never connects, moves, or enables a robot."""
import argparse
import json
from pathlib import Path


def audit(raw):
    issues = []

    def require(ok, field, action):
        if not ok:
            issues.append({'field': field, 'required_action': action})

    require(raw.get('calibration_status') == 'measured', 'calibration_status',
            'Validate the paired calibration on independent observations.')
    require(bool(raw.get('calibration_id')), 'calibration_id', 'Record calibration dataset ID.')
    require(raw.get('collision_geometry_verified') is True, 'collision_geometry_verified',
            'Measure installed open grippers, cameras and attachments against modeled envelopes.')
    require(raw.get('servo_commissioned') is True, 'servo_commissioned',
            'After measured bounds are available, complete supervised empty commissioning.')
    for arm in ('left', 'right'):
        a = raw.get('arms', {}).get(arm, {})
        for key in ('min_mm', 'max_mm'):
            require(a.get('workspace', {}).get(key) is not None,
                    f'arms.{arm}.workspace.{key}',
                    'Measure permitted TCP bounds in this robot base frame; route extrema are not safety bounds.')
    s = raw.get('safety', {})
    require(s.get('status') == 'measured', 'safety.status', 'Supply measured safety bounds with evidence.')
    require(bool(s.get('validation_id')), 'safety.validation_id', 'Link the measurement record.')
    for key, action in {
        'open_tools_and_attachments_verified': 'Verify envelopes with fully open jaws and installed attachments.',
        'controller_watchdog_verified': 'Verify device-side communication-loss stopping under a supervised test procedure.',
        'joint_segment_tracking_verified': 'Measure tracking during supervised low-speed motion.',
    }.items():
        require(s.get(key) is True, 'safety.' + key, action)
    for arm in ('left', 'right'):
        for key in ('base_error_mm', 'geometry_error_mm', 'max_joint_speed_deg_s', 'stop_excursion_deg'):
            require(s.get('arms', {}).get(arm, {}).get(key) is not None,
                    f'safety.arms.{arm}.{key}', 'Measure and record this bound; do not substitute command limits or zero.')
    return {'physical_execution': False, 'inventory_only': True,
            'note': 'Presence inventory, not numeric validation or execution authorization. Runtime checks remain mandatory.',
            'missing_count': len(issues), 'issues': issues}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/dual_arm.local.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"{report['missing_count']} missing entries; report: {args.output}")
