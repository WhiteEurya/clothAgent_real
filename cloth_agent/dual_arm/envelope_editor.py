"""Offline envelope accounting and explicitly unvalidated draft persistence."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np


REACTION_TERMS = (
    ('feedback', 'safety', 'max_feedback_age_s'),
    ('watchdog', 'safety', 'controller_watchdog_s'),
    ('stop_latency', 'safety', 'stop_command_latency_s'),
    ('tick_lateness', 'limits', 'max_tick_lateness_s'),
    ('dispatch', 'limits', 'dispatch_skew_s'),
)
STEP_LABELS = {
    'base': '基座误差', 'geometry': '几何误差', 'tracking_mm': '位置跟踪误差',
    'stop': '制动角位移', 'tracking_deg': '关节跟踪误差',
    'feedback': '反馈延迟', 'watchdog': '看门狗等待', 'stop_latency': '停止指令延迟',
    'tick': '控制周期', 'tick_lateness': '调度迟到', 'dispatch': '双臂发送时差',
}


def envelope_breakdown(raw, models, *, execution_mode='servo_stream'):
    """Scalar reference accounting; native stopping uses joint boxes instead."""
    output = {}
    for key, model in models.items():
        row = raw['safety']['arms'][key]
        axis = raw['arms'][key]['axis']
        ones = np.ones(axis)
        bounds = {'stop': model.capsule_motion_bounds(row['stop_excursion_deg']),
                  'tracking_deg': model.capsule_motion_bounds(ones * raw['limits']['tracking_error_deg'])}
        seconds = {name: raw[section][field] for name, section, field in REACTION_TERMS}
        seconds['tick'] = 1 / raw['limits']['rate_hz']
        if execution_mode == 'controller_sequential':
            seconds = {name: 0.0 for name in seconds}
        for name, delay in seconds.items():
            bounds[name] = model.capsule_motion_bounds(ones * row['max_joint_speed_deg_s'] * delay)
        output[key] = {}
        for cap in raw['arms'][key]['collision_capsules']:
            name = cap['name']
            terms = {'base': row['base_error_mm'], 'geometry': row['geometry_error_mm'],
                     'tracking_mm': raw['limits']['tracking_error_mm']}
            terms.update({label: values[name] for label, values in bounds.items()})
            output[key][name] = {'nominal_mm': cap['radius_mm'], 'increments_mm': terms,
                                 'total_mm': cap['radius_mm'] + sum(terms.values())}
    return output


def save_draft(raw, breakdown, path, *, execution_mode='servo_stream'):
    """Never overwrite the active configuration or claim edits were measured."""
    path = Path(path)
    if path.name != 'dual_arm.envelope_draft.json':
        raise ValueError('Envelope editor only saves dual_arm.envelope_draft.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    data = deepcopy(raw)
    data['safety']['status'] = 'estimated'
    data['safety']['validation_id'] = None
    data['collision_geometry_verified'] = False
    data['servo_commissioned'] = False
    for field in ('open_tools_and_attachments_verified', 'controller_watchdog_verified',
                  'joint_segment_tracking_verified'):
        data['safety'][field] = False
    data['envelope_editor'] = {'saved_at': stamp, 'status': 'unvalidated_draft',
                              'execution_mode': execution_mode,
                              'stopping_model': ('adaptive_joint_box' if execution_mode == 'controller_sequential'
                                                 else 'radial_padding'),
                              'breakdown_note': 'Scalar reference only; native angular stop/tracking ranges are checked by adaptive joint-box coverage.',
                              'radius_breakdown_mm': breakdown,
                              'note': 'Interactive visual edits; no physical validation or execution.'}
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + '.' + stamp + '.bak'))
    temp = path.with_name(path.name + '.' + stamp + '.tmp')
    temp.write_text(payload)
    temp.replace(path)
    return path
