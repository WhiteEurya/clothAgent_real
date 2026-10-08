"""List missing Home prerequisites without connecting to hardware.

This is a presence inventory only. Numerical validation and all runtime safety
checks remain authoritative, even when this list is empty.
"""


def home_missing_fields(raw, *, execution_mode='servo_stream'):
    missing = []
    for key, expected in [('calibration_status', 'measured'),
                          ('collision_geometry_verified', True),
                          ('servo_commissioned', True)]:
        if key == 'servo_commissioned' and execution_mode == 'controller_sequential':
            continue
        if raw.get(key) != expected:
            missing.append(key)
    if not raw.get('calibration_id'):
        missing.append('calibration_id')
    safety = raw.get('safety') or {}
    for key in ('status', 'validation_id', 'open_tools_and_attachments_verified',
                'controller_watchdog_verified', 'joint_segment_tracking_verified'):
        value = safety.get(key)
        ok = (value == 'measured' if key == 'status' else
              bool(value) if key == 'validation_id' else value is True)
        if not ok:
            missing.append('safety.' + key)
    for arm in ('left', 'right'):
        data = (raw.get('arms') or {}).get(arm) or {}
        for key in ('min_mm', 'max_mm'):
            if (data.get('workspace') or {}).get(key) is None:
                missing.append(f'arms.{arm}.workspace.{key}')
        bounds = (safety.get('arms') or {}).get(arm) or {}
        for key in ('base_error_mm', 'geometry_error_mm',
                    'max_joint_speed_deg_s', 'stop_excursion_deg'):
            if bounds.get(key) is None:
                missing.append(f'safety.arms.{arm}.{key}')
    return missing
