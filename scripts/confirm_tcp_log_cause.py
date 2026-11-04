"""Read-only, spaced TCP/IK queries for correlation with controller log export."""
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import urllib.request

import numpy as np
from xarm.wrapper import XArmAPI


def main():
    root = Path(__file__).resolve().parents[1]
    trial = root / 'deployment-artifacts/topdown_physical_trial_001'
    out = trial / 'tcp_log_confirmation'
    out.mkdir(exist_ok=True)
    if (out / 'queries.json').exists():
        raise RuntimeError('Refusing to overwrite diagnostic evidence')
    prep = json.loads((trial / 'controller_preparation/preparation.json').read_text())
    identity = prep['before'][prep['selected_arm']]
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), queries=[],
                  physical_motion_commands=0, configuration_commands=0)
    arm = XArmAPI(identity['ip'], is_radian=False, do_not_open=True)
    try:
        arm.connect()
        time.sleep(.7)
        assert arm.get_robot_sn()[0] == 0
        assert arm.control_box_sn == identity['control_box_sn'] and arm.axis == 7
        code, q = arm.get_servo_angle(is_radian=False)
        assert code == 0 and np.max(np.abs(np.asarray(q)-identity['joints_deg'])) < .1
        assert arm.get_state() == (0, 2) and arm.get_err_warn_code() == (0, [0, 0])
        assert arm.mode == 0
        assert np.allclose(arm.tcp_offset, identity['tcp_offset_mm_deg'], atol=.1, rtol=0)
        assert np.allclose(arm.world_offset, identity['world_offset_mm_deg'], atol=.01, rtol=0)
        report.update(start_joints=q, firmware=arm.version, control_box_sn=arm.control_box_sn)
        for op in prep['operations']:
            if op['kind'] != 'move_tcp':
                continue
            pose = op['pose_base_mm_deg']
            row = dict(phase=op['phase'], pose=pose)
            time.sleep(.5)
            start = time.monotonic()
            row['tcp_limit'] = arm.is_tcp_limit(pose, is_radian=False)
            row['tcp_query_seconds'] = time.monotonic()-start
            time.sleep(.5)
            row['ik'] = arm.get_inverse_kinematics(pose, input_is_radian=False,
                return_is_radian=False, ref_angles=q)
            if row['ik'][0] == 0:
                row['joint_limit'] = arm.is_joint_limit(row['ik'][1], is_radian=False)
            report['queries'].append(row)
            print(json.dumps(row), flush=True)
        report['end_joints'] = arm.get_servo_angle(is_radian=False)
        report['state'] = arm.get_state()
        report['errors'] = arm.get_err_warn_code()
        assert report['end_joints'][0] == 0
        report['maximum_joint_change_deg'] = float(np.max(np.abs(np.asarray(report['end_joints'][1])-q)))
        assert report['maximum_joint_change_deg'] < .1
        report['status'] = 'READ_ONLY_COMPLETE'
    finally:
        arm.disconnect()
        (out / 'queries.json').write_text(json.dumps(report, indent=2)+'\n')
    # The same GET endpoint used by Studio's Log > Download button.
    with urllib.request.urlopen(f"http://{identity['ip']}:18333/handler/log/download", timeout=60) as response:
        (out / 'controller_log_export.tar.gz').write_bytes(response.read())


if __name__ == '__main__':
    main()
