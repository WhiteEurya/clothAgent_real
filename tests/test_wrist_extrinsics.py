from types import SimpleNamespace
import sys

import numpy as np
import pytest
import yaml

from cloth_agent.perception import PerceptionError, load_extrinsics


def test_fixed_camera_extrinsics_need_no_robot(tmp_path):
    path = tmp_path / "fixed.yaml"
    matrix = np.eye(4)
    matrix[0, 3] = 0.4
    path.write_text(yaml.safe_dump({"X_CammountCam": matrix.tolist()}))
    np.testing.assert_allclose(load_extrinsics(path), matrix)


@pytest.mark.parametrize("code", [0, 1])
def test_wrist_composes_mount_and_disconnects(tmp_path, monkeypatch, code):
    import yourdfpy

    path = tmp_path / "wrist.yaml"
    mount_camera = np.eye(4)
    mount_camera[0, 3] = 0.1
    path.write_text(yaml.safe_dump({
        "X_CammountCam": mount_camera.tolist(), "camera_mount": "link_eef",
        "robot_ip": "192.168.2.232", "robot_urdf": "robot.urdf",
    }))
    base_mount = np.array([[0., -1., 0., 0.4], [1., 0., 0., 0.2],
                           [0., 0., 1., 0.5], [0., 0., 0., 1.]])
    updates = []
    disconnected = []
    robot = SimpleNamespace(
        actuated_joint_names=[f"joint{i}" for i in range(1, 7)],
        update_cfg=updates.append,
        get_transform=lambda target, source: base_mount,
    )
    monkeypatch.setattr(yourdfpy.URDF, "load", lambda *a, **kw: robot)
    arm = SimpleNamespace(
        get_servo_angle=lambda **kw: (code, [0.1] * 6 + [0.]),
        disconnect=lambda: disconnected.append(True),
    )
    monkeypatch.setitem(sys.modules, "xarm.wrapper", SimpleNamespace(XArmAPI=lambda *a, **kw: arm))
    if code:
        with pytest.raises(PerceptionError, match="Cannot read"):
            load_extrinsics(path)
    else:
        result = load_extrinsics(path)
        np.testing.assert_allclose(result[:3, 3], [0.4, 0.3, 0.5])
        assert len(updates[0]) == 6
    assert disconnected == [True]
