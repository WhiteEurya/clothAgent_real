"""Synthetic geometry/fault injection only; no robot, camera or model service."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.dual_arm.cli import main
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.execution import (
    DualArmCoordinator,
    SimulatedConnection,
    XArmConnection,
)
from cloth_agent.dual_arm.geometry import (
    Capsule,
    DualArmError,
    check_collision,
    matrix_pose,
    pose_error,
    pose_matrix,
    segment_box_distance,
    segment_distance,
    transform,
)
from cloth_agent.dual_arm.model import VisionPlanner, require_hold
from cloth_agent.dual_arm.observation import Observation, save_observation
from cloth_agent.dual_arm.planning import (
    Motion,
    Phase,
    build_phases,
    compile_motion,
    compile_program,
    ground_targets,
    read_targets,
)
from cloth_agent.dual_arm.safety import synthetic_safety
from cloth_agent.dual_arm.setup import fit_base, initial_config

ROOT = Path(__file__).resolve().parents[1]
GOOD_HOLD = {
    "left_holding": True,
    "right_holding": True,
    "slip": False,
    "overstretched": False,
    "confidence": 1.0,
    "reason": "synthetic test evidence",
}


class CartesianModel:
    """Deliberately nonphysical model for deterministic coordinator tests."""

    def __init__(self, arm):
        self.arm = arm

    def forward(self, q):
        return np.asarray(q[:6], dtype=float)

    def inverse(self, p, seed):
        return np.r_[p, np.zeros(self.arm.axis - 6)]

    def capsules(self, q):
        p = (self.arm.world_from_base @ np.r_[q[:3], 1])[:3]
        return [Capsule("tool", p, p, 5)]

    def capsule_motion_bounds(self, delta):
        return {"tool": float(np.linalg.norm(delta[:3]))}


@pytest.fixture
def scene(tmp_path):
    raw = initial_config(ROOT, ROOT / "data/robot/dual_arm_home.json")
    raw.update(
        calibration_status="synthetic",
        calibration_id="test-calibration",
        arm_layout_description="test",
    )
    raw["safety"] = synthetic_safety(raw["arms"])
    for k, a in raw["arms"].items():
        a["world_from_base_mm"] = np.eye(4).tolist()
        a["workspace"] = {"min_mm": [0, -500, 0], "max_mm": [600, 500, 600]}
        a["home_joints_deg"] = [250, -120 if k == "left" else 120, 180, 0, 0, 0] + [
            0
        ] * (a["axis"] - 6)
        a["grasp_rpy_world_deg"] = [0, 0, 0]
        a["collision_capsules"] = [
            {
                "name": n,
                "frame": "link_eef",
                "start_mm": [0, 0, 0],
                "end_mm": [0, 0, 0],
                "radius_mm": 5,
            }
            for n in [
                "link_base",
                *[f"link{i}" for i in range(1, a["axis"] + 1)],
                "tool",
            ]
        ]
    config = DualConfig.parse(raw, ROOT)
    models = {k: CartesianModel(a) for k, a in config.arms.items()}
    cancel = threading.Event()
    connections = {
        k: SimulatedConnection(a, models[k], cancel) for k, a in config.arms.items()
    }
    _, x = np.mgrid[:64, :64]
    xyz = np.stack(
        [np.full_like(x, 250), (x - 16) * 10 - 120, np.full_like(x, 150)], axis=-1
    )
    obs = save_observation(
        tmp_path / "observation",
        np.zeros((64, 64, 3), dtype=np.uint8),
        xyz,
        config,
        {k: a.home_joints for k, a in config.arms.items()},
        synthetic=True,
    )
    proposal = {
        "schema_version": 2,
        "mode": "center_pair",
        "center": {"pixel_xy": [28, 32], "reason": "fixed middle"},
        "pin_arm": None,
        "observation_id": obs.meta["observation_id"],
        "grasps": {
            "left": {"pixel_xy": [16, 32], "reason": "left"},
            "right": {"pixel_xy": [40, 32], "reason": "right"},
        },
        "lift_mm": 15,
        "spread_mm": 10,
        "approach_mm": 25,
    }
    return SimpleNamespace(
        config=config,
        models=models,
        cancel=cancel,
        connections=connections,
        obs=obs,
        proposal=proposal,
        directory=tmp_path,
    )


def program(scene):
    initial = {k: a.home_joints.copy() for k, a in scene.config.arms.items()}
    world = {k: scene.models[k].forward(q) for k, q in initial.items()}
    targets = ground_targets(scene.proposal, scene.obs, scene.config)
    phases = build_phases(scene.proposal, targets, world, scene.config)
    return initial, compile_program(
        phases,
        initial,
        scene.config,
        scene.models,
        {k: c.inverse for k, c in scene.connections.items()},
    )


def coordinator(scene):
    return DualArmCoordinator(
        scene.config,
        scene.models,
        scene.connections,
        scene.cancel,
        scene.directory / "run",
        realtime=False,
    )


def test_rigid_transform_rotates_pose_not_only_xyz():
    frame = pose_matrix([200, -100, 50, 0, 0, 90])
    pose = matrix_pose(frame @ pose_matrix([100, 0, 10, 180, 0, 0]))
    assert pose[:3] == pytest.approx([200, 0, 60])
    assert pose_error(pose, [200, 0, 60, 180, 0, 90]) == pytest.approx([0, 0], abs=1e-6)
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(DualArmError):
        transform(reflected)


def test_capsule_crossing_and_box_interior():
    assert segment_distance([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]) == 0
    assert segment_distance([0, 0, 0], [0, 0, 0], [3, 0, 0], [4, 0, 0]) == 3
    assert segment_box_distance([-10, 0, 0], [10, 0, 0], [-1, -1, -1], [1, 1, 1]) == 0
    assert segment_box_distance([-10, 3, 0], [10, 3, 0], [-1, -1, -1], [1, 1, 1]) == 2
    shapes = {
        "left": [Capsule("forearm", np.array([-10, 0, 0]), np.array([10, 0, 0]), 1)],
        "right": [Capsule("forearm", np.array([0, -10, 0]), np.array([0, 10, 0]), 1)],
    }
    with pytest.raises(DualArmError, match="inter-arm"):
        check_collision(shapes, [], 1)


def test_template_cannot_be_used_for_execution(tmp_path):
    path = tmp_path / "config.json"
    assert main(["init-config", "--output", str(path)]) == 0
    with pytest.raises(DualArmError, match="complete the measured"):
        DualConfig.load(path, root=ROOT)
    assert (
        main(
            ["run", "--config", str(path), "--output", str(tmp_path / "run"), "--real"]
        )
        == 1
    )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "change", ["missing_arm", "same_ip", "bad_axis", "nan_transform", "missing_link"]
)
def test_invalid_config_rejected(scene, change):
    raw = json.loads(json.dumps(scene.config.raw))
    if change == "missing_arm":
        del raw["arms"]["right"]
    elif change == "same_ip":
        raw["arms"]["right"]["ip"] = raw["arms"]["left"]["ip"]
    elif change == "bad_axis":
        raw["arms"]["left"]["axis"] = 8
    elif change == "nan_transform":
        raw["arms"]["left"]["world_from_base_mm"][0][0] = float("nan")
    else:
        raw["arms"]["left"]["collision_capsules"] = []
    with pytest.raises(DualArmError):
        DualConfig.parse(raw, ROOT)


def test_observation_checks_integrity_depth_and_calibration(scene):
    scene.obs.validate_for(scene.config)
    with pytest.raises(DualArmError, match="real captured"):
        scene.obs.validate_for(scene.config, live=True)
    with pytest.raises(DualArmError, match="interior"):
        scene.obs.sample([0, 0], 8)
    scene.obs.xyz[30:35, 14:19] = np.nan
    with pytest.raises(DualArmError, match="depth"):
        ground_targets(scene.proposal, scene.obs, scene.config)
    scene.obs.image.write_bytes(b"changed")
    with pytest.raises(DualArmError, match="artifact changed"):
        Observation.load(scene.obs.directory)


def test_stale_or_foreign_observation_rejected(scene):
    scene.obs.meta["synthetic"] = False
    scene.obs.meta["captured_unix_s"] = time.time() - 10000
    with pytest.raises(DualArmError, match="expired"):
        scene.obs.validate_for(scene.config, live=True)
    scene.proposal["observation_id"] = "old"
    with pytest.raises(DualArmError, match="different observation"):
        ground_targets(scene.proposal, scene.obs, scene.config)


@pytest.mark.parametrize(
    "field,value",
    [("lift_mm", float("nan")), ("spread_mm", 10000), ("approach_mm", True)],
)
def test_unbounded_model_proposals_fail_before_ik(scene, field, value):
    scene.proposal[field] = value
    with pytest.raises(DualArmError):
        read_targets(scene.proposal, scene.config)
    assert not any(c.commands for c in scene.connections.values())


def test_two_arms_share_clock_and_preserve_span_during_lift(scene):
    _, steps = program(scene)
    for m in [s for s in steps if isinstance(s, Motion)]:
        assert len(m.joints["left"]) == len(m.joints["right"]) == len(m.times)
        dt = m.times[1] - m.times[0]
        for q in m.joints.values():
            velocity = np.diff(q, axis=0) / dt
            accel = (
                np.diff(
                    np.vstack([np.zeros_like(q[0]), velocity, np.zeros_like(q[0])]),
                    axis=0,
                )
                / dt
            )
            assert (
                np.max(np.abs(velocity))
                <= scene.config.limits["joint_speed_deg_s"] * 1.001
            )
            assert (
                np.max(np.abs(accel))
                <= scene.config.limits["joint_accel_deg_s2"] * 1.001
            )
        for poses in m.poses.values():
            velocity = np.diff(poses[:, :3], axis=0) / dt
            acceleration = (
                np.diff(np.vstack([np.zeros(3), velocity, np.zeros(3)]), axis=0) / dt
            )
            assert (
                np.linalg.norm(velocity, axis=1).max()
                <= scene.config.limits["cartesian_speed_mm_s"] * 1.001
            )
            assert (
                np.linalg.norm(acceleration, axis=1).max()
                <= scene.config.limits["cartesian_accel_mm_s2"] * 1.001
            )
        if m.phase.name in {"trial_lift", "lift"}:
            span = np.linalg.norm(
                m.poses["right"][:, :3] - m.poses["left"][:, :3], axis=1
            )
            assert np.ptp(span) < 1e-6
    assert not any(c.commands for c in scene.connections.values())


def test_whole_program_preflight_catches_inter_arm_crossing(scene):
    initial = {k: a.home_joints for k, a in scene.config.arms.items()}
    targets = {k: scene.models[k].forward(initial[k]).copy() for k in initial}
    targets["left"][1] = 120
    targets["right"][1] = -120
    with pytest.raises(DualArmError, match="inter-arm"):
        compile_motion(
            Phase("cross", "move", targets),
            initial,
            scene.config,
            scene.models,
            {k: c.inverse for k, c in scene.connections.items()},
        )
    assert not any(c.commands for c in scene.connections.values())


def test_joint_jump_and_table_obstacle_rejected(scene):
    initial = {k: a.home_joints for k, a in scene.config.arms.items()}
    targets = {
        k: scene.models[k].forward(q) + [0, 0, -20, 0, 0, 0] for k, q in initial.items()
    }
    inverse = {k: c.inverse for k, c in scene.connections.items()}
    inverse["right"] = lambda p, seed: np.asarray(seed) + 20
    with pytest.raises(DualArmError, match="discontinuity"):
        compile_motion(
            Phase("move", "move", targets), initial, scene.config, scene.models, inverse
        )
    scene.config.obstacles.append(
        {"name": "table", "min_mm": [200, -150, 155], "max_mm": [300, -90, 165]}
    )
    with pytest.raises(DualArmError, match="obstacle"):
        compile_motion(
            Phase("move", "move", targets),
            initial,
            scene.config,
            scene.models,
            {k: c.inverse for k, c in scene.connections.items()},
        )


def test_paired_full_cycle_returns_both_without_hardware(scene):
    initial, steps = program(scene)
    c = coordinator(scene)
    try:
        result = c.execute(steps, initial, scene.obs, lambda phase: GOOD_HOLD)
    finally:
        c.close()
    assert result["status"] == "COMPLETED" and not result["physical_execution"]
    assert result["completed_phases"][-1] == "return_start"
    for k, arm in scene.connections.items():
        assert arm.joints == pytest.approx(initial[k])
        assert [v for n, v in arm.commands if n == "gripper"] == [
            "open",
            "close",
            "open",
        ]
    events = [
        json.loads(l)
        for l in (scene.directory / "run/events.jsonl").read_text().splitlines()
    ]
    assert all(
        set(e["state"]) == {"left", "right"}
        for e in events
        if e["event"] == "servo_tick"
    )


@pytest.mark.parametrize(
    "fault",
    ["empty_grasp", "motion_error", "gripper_failure", "feedback_drift", "interrupt"],
)
def test_failure_stops_both_without_release_or_home(scene, fault):
    initial, steps = program(scene)
    c = coordinator(scene)

    def check(phase):
        if phase.name == "grasp_check":
            if fault == "empty_grasp":
                return {**GOOD_HOLD, "right_holding": False}
            if fault == "interrupt":
                raise KeyboardInterrupt()
            if fault == "motion_error":

                def fail(q):
                    raise RuntimeError("right servo fault")

                scene.connections["right"].servo = fail
            if fault == "feedback_drift":
                scene.connections["right"].joints[0] += 20
        return GOOD_HOLD

    if fault == "gripper_failure":
        original = scene.connections["right"].gripper

        def gripper(target):
            if target == "close":
                raise RuntimeError("jaw fault")
            return original(target)

        scene.connections["right"].gripper = gripper
    try:
        result = c.execute(steps, initial, scene.obs, check)
    finally:
        c.close()
    assert result["status"] in {"FAILED", "INTERRUPTED"}
    assert all(a.stopped for a in scene.connections.values())
    assert "release" not in result["completed_phases"]
    assert "return_start" not in result["completed_phases"]
    if fault == "empty_grasp":
        assert "trial_lift" not in result["completed_phases"]
    for arm in scene.connections.values():
        assert len([v for n, v in arm.commands if n == "gripper" and v == "open"]) == 1


def test_vision_timeout_stops_before_lift(scene):
    initial, steps = program(scene)
    scene.config.limits["max_vision_wait_s"] = 0.02
    done = threading.Event()
    c = coordinator(scene)
    try:
        result = c.execute(
            steps, initial, scene.obs, lambda phase: done.wait(0.2) or GOOD_HOLD
        )
    finally:
        done.set()
        c.close()
    assert result["status"] == "FAILED"
    assert "vision deadline" in result["error"]
    assert "trial_lift" not in result["completed_phases"]


def test_copied_gripper_wait_supports_cancel_and_timeout(scene):
    connection = object.__new__(XArmConnection)
    connection.config = scene.config.arms["left"]
    connection.cancel = scene.cancel
    connection.arm = SimpleNamespace(connected=True)
    scene.cancel.set()
    with pytest.raises(DualArmError, match="cancelled"):
        connection.check_gripper_wait(time.monotonic())
    scene.cancel.clear()
    with pytest.raises(DualArmError, match="deadline"):
        connection.check_gripper_wait(time.monotonic() - 30)
    connection.lock = threading.RLock()
    scene.cancel.set()
    with pytest.raises(DualArmError, match="cancelled") as error:
        connection.gripper("close")
    assert error.value.gripper_completion["status"] == "FAILED"


def test_calibration_fits_rotation_translation_and_checks_holdout():
    points = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
    frame = pose_matrix([200, 300, 50, 10, 20, 30])
    target = points @ frame[:3, :3].T + frame[:3, 3]
    hold = np.array([[20, 30, 40], [-50, 30, 100]])
    data = {
        "base_points_mm": points.tolist(),
        "world_points_mm": target.tolist(),
        "validation": {
            "base_points_mm": hold.tolist(),
            "world_points_mm": (hold @ frame[:3, :3].T + frame[:3, 3]).tolist(),
        },
    }
    result = fit_base(data)
    assert np.asarray(result["world_from_base_mm"]) == pytest.approx(frame)
    data["validation"]["world_points_mm"][0][0] += 10
    with pytest.raises(DualArmError, match="tolerance"):
        fit_base(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("left_holding", False),
        ("slip", True),
        ("overstretched", True),
        ("confidence", 0.5),
    ],
)
def test_visual_hold_gate_requires_both_and_confidence(field, value):
    with pytest.raises(DualArmError):
        require_hold({**GOOD_HOLD, field: value})


def test_model_planner_uses_original_pixels_and_both_targets(scene, monkeypatch):
    planner = VisionPlanner()
    calls = []

    def invoke(prompt, images, schema, directory):
        calls.append((prompt, images, schema))
        if directory.name == "center":
            return {
                "observation_id": scene.proposal["observation_id"],
                "center": scene.proposal["center"],
            }
        return scene.proposal

    monkeypatch.setattr(planner, "invoke", invoke)
    assert (
        planner.plan(scene.obs, scene.config, scene.directory / "model")
        == scene.proposal
    )
    assert "ORIGINAL" in calls[0][0] and "two" in calls[1][0]
    assert calls[0][1] == [scene.obs.image]
    assert len(calls) == 2 and calls[1][1][0] == scene.obs.image


def test_local_model_adapter_calls_existing_backend_contract(scene, monkeypatch):
    import cloth_agent.dual_arm.model as module
    import cloth_agent.planner_backend as backend

    monkeypatch.setattr(module.shutil, "which", lambda name: "/mock/claude")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = scene.proposal
        if kwargs["cwd"].name == "center":
            output = {
                "observation_id": scene.proposal["observation_id"],
                "center": scene.proposal["center"],
            }
        return SimpleNamespace(
            stdout=json.dumps({"structured_output": output}),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(
        backend,
        "tracked_call",
        lambda fn, *a, **kw: run(
            *a, **{k: v for k, v in kw.items() if not k.startswith("usage_")}
        ),
    )
    result = VisionPlanner("local").plan(
        scene.obs, scene.config, scene.directory / "local_model"
    )
    assert result == scene.proposal
    assert "--json-schema" in calls[0][0]
    assert calls[0][1]["cwd"] == scene.directory / "local_model/center"
    assert (scene.directory / "local_model/center/image_0.png").is_file()


def test_remote_model_adapter_preserves_original_image_contract(scene, monkeypatch):
    import cloth_agent.dual_arm.model as module

    captured = {}

    class Backend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def invoke(self, **kwargs):
            captured.update(kwargs)
            output = scene.proposal
            if "First select ONE" in kwargs["prompt"]:
                output = {
                    "observation_id": scene.proposal["observation_id"],
                    "center": scene.proposal["center"],
                }
            return SimpleNamespace(stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(module, "RemoteClaudeBackend", Backend)
    assert (
        VisionPlanner("remote").plan(
            scene.obs, scene.config, scene.directory / "remote_model"
        )
        == scene.proposal
    )
    assert captured["image_tools"] is False
    assert captured["overall_timeout_s"] == 60


@pytest.mark.parametrize("arm_id", ["left", "right"])
def test_real_adapter_axis_padding_identity_and_no_auto_enable(scene, arm_id):
    cfg = scene.config.arms[arm_id]

    class SDK:
        connected = True
        axis = cfg.axis
        control_box_sn = cfg.serial
        mode = 0
        tcp_offset = cfg.tcp_offset.tolist()
        motor_enable_states = [1] * cfg.axis
        motor_brake_states = [1] * cfg.axis

        def __init__(self):
            self.commands = []

        def get_robot_sn(self):
            return 0, "robot"

        def register_report_callback(self, callback):
            # Cache is available only after the full report initialization.
            callback({})
            self.tcp_offset = cfg.tcp_offset.tolist()
            callback({})
            return True

        def release_report_callback(self, callback):
            return True

        def get_state(self):
            return 0, 0

        def get_err_warn_code(self):
            return 0, [0, 0]

        def get_servo_angle(self, **kw):
            return 0, [*cfg.home_joints, *([0] * (7 - cfg.axis))]

        def get_position(self, **kw):
            return 0, cfg.home_joints[:6].tolist()

        def get_gripper_position(self):
            return 0, 850

        def get_gripper_status(self):
            return 0, 0

        def get_gripper_err_code(self):
            return 0, 0

        def get_inverse_kinematics(self, pose, **kw):
            return self.get_servo_angle()

        def is_joint_limit(self, joints, **kw):
            return 0, False

        def set_gripper_mode(self, n):
            self.commands.append(("gripper_mode", n))
            return 0

        def set_gripper_enable(self, n):
            self.commands.append(("gripper_enable", n))
            return 0

        def set_mode(self, n):
            self.commands.append(("mode", n))
            self.mode = n
            return 0

        def set_servo_angle_j(self, q, **kw):
            self.commands.append(("servo", q))
            return 0

        def set_state(self, n):
            self.commands.append(("state", n))
            return 0

        def disconnect(self):
            self.connected = False

    sdk = SDK()
    c = XArmConnection(cfg, scene.cancel, sdk_factory=lambda *a, **kw: sdk)
    assert len(c.snapshot()["joints"]) == cfg.axis
    assert len(c.inverse(cfg.home_joints[:6], cfg.home_joints)) == cfg.axis
    assert sdk.commands == []
    c.prepare()
    c.servo(cfg.home_joints)
    c.stop()
    assert ("state", 0) not in sdk.commands
    assert sdk.commands[-1] == ("state", 4)
    c.disconnect()


def test_contact_allowance_is_scoped_to_tool_and_contact_phases():
    a = Capsule("tool", np.array([0, 0, 5]), np.array([0, 0, 5]), 5)
    b = Capsule("tool", np.array([100, 0, 50]), np.array([100, 0, 50]), 5)
    box = {
        "name": "table",
        "min_mm": [-10, -10, -10],
        "max_mm": [10, 10, 0],
        "tool_contact_allowance_mm": 1,
    }
    with pytest.raises(DualArmError, match="obstacle"):
        check_collision({"left": [a], "right": [b]}, [box], 10)
    check_collision({"left": [a], "right": [b]}, [box], 10, allow_tool_contact=True)
    a.name = "link6"
    with pytest.raises(DualArmError, match="obstacle"):
        check_collision({"left": [a], "right": [b]}, [box], 10, allow_tool_contact=True)


def test_deadline_miss_does_not_send_catchup_jumps(scene, monkeypatch):
    initial, steps = program(scene)
    c = coordinator(scene)
    c.realtime = True

    # Oversleep the first scheduled servo tick; never dispatch a late command.
    def oversleep(seconds):
        time.sleep(seconds + 0.02)
        return False

    scene.config.limits["max_tick_lateness_s"] = 0.001
    monkeypatch.setattr(scene.cancel, "wait", oversleep)
    try:
        result = c.execute(steps, initial, scene.obs, lambda p: GOOD_HOLD)
    finally:
        c.close()
    assert result["status"] == "FAILED" and "deadline missed" in result["error"]
    assert not any(
        n == "servo" for a in scene.connections.values() for n, v in a.commands
    )


def test_one_gripper_failure_cancels_peer_wait(scene):
    initial, steps = program(scene)
    left_waiting, left_cancelled = threading.Event(), threading.Event()
    left_open, right_open = (
        scene.connections["left"].gripper,
        scene.connections["right"].gripper,
    )

    def left(target):
        if target == "open":
            return left_open(target)
        left_waiting.set()
        if scene.cancel.wait(1):
            left_cancelled.set()
            raise DualArmError("peer cancelled")
        pytest.fail("peer gripper wait was not cancelled")

    def right(target):
        if target == "open":
            return right_open(target)
        assert left_waiting.wait(1)
        raise DualArmError("right gripper failed")

    scene.connections["left"].gripper = left
    scene.connections["right"].gripper = right
    c = coordinator(scene)
    try:
        result = c.execute(steps, initial, scene.obs, lambda p: GOOD_HOLD)
    finally:
        c.close()
    assert left_cancelled.wait(1)
    assert result["status"] == "FAILED"
    assert all(arm.stopped for arm in scene.connections.values())


def test_capture_rejects_motion_during_window_even_if_arm_returns(scene, monkeypatch):
    import cloth_agent.perception as module
    from cloth_agent.dual_arm.observation import capture

    class Camera:
        def __init__(self, *args):
            self.intrinsics = np.eye(3)

        def start(self):
            pass

        def read(self):
            return np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4))

        def stop(self):
            pass

    monkeypatch.setattr(module, "RealSenseRGBD", Camera)
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        q = {k: arm.snapshot() for k, arm in scene.connections.items()}
        if calls == 3:
            q["left"]["joints"][0] += 1
        return q

    with pytest.raises(DualArmError, match="during RGB-D"):
        capture(
            scene.directory / "moving_capture", scene.config, scene.models, snapshot
        )
    assert not (scene.directory / "moving_capture").exists()


@pytest.mark.parametrize("arm_id,index", [("left", 0), ("right", 1)])
def test_actual_urdf_small_lift_ik_preserves_seed_branch(scene, arm_id, index):
    from cloth_agent.dual_arm.kinematics import ArmModel

    records = json.loads((ROOT / "data/robot/dual_arm_home.json").read_text())["arms"]
    q = np.asarray(records[index]["joints"])
    model = ArmModel(scene.config.arms[arm_id])
    target = model.forward(q)
    target[2] += 3
    solved = model.inverse(target, q)
    assert np.max(np.abs(solved - q)) < 3
    distance, angle = pose_error(model.forward(solved), target)
    assert distance < 0.5 and angle < 0.2


@pytest.mark.parametrize('arm_id,index', [('left',0),('right',1)])
def test_local_motion_bound_covers_independent_joint_rotations(scene,arm_id,index):
    from cloth_agent.dual_arm.kinematics import ArmModel
    records=json.loads((ROOT/'data/robot/dual_arm_home.json').read_text())['arms']
    q=np.asarray(records[index]['joints'])
    model=ArmModel(scene.config.arms[arm_id])
    half=np.full(len(q),10.)
    bounds=model.local_capsule_motion_bounds(q,half)
    nominal=model.capsules(q)
    rng=np.random.default_rng(2718)
    for _ in range(100):
        sample=np.clip(q+rng.uniform(-half,half),np.degrees(model.lower),np.degrees(model.upper))
        for base,actual in zip(nominal,model.capsules(sample)):
            displacement=max(np.linalg.norm(actual.start-base.start),np.linalg.norm(actual.end-base.end))
            assert displacement<=bounds[base.name]+1e-8


@pytest.mark.parametrize('arm_id,index', [('left',0),('right',1)])
def test_directional_projection_covers_independent_rotations(scene,arm_id,index):
    from cloth_agent.dual_arm.kinematics import ArmModel
    arm=scene.config.arms[arm_id]
    arm.capsules[-1]['start_mm']=[30.,20.,100.]
    arm.capsules[-1]['end_mm']=[-40.,50.,180.]
    q=np.asarray(json.loads((ROOT/'data/robot/dual_arm_home.json').read_text())['arms'][index]['joints'])
    model=ArmModel(arm);half=np.full(arm.axis,10.)
    cap=model.capsules(q)[-1];points=np.vstack([cap.start,cap.end])
    rng=np.random.default_rng(812)
    directions=rng.normal(size=(8,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    bounds=[model.projection_motion_interval(q,half,cap.name,points,n) for n in directions]
    for _ in range(100):
        sample=np.clip(q+rng.uniform(-half,half),np.degrees(model.lower),np.degrees(model.upper))
        actual=model.capsules(sample)[-1]
        for n,(lo,hi) in zip(directions,bounds):
            projection=np.vstack([actual.start,actual.end])@n
            assert projection.min()>=lo-1e-8
            assert projection.max()<=hi+1e-8
