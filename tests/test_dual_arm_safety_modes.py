"""Adversarial paths and mode contracts. Never connects to physical devices."""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.geometry import Capsule, DualArmError
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.model import VisionPlanner, require_hold
from cloth_agent.dual_arm.planning import Motion, Phase, ground_targets, read_targets
from cloth_agent.dual_arm.safety import check_state, padded_capsules, validate_sweep

from .test_dual_arm_runtime import (
    GOOD_HOLD,
    ROOT,
    coordinator,
    program,
)
from .test_dual_arm_runtime import (
    scene as scene,  # noqa: PLC0414 -- re-export shared pytest fixture
)


def joint_pair(scene, left_y, right_y):
    joints = {k: a.home_joints.copy() for k, a in scene.config.arms.items()}
    joints["left"][1] = left_y
    joints["right"][1] = right_y
    return joints


def test_continuous_check_rejects_crossing_between_clear_endpoints(scene):
    start, end = joint_pair(scene, -80, 80), joint_pair(scene, 80, -80)
    check_state(scene.config, scene.models, start)
    check_state(scene.config, scene.models, end)
    with pytest.raises(DualArmError, match="inter-arm"):
        validate_sweep(scene.config, scene.models, start, end)


def test_continuous_check_covers_independent_arm_progress(scene):
    # Equal progress preserves 40 mm separation at EVERY instant. If the right
    # arm lags/stops, the left arm crosses its initial position.
    start, end = joint_pair(scene, 0, 40), joint_pair(scene, 60, 100)
    for fraction in np.linspace(0, 1, 21):
        check_state(
            scene.config,
            scene.models,
            {k: start[k] * (1 - fraction) + end[k] * fraction for k in start},
        )
    with pytest.raises(DualArmError, match="inter-arm|cannot be certified"):
        validate_sweep(scene.config, scene.models, start, end)


def test_rotating_tool_arc_collides_even_when_endpoint_chord_is_clear(scene):
    class Arc:
        def capsules(self, q):
            angle = np.radians(q[0])
            p = np.array([100 * np.cos(angle), 100 * np.sin(angle), 0])
            return [Capsule("tool", p, p, 2)]

        def capsule_motion_bounds(self, delta):
            return {"tool": 100 * np.radians(abs(delta[0]))}

    class Still:
        def capsules(self, q):
            p = np.array([100, 0, 0])
            return [Capsule("tool", p, p, 2)]

        def capsule_motion_bounds(self, delta):
            return {"tool": 0}

    models = {"left": Arc(), "right": Still()}
    start, end = joint_pair(scene, 0, 0), joint_pair(scene, 0, 0)
    start["left"][0], end["left"][0] = -60, 60
    check_state(scene.config, models, start)
    check_state(scene.config, models, end)
    with pytest.raises(DualArmError, match="inter-arm"):
        validate_sweep(scene.config, models, start, end)


def test_uncertifiable_sweep_fails_closed_on_budget(scene):
    scene.config.raw["safety"]["max_sweep_nodes"] = 1
    with pytest.raises(DualArmError, match="budget"):
        validate_sweep(
            scene.config,
            scene.models,
            joint_pair(scene, 0, 40),
            joint_pair(scene, 60, 100),
        )


def test_measured_padding_includes_errors_tracking_latency_and_stopping(scene):
    safety = scene.config.raw["safety"]
    safety["status"] = "measured"
    for row in safety["arms"].values():
        row["base_error_mm"], row["geometry_error_mm"] = 2, 3
        row["stop_excursion_deg"] = [0.5] * len(row["stop_excursion_deg"])
    joints = joint_pair(scene, -120, 120)
    first = padded_capsules(scene.config, scene.models, joints)
    assert (
        first["left"][0].radius > 5 + 2 + 3 + scene.config.limits["tracking_error_mm"]
    )
    safety["controller_watchdog_s"] += 0.1
    second = padded_capsules(scene.config, scene.models, joints)
    assert second["left"][0].radius > first["left"][0].radius
    # A nominally clear 25 mm gap is insufficient once uncertainty is included.
    with pytest.raises(DualArmError, match="inter-arm"):
        check_state(scene.config, scene.models, joint_pair(scene, 0, 25))


def test_unmeasured_or_synthetic_bounds_cannot_enable_hardware(scene):
    with pytest.raises(DualArmError, match="base_error_mm|positive|real motion"):
        scene.config.require_real()
    raw = copy.deepcopy(scene.config.raw)
    del raw["safety"]
    with pytest.raises(DualArmError, match="no legacy fallback"):
        DualConfig.parse(raw, ROOT)
    raw = copy.deepcopy(scene.config.raw)
    raw["calibration_status"] = "measured"
    with pytest.raises(DualArmError, match="synthetic safety"):
        DualConfig.parse(raw, ROOT)


def test_moving_capsules_cannot_be_excluded_from_obstacle_checks(scene):
    raw = copy.deepcopy(scene.config.raw)
    raw["obstacles"] = [
        {
            "name": "table",
            "min_mm": [0, 0, 0],
            "max_mm": [1, 1, 1],
            "excluded_capsules": ["left/tool"],
        }
    ]
    with pytest.raises(DualArmError, match="fixed base"):
        DualConfig.parse(raw, ROOT)


def test_contact_exception_never_disables_inter_arm_collision(scene):
    with pytest.raises(DualArmError, match="inter-arm"):
        validate_sweep(
            scene.config,
            scene.models,
            joint_pair(scene, 0, 0),
            joint_pair(scene, 0, 0),
            tool_contact=True,
        )


def test_base_label_cannot_hide_a_moving_frame_from_obstacles(scene):
    raw = copy.deepcopy(scene.config.raw)
    # The Cartesian fixture intentionally attaches its named envelopes to the
    # flange; such an envelope is not eligible for a fixed mounting exemption.
    raw["obstacles"] = [
        {
            "name": "mount",
            "min_mm": [0, 0, 0],
            "max_mm": [1, 1, 1],
            "excluded_capsules": ["left/link_base"],
        }
    ]
    with pytest.raises(DualArmError, match="actually be fixed"):
        DualConfig.parse(raw, ROOT)


@pytest.mark.parametrize("arm_id,index", [("left", 0), ("right", 1)])
def test_urdf_displacement_bounds_cover_all_sampled_capsule_endpoints(
    scene, arm_id, index
):
    import json
    from dataclasses import replace

    from cloth_agent.dual_arm.setup import mesh_capsules

    arm = scene.config.arms[arm_id]
    model = ArmModel(
        replace(
            arm,
            capsules=mesh_capsules(
                arm.urdf, arm.axis, tool_radius_mm=45, tcp_offset=arm.tcp_offset
            ),
        )
    )
    q = np.asarray(
        json.loads((ROOT / "data/robot/dual_arm_home.json").read_text())["arms"][index][
            "joints"
        ]
    )
    delta = np.full(q.shape, 1.5)
    bounds = model.capsule_motion_bounds(delta)
    initial = model.capsules(q)
    for noise in np.random.default_rng(7).uniform(-delta, delta, (12, len(q))):
        for a, b in zip(initial, model.capsules(q + noise)):
            assert np.linalg.norm(a.start - b.start) <= bounds[a.name] + 1e-8
            assert np.linalg.norm(a.end - b.end) <= bounds[a.name] + 1e-8


def test_stale_feedback_rejects_before_any_motion_or_gripper_command(scene):
    initial, steps = program(scene)
    original = scene.connections["right"].snapshot

    def stale():
        row = original()
        row["sampled_monotonic_s"] = time.monotonic() - 10
        return row

    scene.connections["right"].snapshot = stale
    c = coordinator(scene)
    try:
        result = c.execute(steps, initial, scene.obs, lambda p: GOOD_HOLD)
    finally:
        c.close()
    assert result["status"] == "FAILED" and "stale" in result["error"]
    assert all(a.stopped and not a.commands for a in scene.connections.values())


@pytest.mark.parametrize("pin", ["left", "right"])
def test_pin_pull_keeps_pin_joints_fixed_and_only_moving_arm_grasps(scene, pin):
    scene.proposal.update(mode="pin_pull", pin_arm=pin, center=None)
    initial, steps = program(scene)
    moving = "right" if pin == "left" else "left"
    reference = None
    for item in steps:
        if isinstance(item, Motion) and item.phase.name == "pin_descend":
            reference = item.joints[pin][-1].copy()
        if isinstance(item, Motion) and item.phase.pin_arm:
            assert np.max(np.abs(item.joints[pin] - reference)) == 0
            if item.phase.name.startswith("pull_"):
                direction = item.poses[moving][-1, :2] - item.poses[pin][-1, :2]
                travel = item.poses[moving][-1, :2] - item.poses[moving][0, :2]
                assert direction @ travel > 0
    grasp = next(i for i in steps if isinstance(i, Phase) and i.name == "moving_grasp")
    assert grasp.gripper_arms == (moving,)
    c = coordinator(scene)
    hold = {
        **GOOD_HOLD,
        f"{pin}_holding": False,
        "pin_contact": True,
        "pin_slip": False,
    }
    try:
        result = c.execute(steps, initial, scene.obs, lambda p: hold)
    finally:
        c.close()
    assert result["status"] == "COMPLETED", result
    assert result["completed_phases"][-1] == "return_start"


def test_pinned_arm_drift_stops_both_before_pulling(scene):
    scene.proposal.update(mode="pin_pull", pin_arm="left", center=None)
    initial, steps = program(scene)
    c = coordinator(scene)

    def hold(phase):
        if phase.name == "pin_check":
            scene.connections["left"].joints[0] += 0.5
        return {**GOOD_HOLD, "pin_contact": True, "pin_slip": False}

    try:
        result = c.execute(steps, initial, scene.obs, hold)
    finally:
        c.close()
    assert result["status"] == "FAILED" and "pinned arm moved" in result["error"]
    assert "moving_grasp" not in result["completed_phases"]
    assert all(c.stopped for c in scene.connections.values())


def test_real_pin_mode_rejected_even_if_configuration_was_approved(scene, monkeypatch):
    scene.proposal.update(mode="pin_pull", pin_arm="left", center=None)
    initial, steps = program(scene)
    c = coordinator(scene)
    c.simulated = False
    monkeypatch.setattr(scene.config, "require_real", lambda **kw: None)
    try:
        result = c.execute(
            steps, initial, scene.obs, lambda p: GOOD_HOLD, confirmed=True
        )
    finally:
        c.close()
    assert result["status"] == "FAILED" and "force-limited" in result["error"]
    assert not any(a.commands for a in scene.connections.values())


@pytest.mark.parametrize("key", ["pin_contact", "moving_hold", "pin_slip"])
def test_pin_visual_gate_requires_pin_contact_and_active_grasp(key):
    phase = Phase(
        "pull_check", "observe", holding=True, pin_arm="left", checkpoint="pin_hold"
    )
    evidence = {
        **GOOD_HOLD,
        "left_holding": False,
        "pin_contact": True,
        "pin_slip": False,
    }
    require_hold(evidence, phase)
    evidence[
        {
            "pin_contact": "pin_contact",
            "moving_hold": "right_holding",
            "pin_slip": "pin_slip",
        }[key]
    ] = key == "pin_slip"
    with pytest.raises(DualArmError):
        require_hold(evidence, phase)


def test_center_selection_is_two_stage_and_cannot_be_changed(scene, monkeypatch):
    planner = VisionPlanner()
    calls = []

    def invoke(prompt, images, schema, directory):
        calls.append((prompt, images, schema))
        if len(calls) == 1:
            return {
                "observation_id": scene.proposal["observation_id"],
                "center": scene.proposal["center"],
            }
        result = copy.deepcopy(scene.proposal)
        result["center"]["pixel_xy"][0] += 1
        return result

    monkeypatch.setattr(planner, "invoke", invoke)
    with pytest.raises(DualArmError, match="changed the fixed center"):
        planner.plan(scene.obs, scene.config, scene.directory / "model")
    assert len(calls) == 2
    assert "FIXED" in calls[1][0] and len(calls[1][1]) == 2


def test_points_outside_center_neighborhood_or_same_side_are_rejected(scene):
    scene.config.limits["center_radius_mm"] = 100
    with pytest.raises(DualArmError, match="neighborhood"):
        ground_targets(scene.proposal, scene.obs, scene.config)
    scene.config.limits["center_radius_mm"] = 500
    scene.proposal["center"]["pixel_xy"] = [10, 32]
    with pytest.raises(DualArmError, match="opposite sides"):
        ground_targets(scene.proposal, scene.obs, scene.config)


@pytest.mark.parametrize("mode", ["center_pair", "pin_pull"])
def test_claude_can_abstain_without_inventing_targets(scene, monkeypatch, mode):
    planner = VisionPlanner()
    monkeypatch.setattr(
        planner, "invoke", lambda *a: {"decision": "abstain", "reason": "occluded"}
    )
    with pytest.raises(DualArmError, match="declined.*occluded"):
        planner.plan(scene.obs, scene.config, scene.directory / "model", mode=mode)


def test_legacy_proposal_cannot_bypass_mode_and_center_constraints(scene):
    scene.proposal["schema_version"] = 1
    with pytest.raises(DualArmError, match="schema_version=2"):
        read_targets(scene.proposal, scene.config)
