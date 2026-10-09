"""Offline real-FK/FCL/OMPL planner regression; never connect to hardware."""

import copy
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("fcl")

from cloth_agent.dual_arm.collision import CollisionScene
from cloth_agent.dual_arm.collision.calibration import template
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.motion import DualArmPlanner
from cloth_agent.dual_arm.motion.ik import IKOptions
from cloth_agent.dual_arm.motion.preflight import (
    plan_digest,
    preflight,
    validate_artifact,
)
from cloth_agent.dual_arm.motion.trajectory import sample_segment, validate_timed


@pytest.fixture
def planner():
    scene = CollisionScene(template(synthetic=True))
    return DualArmPlanner(
        scene, ik_options=IKOptions(starts=1, iterations=45, timeout_s=12)
    )


def test_continuous_grasp_ik_satisfies_world_points_limits_and_clearance(planner):
    scene = planner.scene
    qa, qb = scene.initial["left"], scene.initial["right"]
    frames = scene.tcp_world(qa, qb)
    result = planner.solve_grasp_pose(
        frames["left"][:3, 3] + [0, 0, 0.005],
        frames["right"][:3, 3] + [0, 0, 0.005],
        qa,
        qb,
    )
    assert result["success"], result
    assert max(result["position_error_m"]) <= 0.001
    assert max(result["approach_error_rad"]) <= np.deg2rad(15) + 1e-7
    assert planner.checker.check(result["q_a"], result["q_b"])["safe"]
    assert not result["execution_authorized"]


def test_continuous_yaw_respects_cloth_edge(planner):
    scene = planner.scene
    qa, qb = scene.initial["left"], scene.initial["right"]
    frames = scene.tcp_world(qa, qb)
    planner.ik.options = IKOptions(
        starts=1, iterations=80, timeout_s=18, edge_tolerance_rad=0.04
    )
    edge = [np.cos(0.37), np.sin(0.37), 0]
    result = planner.solve_grasp_pose(
        frames["left"][:3, 3], frames["right"][:3, 3], qa, qb, edge_a=edge, edge_b=edge
    )
    assert result["success"], result
    for key in ("grasp_pose_a", "grasp_pose_b"):
        axis = np.asarray(result[key])[:3, 0]
        assert abs(axis @ edge) >= np.cos(0.04) - 1e-7
    assert np.linalg.norm(np.asarray(result["q_a"]) - qa) > 0.05


def test_infeasible_grasp_returns_failure_not_unsafe_pose(planner):
    planner.ik.options = IKOptions(starts=1, iterations=8, timeout_s=2)
    result = planner.solve_grasp_pose(
        [10, 0, 10],
        [10, 0, 10],
        planner.scene.initial["left"],
        planner.scene.initial["right"],
    )
    assert not result["success"] and "q_a" not in result
    assert "budget" in result["reason"]


def test_interval_certificate_checks_interior_collision(planner):
    # Put a tiny obstacle on a camera's midpoint pose; endpoints remain clear.
    scene = planner.scene
    qa, qb = scene.initial["left"].copy(), scene.initial["right"].copy()
    center = scene.transforms(qa, qb)["cam_a"]
    raw = copy.deepcopy(scene.raw)
    raw["obstacles"].append(
        {
            "name": "midpoint_block",
            "size_m": [0.006] * 3,
            "world_from_box_m": center.tolist(),
        }
    )
    other = DualArmPlanner(CollisionScene(raw))
    a, b = np.r_[qa, qb], np.r_[qa, qb]
    a[5] -= 0.8
    b[5] += 0.8
    assert other.checker.check(a[:6], a[6:])["safe"]
    assert other.checker.check(b[:6], b[6:])["safe"]
    with pytest.raises(DualArmError, match="unsafe state"):
        other.validator.certify(a, b)


def test_real_ompl_rrtconnect_routes_around_midpoint_obstacle(planner):
    pytest.importorskip("ompl")
    from ompl import util as ou

    ou.RNG.setSeed(19)
    scene = planner.scene
    a = np.r_[scene.initial["left"], scene.initial["right"]]
    b = a.copy()
    raw = copy.deepcopy(scene.raw)
    center = scene.transforms(a[:6], a[6:])["cam_a"]
    raw["obstacles"].append(
        {
            "name": "rrt_block",
            "size_m": [0.006] * 3,
            "world_from_box_m": center.tolist(),
        }
    )
    a[5] -= 0.8
    b[5] += 0.8
    other = DualArmPlanner(CollisionScene(raw))
    path = other.paths.joint_path(a, b, timeout_s=30)
    assert len(path) > 2
    assert path[0] == pytest.approx(a)
    assert path[-1] == pytest.approx(b)
    for i in range(len(path) - 1):
        assert (
            other.validator.certify(path[i], path[i + 1])["minimum_clearance_bound_m"]
            > 0.01
        )


@pytest.mark.parametrize("order", ["left_first", "right_first"])
def test_sequential_transit_keeps_inactive_arm_fixed(planner, order):
    a = np.r_[planner.scene.initial["left"], planner.scene.initial["right"]]
    b = a.copy()
    b[5] += 0.02
    b[12] += 0.02
    path = planner.paths.joint_path(a, b, order=order)
    assert len(path) == 3
    active, inactive = (
        (slice(0, 6), slice(6, 13))
        if order == "left_first"
        else (slice(6, 13), slice(0, 6))
    )
    assert path[1][inactive] == pytest.approx(a[inactive])
    assert path[1][active] == pytest.approx(b[active])


@pytest.fixture(scope="module")
def planned():
    scene = CollisionScene(template(synthetic=True))
    planner = DualArmPlanner(
        scene, ik_options=IKOptions(starts=1, iterations=45, timeout_s=12)
    )
    a, b = scene.initial["left"], scene.initial["right"]
    frames = scene.tcp_world(a, b)
    plan = planner.plan(
        frames["left"][:3, 3],
        frames["right"][:3, 3],
        a,
        b,
        approach_m=0.01,
        lift_m=0.01,
        spread_m=0.01,
        return_after_release=True,
    )
    assert plan["success"], plan
    return planner, plan


def test_full_grasp_lift_spread_return_shared_clock_and_analytic_limits(planned):
    planner, plan = planned
    assert {s["phase"] for s in plan["segments"]} == {
        "transit",
        "descend",
        "lift",
        "spread",
        "return_after_release",
    }
    assert plan["trajectory_a"]["times_s"] == plan["trajectory_b"]["times_s"]
    assert plan["execution_order"]["grasp_lift_spread"] == "simultaneous"
    assert plan["minimum_clearance"] > planner.scene.clearance
    assert not plan["physical_execution_supported"]
    assert plan['velocity_limits_rad_s'] == pytest.approx(np.full(13, np.deg2rad(5)))
    assert plan['acceleration_limits_rad_s2'] == pytest.approx(np.full(13, np.deg2rad(10)))
    for segment in plan["segments"]:
        for u in (0, 0.5, (3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6, 1):
            _, v, a = sample_segment(segment, u * segment["duration_s"])
            assert np.all(np.abs(v) <= np.asarray(plan["velocity_limits_rad_s"]) + 1e-8)
            assert np.all(
                np.abs(a) <= np.asarray(plan["acceleration_limits_rad_s2"]) + 1e-8
            )
    for previous, following in zip(plan['segments'], plan['segments'][1:]):
        before = sample_segment(previous, previous['duration_s'])
        after = sample_segment(following, 0)
        assert np.allclose(before, after, atol=1e-8)
        if previous['phase'] != following['phase']:
            assert np.max(np.abs(before[1])) < 1e-8
    assert any(e["kind"] == "require_release_confirmation" for e in plan["events"])


def test_cartesian_tcp_at_interior_time_stays_within_certified_corridor(planned):
    planner, plan = planned
    for segment in plan["segments"]:
        if "cartesian" not in segment:
            continue
        q, _, _ = sample_segment(segment, 0.37 * segment["duration_s"])
        start = np.asarray(segment['start_q_rad'])
        delta = np.asarray(segment['end_q_rad']) - start
        progress = float((q-start) @ delta / (delta @ delta)) if delta @ delta > 1e-24 else 0.
        frames = planner.scene.tcp_world(q[:6], q[6:])
        for key in frames:
            first = np.array(segment["cartesian"]["start"][key])[:3, 3]
            last = np.array(segment["cartesian"]["end"][key])[:3, 3]
            desired = first * (1 - progress) + last * progress
            assert np.linalg.norm(frames[key][:3, 3] - desired) <= 0.001


def test_offline_preflight_passes_without_authorizing_motion(planned):
    planner, plan = planned
    now = datetime.fromisoformat(plan["created_at"]).timestamp() + 0.05
    result = preflight(
        plan,
        planner,
        plan["initial_q_a"],
        plan["initial_q_b"],
        sampled_at_a=now - 0.01,
        sampled_at_b=now,
        now=now,
    )
    assert result["success"] and not result["execution_authorized"]


@pytest.mark.parametrize(
    "fault", ["expired", "stale", "skew", "state_changed", "content_changed", "real"]
)
def test_preflight_rejects_bad_evidence_and_real_execution(planned, fault):
    planner, original = planned
    plan = copy.deepcopy(original)
    now = datetime.fromisoformat(plan["created_at"]).timestamp() + 0.05
    qa = plan["initial_q_a"].copy()
    ta = tb = now
    if fault == "expired":
        now += 121
        ta = tb = now
    elif fault == "stale":
        ta -= 0.2
    elif fault == "skew":
        ta -= 0.03
    elif fault == "state_changed":
        qa[0] += 0.01
    elif fault == "content_changed":
        plan["segments"][0]["end_q_rad"][0] += 0.01
    with pytest.raises(DualArmError):
        preflight(
            plan,
            planner,
            qa,
            plan["initial_q_b"],
            sampled_at_a=ta,
            sampled_at_b=tb,
            now=now,
            real=fault == "real",
        )


def test_retiming_and_sample_corruption_are_revalidated(planned):
    planner, plan = planned
    bad = copy.deepcopy(plan)
    bad["segments"][0]["duration_s"] /= 100
    with pytest.raises(DualArmError, match="velocity/acceleration"):
        validate_timed(bad, planner.validator)
    bad = copy.deepcopy(plan)
    bad["trajectory_a"]["positions_rad"][1][0] += 0.1
    bad["plan_digest"] = plan_digest(bad)
    with pytest.raises(DualArmError, match="differs from certified"):
        validate_artifact(bad, planner)


def test_changed_scene_rejects_old_plan(planned):
    planner, plan = planned
    raw = copy.deepcopy(planner.scene.raw)
    raw["geometry_uncertainty_m"]["left"] += 0.001
    changed = DualArmPlanner(CollisionScene(raw))
    with pytest.raises(DualArmError, match="scene changed"):
        validate_artifact(plan, changed)


def test_failed_plan_never_returns_partial_trajectory(planner):
    result = planner.plan(
        [0, 0, 0],
        [0, 0, 0],
        planner.scene.initial["left"],
        planner.scene.initial["right"],
        task="pin_pull",
    )
    assert not result["success"] and "trajectory_a" not in result
    assert "contact controller" in result["reason"]
