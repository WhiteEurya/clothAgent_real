import copy
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest

pytestmark = pytest.mark.usefixtures("fake_one_shot_view_hardware")

from cloth_agent import codex_recovery as recovery
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.observation import DepthUnavailableError
from cloth_agent.experiments import one_shot_goal_full_episode_v1 as m
from cloth_agent.unfolding_planners import ShadowPlannerRouter, PlannerDisabledError


@pytest.mark.parametrize("error,phase,dispatched,allowed", [
    (DepthUnavailableError("grasp has insufficient measured depth"), "HOST_CHECKS", False, True),
    (DualArmError("grasp depth patch crosses a discontinuity"), "HOST_CHECKS", False, True),
    (DualArmError("metric displacement 160 mm outside existing 10–150 mm range"), "HOST_CHECKS", False, True),
    (DualArmError("left: target outside measured workspace"), "HOST_CHECKS", False, True),
    (ValueError("invalid JSON"), "MODEL", False, True),
    (RuntimeError("HTTP 503; no retry"), "MODEL", False, True),
    (RuntimeError("HTTP 401; no retry"), "MODEL", False, False),
    (DepthUnavailableError("missing"), "EXECUTION", True, False),
    (DepthUnavailableError("missing"), "AFTER_OBSERVATION", False, False),
    (DualArmError("left: controller fault/state 4"), "HOST_CHECKS", False, False),
    (DualArmError("robot moved from expected stationary state"), "HOST_CHECKS", False, False),
    (DualArmError("inter-arm collision"), "HOST_CHECKS", False, False),
    (DualArmError("intrinsics/cloud inconsistency"), "HOST_CHECKS", False, False),
    (DualArmError("unknown failure"), "HOST_CHECKS", False, False),
    (KeyboardInterrupt(), "MODEL", False, False),
    (ValueError("Codex response bad"), "DELIBERATIVE_MODEL", False, False),
])
def test_recovery_allowlist(error, phase, dispatched, allowed):
    assert bool(recovery.recovery_category(error, phase, dispatched)) == allowed


def test_controller_stop_is_never_cleared(tmp_path):
    calls = []
    conn = SimpleNamespace(lock=threading.RLock(),
        arm=SimpleNamespace(get_state=lambda: (0, 4), get_err_warn_code=lambda: (0, [0, 0])),
        checked_value=lambda value, name: value[1])
    runtime = SimpleNamespace(cancel=threading.Event(),
        coordinator=SimpleNamespace(connections={"left": conn}),
        arrival=lambda q: calls.append("arrival"))
    with pytest.raises(DualArmError, match="controller fault/state"):
        recovery.check_stationary(runtime, {"left": [0]*6}, tmp_path/"preflight.json")
    assert calls == []
    assert m.read(tmp_path/"preflight.json")["clears_faults"] is False


def action(offset=0):
    return dict(decision="ACTION", grasp_px=[20+offset, 30], target_px=[70+offset, 40],
                motion_type="lift_pull_place", short_reason="展开可见衣边。")


def test_shadow_gate_cannot_enable_recovery(tmp_path):
    router = ShadowPlannerRouter(tmp_path)
    router.accept_fast(action())
    with pytest.raises(PlannerDisabledError):
        router.accept_deliberative(action(12))
    assert router.execution_plan == action()
    enabled = ShadowPlannerRouter(tmp_path, recovery_enabled=True)
    replacement = action(12)
    enabled.accept_deliberative(replacement)
    replacement["grasp_px"][0] = 99
    assert enabled.execution_plan == action(12)


@pytest.mark.parametrize("mode,expected_calls,expected_actions,termination", [
    ("depth", 1, 1, "DONE"), ("ik", 1, 1, "DONE"), ("model", 1, 1, "DONE"),
    ("persistent", 2, 0, "SAFETY_STOP"), ("repeat", 2, 0, "SAFETY_STOP"),
    ("hardware", 0, 0, "SAFETY_STOP"), ("execution", 0, 1, "SAFETY_STOP"),
    ("interrupt", 1, 0, "USER_STOP"), ("total_cap", 1, 0, "SAFETY_STOP"),
])
def test_recovery_loop_is_fresh_bounded_and_checked(tmp_path, monkeypatch, mode,
                                                   expected_calls, expected_actions, termination):
    calls = []
    snapshots = []
    class Runtime:
        def __init__(self, *args):
            self.cancel = threading.Event()
        def capture_current(self, path):
            path.mkdir()
            Image.new("RGB", (110, 80), "purple").save(path/"rgb.png")
            observation = SimpleNamespace(image=path/"rgb.png", directory=path,
                                           meta={"joints_deg": {"left": [0]*6}})
            snapshots.append(observation)
            return observation
        def arrival(self, *args): pass
        def move_to_perception(self, *args): pass
        def _execute_operations(self, ops, folder):
            calls.append("execute")
            if mode == "execution":
                raise DualArmError("controller disconnected during execution")
            assert ops == action(12)
            return {"status": "EXECUTED"}
        def stop(self):
            calls.append("stop")
            return {}
        def close(self): pass
    class Recorder:
        def __init__(self, *a): pass
        def start(self): pass
        def check(self): pass
        def stage(self, *a): pass
        def close(self): pass
    def initialize(runtime, folder, *args):
        folder.mkdir()
        m.save(folder/"result.json", dict(physical_actions=1, robot_execution_s=1))
    def fast(root, folder, before, *args):
        calls.append("fast")
        m.save(folder/"attempt.json", {})
        m.save(folder/"model_metrics.json", {"wall_latency_s": 20.})
        if mode == "model" and calls.count("fast") == 1:
            raise ValueError("invalid model response")
        plan = action() if calls.count("fast") == 1 else dict(
            decision="DONE", grasp_px=None, target_px=None, motion_type=None, short_reason="已展开。")
        m.save(folder/"plan.json", plan)
        m.planner.overlay(before.image, plan, folder/"proposed_action_overlay.png", "fast")
        return plan, {"wall_latency_s": 20.}
    def slow(self, root, folder, before, request, timeout, **kwargs):
        calls.append("codex")
        assert before is snapshots[-1] and before.directory != snapshots[0].directory
        assert request["failures"][0]["physical_execution"] is False
        if mode == "interrupt":
            raise KeyboardInterrupt("stop during recovery")
        plan = action(0 if mode == "repeat" else 12*request["attempt"])
        m.save(folder/"plan.json", plan)
        m.save(folder/"deliberative_plan.json", plan)
        m.save(folder/"deliberative_metrics.json", {"wall_latency_s": 30.})
        m.planner.overlay(before.image, plan, folder/"proposed_action_overlay.png", "codex")
        return plan, {"wall_latency_s": 30.}
    def preflight(*args):
        calls.append("preflight")
    def host(runtime, before, plan, folder):
        calls.append("host")
        if mode == "hardware":
            raise DualArmError("left: controller fault/state 4")
        if mode == "ik" and calls.count("host") == 1:
            error = DualArmError("left: PyRoki IK rejected target")
            error.ik_failure = {"position_residual_mm": 1.2}
            raise error
        if (mode in ("persistent", "total_cap") or
            (calls.count("host") == 1 and mode not in ("model", "execution"))):
            raise DepthUnavailableError("grasp has insufficient measured depth")
        return copy.deepcopy(plan), {}
    def gate(*args):
        calls.append("gate")
        return {"decision": "DELIBERATE"}, {"wall_latency_s": 2.}
    for name, value in (("EpisodeCollectionRuntime", Runtime), ("TrajectoryRecorder", Recorder),
                        ("initialize", initialize), ("plan_round", fast),
                        ("prepare_action", host), ("direct_once", gate)):
        monkeypatch.setattr(m, name, value)
    monkeypatch.setattr(m.planner, "auth_headers", lambda: {})
    monkeypatch.setattr(recovery.CodexRecoveryPlanner, "plan", slow)
    monkeypatch.setattr(recovery, "check_stationary", preflight)
    goal = tmp_path/"goal.png"; Image.new("RGB", (110, 80), "purple").save(goal)
    args = SimpleNamespace(output=tmp_path/"episode", goal_image=goal, config=Path("unused"),
        max_rounds=6, timeout_s=1, codex_recovery=True, gate_timeout_s=1,
        max_codex_replans=1 if mode == "total_cap" else 4, max_replans_per_incident=2)
    m.run(args)
    summary = m.read(args.output/"episode_summary.json")
    assert summary["termination_reason"] == termination
    assert summary["codex_recovery_calls"] == calls.count("codex") == expected_calls
    assert summary["physical_actions"] == calls.count("execute") == expected_actions
    assert calls.count("gate") == int(mode != "model")
    assert not summary.get("report_error")
    if expected_calls:
        original = m.read(args.output/"round_000/execution.json")
        assert original["physical_execution"] is False and original["recovery_scheduled"]
        assert (args.output/"round_000/after/unavailable.json").exists()
        assert (args.output/"round_001/fast_plan.json").exists() is False
    if termination == "DONE":
        assert calls.count("fast") == 2
        assert summary["planner_layers"]["fast_planner_latency_s"]["n"] == (1 if mode == "model" else 2)
        assert summary["total_deliberative_planning_s"] == 30


@pytest.mark.parametrize('multiview', [False, True])
@pytest.mark.parametrize('rotation', [False, True])
def test_actual_codex_wrapper_inputs_and_output_no_memory(tmp_path, monkeypatch, multiview, rotation):
    from cloth_agent.experiments import trajectory_memory_full_episode_v1 as shared
    root = tmp_path/"episode"; folder = root/"round_001"
    (root/"goal").mkdir(parents=True); (folder/"before").mkdir(parents=True)
    Image.new("RGB", (110, 80), "purple").save(root/"goal/target.png")
    Image.new("RGB", (110, 80), "purple").save(folder/"before/rgb.png")
    if multiview:
        (folder/'right_view').mkdir()
        Image.new('RGB', (110, 80), 'blue').save(folder/'right_view/rgb.png')
    before = SimpleNamespace(directory=folder/"before", image=folder/"before/rgb.png")
    def agent(inputs, target, observed, history, manifest, timeout, **kwargs):
        assert observed is before and history == [] and manifest["records"] == []
        assert kwargs["memory_enabled"] is False
        assert "HOST_REJECTION_DATA" in kwargs["prompt"] and "missing measured depth" in kwargs["prompt"]
        assert recovery.wire.GRIPPER_BOUNDARY_PROMPT in kwargs["prompt"]
        assert kwargs["goal_directory"] == folder/"recovery_inputs/goal"
        if multiview:
            assert kwargs['right_view_image'] == folder/'right_view/rgb.png'
            assert 'Pa/Pb 只能用第一张 perception' in kwargs['prompt']
        else:
            assert 'right_view_image' not in kwargs
        result = dict(decision="ACTION", grasp_px=[32, 30], target_px=[82, 40],
                    motion_type="lift_pull_place", supporting_experiences=[],
                    short_rationale="改抓有深度的可见衣边。")
        if rotation:
            result['grasp_angle_deg'] = 65.
            assert 'grasp_angle_deg' in kwargs['plan_schema']['required']
            assert '夹住后保持该方向' in kwargs['prompt']
            kwargs['plan_validator'](result, [110, 80], [])
            with pytest.raises(recovery.wire.jsonschema.ValidationError):
                kwargs['plan_validator'](dict(result, grasp_angle_deg=190), [110, 80], [])
        return result, dict(
            historical_source_images_read=0, goal_image_file_read=True)
    monkeypatch.setattr(shared, "agent_round", agent)
    plan, metrics = recovery.CodexRecoveryPlanner().plan(
        root, folder, before, {"failures": [{"error": "missing measured depth"}]}, 600,
        allow_grasp_rotation=rotation)
    assert plan["grasp_px"] == [32, 30] and metrics["status"] == "VALID"
    assert (folder/"deliberative_agent_plan.json").exists()
    assert not (root/"memory").exists()
    assert plan.get('grasp_angle_deg') == (65. if rotation else None)
