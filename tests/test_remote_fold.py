from __future__ import annotations

import json
import subprocess
import shutil
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent.config import ExperimentConfig, RobotConfig, SafetyError, WorkspaceBounds
from cloth_agent.fold_exploration_pipeline import (
    FoldExplorationPipeline, FoldSupervisor, build_parser, _validate_model_acquisition_probe,
)
from cloth_agent.free_exploration import ExplorationPlanningError
from cloth_agent.planner_backend import BackendResult, PlannerBackendError, RemoteClaudeBackend, parse_claude_json
from cloth_agent.remote_fold import RemoteFoldClient, compile_pixel_motion
from cloth_agent.auto_exploration import validate_visual_plan_payload
from cloth_agent.garment_grounding_mcp import GarmentGrounding
from cloth_agent.workspace_debug import WorkspaceTargetError, lateral_clearance, save_workspace_debug
from cloth_agent.auto_exploration import ClaudeAutoClient, SelectedReferenceNotExecutableError, ReferenceReselectionExhaustedError


def visual_payload():
    return {"garment_observation": "White shirt", "opening_strategy": "Lift and reverse the selected edge",
        "confidence": 0.8, "selected_reference": {"camera": "A", "reference_id": "R001", "reason": "Visible cloth"},
        "motion_intent": "A reversible acquisition probe", "expected_observation": "Cloth lifts with gripper",
        "safety_notes": ["Host must validate all motion"]}


def motion_payload():
    def move(height):
        return {"name": "move", "args": {"target": "grasp", "pixel_xy": None,
            "height_above_grasp_mm": height, "yaw_deg": 0}}
    return {"requires_lift_checkpoint": True, "actions": [
        move(60), {"name": "open_gripper", "args": {}}, move(0),
        {"name": "close_gripper", "args": {}}, move(30), move(0),
        {"name": "open_gripper", "args": {}}, move(60), {"name": "home", "args": {}}]}


def supervisor_payload():
    return {"status": "READY", "current_step": "left_sleeve", "completed_steps": [],
        "garment_visibility": "FULL", "trajectory_decision": "CONTINUE", "confidence": 0.8,
        "evidence": ["Sleeves are extended"], "reason": "Left sleeve is next"}


def evaluation_payload():
    def stage(status):
        return {"status": status, "confidence": 0.5, "evidence": ["Visible RGB evidence is inconclusive"]}
    return {"target_selection": stage("UNKNOWN"), "grasp_acquisition": stage("UNKNOWN"),
        "target_structure_acquired": stage("UNKNOWN"), "transport": stage("UNKNOWN"),
        "laydown": stage("NOT_REACHED"), "task_progress": {"status": "NEUTRAL", "confidence": 0.5,
            "metrics": {"visible_area_delta": "UNKNOWN", "overlap_delta": "UNKNOWN",
                "relief_delta": "UNKNOWN", "boundary_change": "No conclusive change"}},
        "earliest_failure_stage": "UNKNOWN", "next_experiment": {
            "keep": ["target"], "change": ["view angle"], "reason": "Need acquisition evidence"}}


def test_remote_evaluator_receives_skill_body(saved_scene):
    session, images, _ = saved_scene
    backend = FakeBackend(evaluation_payload())
    client = RemoteFoldClient(backend=backend, binary="not-installed")
    client.evaluate(images, images, proposal=SimpleNamespace(reveal_strategy="fold", expected_observation="folded sleeve"),
                    run_dir=session.run_dir, skill_guidance="Provisional detector: occlusion is UNKNOWN, never EMPTY.")
    assert "Provisional detector: occlusion is UNKNOWN, never EMPTY." in backend.calls[0]["prompt"]


class FakeBackend:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return BackendResult(json.dumps({"result": "```json\n" + json.dumps(result) + "\n```"}), "", 0,
                             ("ssh", "company-planner", "remote job"))


@pytest.fixture
def saved_scene(tmp_path):
    views = tmp_path / "workspace" / "perception_views"
    views.mkdir(parents=True)
    raw = Image.new("RGB", (40, 30), (220, 210, 200))
    raw.save(views / "camera_0_A.png")
    images = []
    for name in ("camera_A_rgb_upright.png", "camera_A_rxxx_overlay_upright.png", "camera_A_height_map.png"):
        path = tmp_path / name
        raw.rotate(-90, expand=True).save(path)
        images.append(path)
    xyz = np.zeros((30, 40, 3), dtype=float)
    xyz[:] = [500, 40, 30]
    xyz[20, 10] = [550, 80, 30]
    np.save(views / "camera_A_base_xyz_mm.npy", xyz)
    np.save(views / "camera_A_height_above_table_mm.npy", np.full((30, 40), 25.0))
    np.save(views / "camera_A_table_z_mm.npy", np.full((30, 40), 5.0))
    np.save(views / "camera_A_garment_mask.npy", np.ones((30, 40), dtype=bool))
    (views / "camera_A_coordinate_guide.json").write_text(json.dumps({"samples": [
        {"reference_id": "R001", "pixel_xy": [15, 15], "base_xyz_mm": [500, 40, 30],
         "height_above_table_mm": 25}]}))
    robot = RobotConfig(robot_ip="127.0.0.1", boundaries=WorkspaceBounds(
        x_min=350, x_max=800, y_min=-300, y_max=170, z_min=6, z_max=500),
        init_joints_deg=(0,) * 6, init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180, orientation_pitch_deg=0)
    session = SimpleNamespace(run_dir=tmp_path, workspace=tmp_path / "workspace", project_root=tmp_path,
        robot_config=robot, experiment_config=ExperimentConfig(500, 40, 30))
    return session, images, GarmentGrounding(views)


def test_real_plan_path_remote_only_with_local_grounding(saved_scene, monkeypatch):
    session, images, grounding = saved_scene
    backend = FakeBackend(visual_payload(), motion_payload())
    client = RemoteFoldClient(backend=backend, binary="definitely-not-installed")
    client.skill_guidance = "Learned detector: do not confuse camera occlusion with an empty grasp."
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("local Claude/robot process invoked"))
    proposal = client.plan(images, session, "Probe the garment edge.\nEXECUTION CONTRACT — ACQUISITION PROBE",
        history=[{"robot_state": {"secret": "private_robot"}, "base_xyz_mm": [999, 999, 999],
                  "evaluation": {"grasp_acquisition": {"status": "UNKNOWN"}}}])
    assert len(backend.calls) == 2
    assert client.skill_guidance in backend.calls[0]["prompt"]
    assert proposal.actions[2]["args"] == {"x": 500, "y": 40, "z": 27, "yaw": 0}
    _validate_model_acquisition_probe(proposal)
    assert client.last_reference_validation["reference_id"] == "R001"
    assert client.last_grounding_verification["authority"] == "local_pixel_compiler"
    for call in backend.calls:
        assert images[2] not in call["image_paths"]
        assert str(session.run_dir) not in call["prompt"]
        assert "private_robot" not in call["prompt"]
        assert "base_xyz_mm" not in call["prompt"]
        assert "999" not in call["prompt"]
    client.backend = FakeBackend(PlannerBackendError("SSH failed"))
    with pytest.raises(PlannerBackendError):
        client.plan(images, session, "Probe the garment edge")
    assert client.last_plan_result is None


def test_upright_pixel_transform_and_repair_failure(saved_scene):
    session, _, grounding = saved_scene
    motion = motion_payload()
    motion["actions"][5]["args"] = {"target": "pixel", "pixel_xy": [9, 10],
        "height_above_grasp_mm": 30, "yaw_deg": 0}
    proposal, _ = compile_pixel_motion(motion, validate_visual_plan_payload(visual_payload()), grounding,
                                      session.robot_config, (30, 40))
    assert proposal.actions[5]["args"] == {"x": 550, "y": 80, "z": 57, "yaw": 0}


def test_remote_stages_share_current_garment_frame(saved_scene):
    from cloth_agent.fold_frame import build_frame, FRAME_RULE
    session, images, _ = saved_scene
    with Image.open(images[0]) as image:
        frame = build_frame(image, [5, 20], [25, 20])
    views = session.workspace / 'perception_views'
    (views / 'garment_frame.json').write_text(json.dumps(frame))
    backend = FakeBackend(visual_payload(), motion_payload())
    client = RemoteFoldClient(backend=backend)
    client.plan(images, session, 'Probe garment. ' + FRAME_RULE)
    assert len(backend.calls) == 2
    for call in backend.calls:
        assert json.dumps(frame) in call['prompt']
        assert 'left/right refer to that displayed image' not in call['prompt']
        assert call['image_paths'] == images[:2]
    frame['image_sha256'] = 'stale'
    (views / 'garment_frame.json').write_text(json.dumps(frame))
    backend.responses.extend([visual_payload(), motion_payload()])
    client.plan(images, session, 'Probe garment. ' + FRAME_RULE)
    assert len(backend.calls) == 4
    assert '"molmo_frame_hint": null' in backend.calls[2]['prompt']


def test_claude_semantic_authority_retains_mask_and_workspace_checks(saved_scene):
    from cloth_agent.fold_frame import CLAUDE_FOLD_RULE
    session, images, grounding = saved_scene
    measurement = grounding.lookup_reference('A', 'R001')
    objective = 'The current_step is left_sleeve. ' + CLAUDE_FOLD_RULE
    # Central point that fails the old outer-side heuristic is allowed when
    # Claude owns semantics, but still must be valid measured garment fabric.
    result = ClaudeAutoClient._validate_measurement_for_stage2('A', 'R001', measurement, session, objective)
    assert result['fold_semantic_validation']['authority'] == 'Claude'
    with pytest.raises(SelectedReferenceNotExecutableError):
        ClaudeAutoClient._validate_measurement_for_stage2('A', 'R001', measurement, session,
                                                        'The current_step is left_sleeve.')
    mask_path = grounding.perception_dir / 'camera_A_garment_mask.npy'
    np.save(mask_path, np.zeros((30, 40), dtype=bool))
    with pytest.raises(SelectedReferenceNotExecutableError, match='mask'):
        ClaudeAutoClient._validate_measurement_for_stage2('A', 'R001', measurement, session, objective)
    invalid = dict(measurement, base_xyz_mm=[100000, 0, 30])
    with pytest.raises(SelectedReferenceNotExecutableError, match='bound'):
        ClaudeAutoClient._validate_measurement_for_stage2('A', 'R001', invalid, session, objective)


def test_remote_claude_receives_molmo_rgb_hint_and_host_resolves_float_destination(saved_scene):
    session, images, grounding = saved_scene
    hint = session.run_dir / 'camera_A_molmo_hint_upright.png'
    Image.new('RGB', (30, 40), 'magenta').save(hint)
    motion = motion_payload()
    motion['actions'][5]['args'].update(target='pixel', image_id='image_0', pixel_xy=[5.25, 6.75])
    backend = FakeBackend(visual_payload(), motion)
    client = RemoteFoldClient(backend=backend)
    client.plan([*images, hint], session, 'Fold garment')
    for call in backend.calls:
        assert hint in call['image_paths']
        assert 'CLAUDE_FOLD_AUTHORITY_V1' in call['prompt']
    trace = client.last_grounding_verification['image_source_resolution']
    assert trace[0]['grounding_pixel_xy'] == [5, 7]
    assert list(session.run_dir.rglob('pixel_source_resolution.json'))


@pytest.mark.parametrize("corruption", ["nan", "bad_pixel", "depth_hole", "unknown_action", "missing_contact", "workspace", "extra"])
def test_invalid_remote_motion_fails_closed(saved_scene, corruption):
    session, _, grounding = saved_scene
    motion = motion_payload()
    if corruption == "nan":
        motion["actions"][0]["args"]["yaw_deg"] = float("nan")
    elif corruption in {"bad_pixel", "depth_hole"}:
        motion["actions"][5]["args"].update(target="pixel", pixel_xy=[100, 100] if corruption == "bad_pixel" else [9, 10])
        if corruption == "depth_hole":
            path = grounding.perception_dir / "camera_A_base_xyz_mm.npy"
            xyz = np.load(path)
            xyz[20, 10] = np.nan
            np.save(path, xyz)
    elif corruption == "unknown_action":
        motion["actions"].append({"name": "execute", "args": {}})
    elif corruption == "missing_contact":
        motion["actions"][2]["args"]["height_above_grasp_mm"] = 30
    elif corruption == "workspace":
        motion["actions"][0]["args"]["height_above_grasp_mm"] = 1000
    else:
        motion["xyz"] = [0, 0, 0]
    with pytest.raises((ExplorationPlanningError, ValueError, SafetyError)):
        compile_pixel_motion(motion, validate_visual_plan_payload(visual_payload()), grounding,
                             session.robot_config, (30, 40))


def test_supervisor_and_both_evaluators_use_bridge(saved_scene, monkeypatch):
    session, images, grounding = saved_scene
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("local process invoked"))
    backend = FakeBackend(supervisor_payload(), evaluation_payload(), evaluation_payload())
    supervisor = FoldSupervisor(binary="missing", backend=backend)
    result = supervisor.inspect(images, session.run_dir, history=[], screen={"base_xyz_mm": [123456, 0, 0]})
    assert result["backend"] == "remote"
    proposal, _ = compile_pixel_motion(motion_payload(), validate_visual_plan_payload(visual_payload()),
                                      grounding, session.robot_config, (30, 40))
    client = RemoteFoldClient(backend=backend, binary="missing")
    snapshots = []
    for name in ('camera_A_grasp_after_close.png', 'camera_A_grasp_after_lift.png'):
        path = session.run_dir / name
        Image.new('RGB', (10, 10), 'blue').save(path)
        snapshots.append(path)
    client.evaluate(images, images, proposal=proposal, run_dir=session.run_dir,
                    gripper_telemetry={"secret": "private_robot"}, observer_images=snapshots)
    client.evaluate_acquisition_probe(images, images, proposal=proposal, run_dir=session.run_dir)
    assert len(backend.calls) == 3
    for call in backend.calls:
        assert call['max_turns'] == 8
        assert call['image_edit_limit'] == 2
    assert all(path in backend.calls[1]['image_paths'] for path in snapshots)
    assert 'BEFORE lift' in backend.calls[1]['prompt']
    assert 'BEFORE transport' in backend.calls[1]['prompt']
    assert 'Closure confirmation is not proof' in backend.calls[1]['prompt']
    for call in backend.calls:
        assert str(session.run_dir) not in call["prompt"]
        assert "123456" not in call["prompt"]
        assert "private_robot" not in call["prompt"]
        assert images[2] not in call["image_paths"]


def test_parser_and_transport_cleanup(saved_scene, monkeypatch):
    _, images, _ = saved_scene
    png = images[0]
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "curl":
            return SimpleNamespace(returncode=0, stdout='{"success":true,"files":[{"id":"poc-id"}]}', stderr="")
        if cmd[-1].startswith("rm -rf"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(PlannerBackendError, match="SSH"):
        RemoteClaudeBackend(timeout_s=1).invoke(prompt="Read", image_paths=[png], schema={}, system_prompt="Read")
    assert len(calls) == 3
    assert "sha256sum" in calls[1][0][-1]
    assert "https://tempfile.org/poc-id/download" in calls[1][0][-1]
    assert calls[2][0][-1].startswith("rm -rf -- /tmp/cloth_remote_")
    for envelope in ({"result": 'Explanation\n```json\n{"ok":true}\n```'}, {"structured_output": {"ok": True}}):
        assert parse_claude_json(json.dumps(envelope)) == {"ok": True}
    for envelope in ({"is_error": True, "result": '{"ok":true}'}, {"subtype": "error_max_turns", "result": '{"ok":true}'}, {"result": '{} {}'}):
        with pytest.raises(PlannerBackendError):
            parse_claude_json(json.dumps(envelope))


def test_fold_constructor_and_cli_default_remote(saved_scene):
    session, _, _ = saved_scene
    pipeline = FoldExplorationPipeline(session, perception_config=Path("config/perception.free_exploration.json"))
    assert isinstance(pipeline.client, RemoteFoldClient)
    assert isinstance(pipeline.supervisor.backend, RemoteClaudeBackend)
    assert build_parser().parse_args([]).planner_backend == "remote"
    local = FoldExplorationPipeline(session, perception_config=Path("config/perception.free_exploration.json"), planner_backend="local")
    assert not isinstance(local.client, RemoteFoldClient)
    assert local.supervisor.backend is None


def test_failed_remote_supervision_and_evaluation_never_fallback(saved_scene):
    session, images, grounding = saved_scene
    pipeline = FoldExplorationPipeline(session, perception_config=Path("config/perception.free_exploration.json"),
        max_stage_retries=0, retry_backoff_s=0)
    pipeline.supervisor.backend = FakeBackend(PlannerBackendError("unavailable"))
    with pytest.raises(PlannerBackendError, match="no fallback decision"):
        pipeline._supervisor(images, {}, [])
    pipeline.client.backend = FakeBackend(PlannerBackendError("unavailable"))
    proposal, _ = compile_pixel_motion(motion_payload(), validate_visual_plan_payload(visual_payload()),
        grounding, session.robot_config, (30, 40))
    with pytest.raises(PlannerBackendError, match="no fallback outcome"):
        pipeline._evaluate_with_retries(images, images, proposal=proposal, run_dir=session.run_dir, iteration=1)


def test_repair_failure_clears_previous_proposal(saved_scene):
    session, images, _ = saved_scene
    client = RemoteFoldClient(backend=FakeBackend(visual_payload(), motion_payload(), {"wrong_schema": True}))
    client.plan(images, session, "Probe the garment edge")
    assert client.last_plan_result is not None
    with pytest.raises(ExplorationPlanningError):
        client.repair_last_grounding_plan(session, "Probe the garment edge", feedback="workspace rejected")
    assert client.last_plan_result is None
    assert client.last_grounding_verification is None


def test_feedback_keeps_previous_visual_reference_excluded(saved_scene):
    session, images, grounding = saved_scene
    guide_path = grounding.perception_dir / "camera_A_coordinate_guide.json"
    guide = json.loads(guide_path.read_text())
    guide["samples"].append({**guide["samples"][0], "reference_id": "R002"})
    guide_path.write_text(json.dumps(guide))
    visual = visual_payload()
    visual["selected_reference"]["reference_id"] = "R002"
    backend = FakeBackend(visual_payload(), motion_payload(), visual, motion_payload())
    client = RemoteFoldClient(backend=backend)
    client.plan(images, session, "Fold garment")
    client.plan(images, session, "Fold garment", feedback="selected reference is inconsistent with the final grasp")
    context = json.loads(backend.calls[2]["prompt"].split("\n", 1)[1])
    assert context["rejected_references"] == [{"camera": "A", "reference_id": "R001"}]


def test_mismatched_rgb_blocked_before_upload(saved_scene):
    session, images, _ = saved_scene
    Image.new("RGB", (30, 40), (0, 0, 0)).save(images[0])
    backend = FakeBackend()
    with pytest.raises(ExplorationPlanningError, match="does not match"):
        RemoteFoldClient(backend=backend).plan(images, session, "Probe garment")
    assert not backend.calls


def test_lateral_strip_rejected_before_remote_call(saved_scene):
    session, images, grounding = saved_scene
    session.robot_config = replace(session.robot_config, boundaries=WorkspaceBounds(
        lateral_points_mm=((400, -20), (400, 20)), z_min=0))
    with pytest.raises(SelectedReferenceNotExecutableError, match="left/right"):
        ClaudeAutoClient._validate_measurement_for_stage2("A", "R001",
            grounding.lookup_reference("A", "R001"), session)
    backend = FakeBackend()
    with pytest.raises(ReferenceReselectionExhaustedError):
        RemoteFoldClient(backend=backend).plan(images, session,
            "Fold the shirt. The requested smoke-test current_step is left_sleeve. Plan exactly this step.")
    assert backend.calls == []
    report_path = next(session.run_dir.glob("remote_diagnostics/*/workspace_diagnostics.json"))
    report = json.loads(report_path.read_text())
    assert report["references"][0]["xy_eligible"] is False
    assert "render_error" not in report


def test_rotated_strip_signed_distances_and_local_failure_images(saved_scene):
    session, images, grounding = saved_scene
    bounds = WorkspaceBounds(lateral_points_mm=((0, 0), (100, 100)))
    clearances = lateral_clearance(bounds, 150, 100, 5)["signed_clearance_mm"]
    assert clearances[1] == pytest.approx(-50 / 2**.5 - 5)
    session.robot_config = replace(session.robot_config, boundaries=WorkspaceBounds(
        lateral_points_mm=((400, 0), (400, 60)), z_min=0))
    motion = motion_payload()
    motion["actions"][5]["args"].update(target="pixel", pixel_xy=[9, 10], height_above_grasp_mm=30)
    with pytest.raises(WorkspaceTargetError) as failure:
        compile_pixel_motion(motion, validate_visual_plan_payload(visual_payload()),
                             grounding, session.robot_config, (30, 40))
    point = next(p for p in failure.value.trace["moves"] if p.get("error"))
    assert point["action_index"] == 6
    assert point["raw_pixel_xy"] == [10, 20]
    assert point["base_xyz_mm"] == [550, 80, 57]
    assert point["lateral"]["signed_clearance_mm"] == [80, -20]
    directory = session.run_dir / "diagnostic_test"
    report = save_workspace_debug(directory, grounding.perception_dir, session.robot_config, failure.value.trace)
    assert "render_error" not in report
    assert len(report["moves"]) == 5  # Includes waypoints after the rejection, never executed.
    for name in ("workspace_targets_raw.png", "workspace_targets_upright.png", "workspace_base_xy.png"):
        with Image.open(directory / name) as im:
            assert im.width >= 780
            assert np.any(np.asarray(im)[..., 0] > np.asarray(im)[..., 1])
    backend = FakeBackend(visual_payload(), motion)
    motion['actions'][5]['args']['image_id'] = 'image_0'
    client = RemoteFoldClient(backend=backend)
    with pytest.raises(WorkspaceTargetError):
        client.plan(images, session, "Fold the garment")
    assert client.last_plan_result is None
    assert client.last_grounding_verification is None
    for call in backend.calls:
        assert "signed_clearance" not in call["prompt"]
        assert all(not p.name.startswith("workspace_") for p in call["image_paths"])
    context = json.loads(backend.calls[1]["prompt"].split("\n", 1)[1])
    assert context["xy_eligible_transport_pixels_upright"] == [[14, 15]]


def test_workspace_failure_not_blindly_retried(saved_scene):
    session, images, _ = saved_scene
    pipeline = FoldExplorationPipeline(session, perception_config=Path("config/perception.free_exploration.json"),
        max_stage_retries=4, retry_backoff_s=0)
    motion = motion_payload()
    motion["actions"][0]["args"]["height_above_grasp_mm"] = 1000
    backend = FakeBackend(visual_payload(), motion)
    pipeline.client.backend = backend
    with pytest.raises(RuntimeError, match="will not be retried"):
        pipeline._plan_fold_with_retries(images, "Fold garment", [], iteration=1)
    assert len(backend.calls) == 2  # One selection + one motion call, not a second attempt.


@pytest.mark.parametrize("failed", [False, True])
def test_remote_timing_and_failure_snapshot(saved_scene, monkeypatch, failed):
    _, images, _ = saved_scene
    def run(command, **kwargs):
        if command[0] == "curl":
            return SimpleNamespace(returncode=0, stdout='{"id":"timed"}', stderr="")
        if command[-1].startswith("rm -rf"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=int(failed), stdout='{"result":"{}"}',
            stderr="__CLOTH_TIMING__ download_0 2500000000\n__CLOTH_TIMING__ hash_0 1000000\n__CLOTH_TIMING__ claude 7000000000\n")
    monkeypatch.setattr(subprocess, "run", run)
    backend = RemoteClaudeBackend()
    events = []
    backend.progress_callback = lambda *a, **k: events.append((a, k))
    if failed:
        with pytest.raises(PlannerBackendError) as error:
            backend.invoke(prompt="Read", image_paths=images[:1], schema={}, system_prompt="Read")
        timing = error.value.timings
    else:
        timing = backend.invoke(prompt="Read", image_paths=images[:1], schema={}, system_prompt="Read").timings
    assert timing["remote_download_0_s"] == 2.5
    assert timing["remote_claude_s"] == 7
    assert {"upload_0_s", "ssh_download_and_claude_s", "cleanup_s", "total_s"} <= timing.keys()
    assert any(e[0][:2] == ("cleanup", "started") for e in events)


@pytest.mark.parametrize("failure", ["upload", "ssh", "claude", "json", "success"])
def test_transport_result_and_cleanup_paths(saved_scene, monkeypatch, failure):
    _, images, _ = saved_scene
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "curl":
            return SimpleNamespace(returncode=0, stderr="", stdout=(
                '{"success":false}' if failure == "upload" else '{"files":[{"id":"relay-id"}]}'))
        if command[-1].startswith("rm -rf"):
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        return SimpleNamespace(returncode=1 if failure == "ssh" else 0, stderr="error" if failure == "ssh" else "",
            stdout='not json' if failure == "json" else json.dumps({"is_error": failure == "claude", "result": '{"ok":true}'}))
    monkeypatch.setattr(subprocess, "run", run)
    backend = RemoteClaudeBackend()
    kwargs = dict(prompt="Read RGB", image_paths=[images[0]], schema={}, system_prompt="Read only")
    if failure == "success":
        assert parse_claude_json(backend.invoke(**kwargs).stdout)["ok"] is True
    else:
        with pytest.raises(PlannerBackendError):
            backend.invoke(**kwargs)
    assert len(calls) == (1 if failure == "upload" else 3)
    if failure != "upload":
        assert calls[-1][-1].startswith("rm -rf -- /tmp/cloth_remote_")


def test_saved_run_smoke_uses_production_path_without_hardware(saved_scene, monkeypatch):
    from cloth_agent.session import AgentSession
    from scripts.remote_fold_smoke import main
    source, _, _ = saved_scene
    project = source.run_dir / "project"
    session = AgentSession.create(project, "offline bridge test", source.robot_config,
        source.experiment_config, run_id="saved")
    (project / "xarm_boundaries.json").write_text(json.dumps({"boundary_mm": {"z_min": 6}}))
    robot_data = project / "data" / "robot"
    robot_data.mkdir(parents=True)
    (robot_data / "xarm_init_pose.json").write_text(json.dumps({
        "joint_angles_deg": [0] * 6, "tcp_pose_mm_deg": [500, 0, 280, 180, 0, 0]}))
    shutil.copytree(source.workspace / "perception_views", session.workspace / "perception_views")
    backend = FakeBackend(supervisor_payload(), visual_payload(), motion_payload())
    monkeypatch.setattr(RemoteClaudeBackend, "invoke", lambda self, **kwargs: backend.invoke(**kwargs))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("hardware/local process invoked"))
    assert main(["--project-root", str(project), "--run-dir", str(session.run_dir), "--step", "left_side"]) == 0
    proposals = list(session.results.glob("remote_fold_smoke/*/proposal.json"))
    assert len(proposals) == 1
    result = json.loads(proposals[0].read_text())
    assert result["hardware_connected"] is False
    assert result["proposal"]["actions"][2]["args"]["z"] == 27
    assert len(backend.calls) == 3
    # The offline visualizer can replay these exact saved payloads without a
    # fourth remote call, using the saved run's effective robot configuration.
    from scripts.debug_workspace_targets import main as debug_main
    motion_log = next(session.results.glob("remote_fold_smoke/*/remote_diagnostics/*/pixel_motion_invocation.json"))
    visual_log = motion_log.with_name("visual_planning_invocation.json")
    output = session.results / "offline_workspace_debug"
    assert debug_main(["--project-root", str(project), "--run-dir", str(session.run_dir),
        "--motion-json", str(motion_log), "--visual-json", str(visual_log), "--output-dir", str(output)]) == 0
    assert (output / "workspace_base_xy.png").is_file()
    assert len(backend.calls) == 3


@pytest.mark.parametrize("download_succeeds", [False, True])
def test_remote_shell_does_not_run_claude_after_download_or_hash_failure(saved_scene, monkeypatch, download_succeeds):
    session, images, _ = saved_scene
    real_run = subprocess.run
    curl_stub = session.run_dir / "curl_stub.sh"
    # Successful GET with no file simulates missing/corrupt download; sha256
    # must reject it. A failed GET must not reach the hash or the model either.
    curl_stub.write_text("exit 0\n" if download_succeeds else "exit 22\n")
    called = []
    def run(command, **kwargs):
        if command[0] == "curl":
            return SimpleNamespace(returncode=0, stderr="", stdout='{"id":"relay"}')
        remote = command[-1]
        if remote.startswith("rm -rf"):
            return real_run(["sh", "-c", remote], **kwargs)
        remote = remote.replace("curl -fsSL", "sh " + shlex.quote(str(curl_stub)) + " -fsSL")
        # If command sequencing is wrong, 'echo' produces a unique marker.
        remote = remote.replace("claude -p", "echo CLAUDE_WAS_CALLED")
        result = real_run(["sh", "-c", remote], **kwargs)
        called.append(result)
        return result
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(PlannerBackendError, match="exited"):
        RemoteClaudeBackend().invoke(prompt="RGB only", image_paths=images[:2], schema={}, system_prompt="read")
    assert len(called) == 1
    assert "CLAUDE_WAS_CALLED" not in called[0].stdout


def test_real_remote_shell_success_reports_timings_and_cleans_job(saved_scene, monkeypatch):
    session, images, _ = saved_scene
    real_run = subprocess.run
    stub = session.run_dir / "claude_stub.sh"
    stub.write_text("printf '%s' '{\"result\":\"{\\\"ok\\\":true}\"}'\n")
    def run(command, **kwargs):
        if command[0] == "curl":
            return SimpleNamespace(returncode=0, stdout='{"id":"relay"}', stderr="")
        remote = command[-1]
        if not remote.startswith("rm -rf"):
            import re
            remote = re.sub(r"curl -fsSL --connect-timeout 20 --max-time 120 \S+ -o (\S+)",
                lambda m: f"cp {shlex.quote(str(images[0]))} {m[1]}", remote)
            remote = remote.replace("claude -p", f"sh {shlex.quote(str(stub))}")
        return real_run(["sh", "-c", remote], **kwargs)
    monkeypatch.setattr(subprocess, "run", run)
    result = RemoteClaudeBackend().invoke(prompt="Read", image_paths=images[:1], schema={}, system_prompt="Read")
    assert parse_claude_json(result.stdout) == {"ok": True}
    assert {"remote_download_0_s", "remote_hash_0_s", "remote_claude_s"} <= result.timings.keys()
    import re
    job = re.search(r"/tmp/cloth_remote_[a-f0-9]+", result.command[-1]).group()
    assert not Path(job).exists()
