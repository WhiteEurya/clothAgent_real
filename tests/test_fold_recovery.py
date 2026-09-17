import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.fold_recovery import (checkpoint_evaluation, failure_detection, failure_skill,
                                       inherit_fold_lessons, released_and_homed)
from cloth_agent.fold_exploration_pipeline import (FoldExplorationPipeline, FoldExperienceStore,
    _confirmed_completion_ledger, _fold_acquisition_learning_state, _acquisition_supervisor_reuse_step)
from cloth_agent.skill_lifecycle import RunSkillLedger, SkillStore
from cloth_agent.grasp_checkpoint import inspect_grasp, GraspCheckpointRejected
from cloth_agent.fold_exploration_viser import _error_html
from scripts.watch_fold_exploration import classify_child_exit


def execution(kind="EMPTY", *, success=True):
    return {"execution_completed": success, "physical_execution": True, "robot_errors": [],
            "actual_robot_actions": [{"name": "close_gripper"}, {"name": "move"},
                                     {"name": "open_gripper"}, {"name": "move"}, {"name": "home"}],
            "mandatory_return_home": {"completed": success},
            "checkpoint": {"status": "ASSESSED", "classification": kind, "confidence": .9,
                "evidence": ["Clearly empty jaws after the lift"], "reason": "No retained fabric",
                "continue_transport": False, "executed_branch": "ABORT_RELEASE"}}


def learning_pipeline(tmp_path):
    pipe = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipe.session = SimpleNamespace(run_dir=tmp_path, results=tmp_path / "results")
    pipe.experiences = FoldExperienceStore(tmp_path)
    pipe.skill_ledger = RunSkillLedger(tmp_path / "workspace")
    pipe.skill_store = SkillStore(tmp_path / "skills")
    pipe._debug_exception = lambda *a, **kw: None
    return pipe


def test_empty_and_unknown_saved_once_and_only_empty_learns_detector(tmp_path):
    pipe = learning_pipeline(tmp_path)
    for i, kind in enumerate(("EMPTY", "UNKNOWN"), 1):
        result = execution(kind)
        record = {"iteration": i, "mode": "FOLD", "planned_step": "left_sleeve",
                  "status": "GRASP_REJECTED", "execution": result,
                  "evaluation": checkpoint_evaluation(result["checkpoint"])}
        path = tmp_path / f"iteration_{i:03d}"
        pipe._save_iteration_learning(path, record)
        pipe._save_iteration_learning(path, record)
        assert (path / "failure_detection.json").is_file()
        assert (path / "record.json").is_file()
    rows = pipe.experiences.history(limit=None)
    assert len(rows) == 2
    assert rows[0]["failure_detection"]["category"] == "EMPTY_GRASP"
    assert rows[1]["evaluation"]["grasp_acquisition"]["status"] == "UNKNOWN"
    assert _fold_acquisition_learning_state(rows, "left_sleeve")["consecutive_acquisition_failures"] == 1
    assert len(pipe.skill_ledger.candidates_path.read_text().splitlines()) == 1
    assert "fold-empty-grasp-detection" in pipe._skill_prompt()
    assert pipe.skill_ledger.finalize(pipe.skill_store)["skill_group_count"] == 1


@pytest.mark.parametrize("mutate", [
    lambda e: e.update(execution_completed=False),
    lambda e: e.update(robot_errors=["servo error"]),
    lambda e: e.update(operator_interrupted=True),
    lambda e: e.update(gripper_completion_failed=True),
    lambda e: e.update(mandatory_return_home={"completed": False}),
    lambda e: e["actual_robot_actions"].pop(),
    lambda e: e["actual_robot_actions"].append({"name": "close_gripper"}),
])
def test_unconfirmed_release_or_home_cannot_authorize_recovery(mutate):
    value = execution()
    assert released_and_homed(value)
    mutate(value)
    assert not released_and_homed(value)


def test_black_images_skip_model_without_teaching_empty_grasp(tmp_path):
    paths = []
    for stage in ("close", "lift"):
        path = tmp_path / f"camera_A_grasp_after_{stage}.png"
        Image.new("RGB", (40, 40)).save(path)
        paths.append(path)
    backend = SimpleNamespace(invoke=lambda **kw: pytest.fail("black images must not call Claude"))
    result = inspect_grasp(backend, paths, tmp_path)
    assert result["classification"] == "UNKNOWN"
    assert result["status"] == "EVIDENCE_UNUSABLE"
    assert checkpoint_evaluation(result)["grasp_acquisition"]["status"] == "UNKNOWN"
    assert failure_skill({"evaluation": checkpoint_evaluation(result)}) is None


def test_inherit_lessons_without_transplanting_physical_state(tmp_path):
    old = tmp_path / "runs" / "old"
    store = FoldExperienceStore(old)
    record = {"iteration": 1, "mode": "FOLD", "planned_step": "left_sleeve",
              "evaluation": checkpoint_evaluation(execution()["checkpoint"]),
              "supervisor_after": {"completed_steps": ["left_sleeve"]},
              "garment_condition_after": {"condition": "BUNCHED"}}
    store.append(record)
    new = tmp_path / "runs" / "new"
    result = inherit_fold_lessons(tmp_path, new)
    rows = FoldExperienceStore(new).history(limit=None)
    assert result["count"] == 1 and result["completion_inherited"] is False
    assert rows[0]["evaluation"] == record["evaluation"]
    assert _confirmed_completion_ledger(rows) == []
    assert _acquisition_supervisor_reuse_step(rows) is None
    assert _fold_acquisition_learning_state(rows, "left_sleeve")["attempt_count"] == 0


def test_watchdog_respects_explicit_stop_and_new_checkpoint_failure():
    last = {"stage": "run", "message": "pipeline finished"}
    assert classify_child_exit(1, {"status": "FAILED", "restart_safe": False}, last) == "STOP"
    assert classify_child_exit(1, {"status": "FAILED", "error": "GraspCheckpointRejected: cannot recover"}, last) == "STOP"


def test_red_error_panel_preserves_recovery_and_escapes_content(tmp_path):
    event = {"elapsed_s": 2, "stage": "recovery", "level": "ERROR",
             "message": "<script>bad</script>", "fields": {"recovery_status": "RECOVERED"}}
    (tmp_path / "debug_events.jsonl").write_text(json.dumps(event) + '\n{"partial":')
    html = _error_html(tmp_path)
    assert "#ff6b6b" in html and "RECOVERED" in html
    assert "<script>" not in html


def test_partial_execution_error_is_persisted_on_interrupt(tmp_path):
    pipe = learning_pipeline(tmp_path)
    iteration = tmp_path / "iteration_001"
    iteration.mkdir()
    result = execution()
    result["operator_interrupted"] = True
    (iteration / "execution.json").write_text(json.dumps(result))
    pipe._active_iteration = (iteration, {"iteration": 1, "planned_step": "left_sleeve"})
    pipe._last_operational_stage = "execution"
    row = pipe._save_interrupted_iteration(KeyboardInterrupt())
    assert row["failure_detection"]["category"] == "OPERATOR_INTERRUPTED"
    assert len(pipe.experiences.history(limit=None)) == 1


def make_loop(tmp_path, monkeypatch, *, kind="EMPTY", safe=True):
    from cloth_agent import fold_exploration_pipeline as module
    from cloth_agent.config import RobotConfig, WorkspaceBounds, ExperimentConfig
    from cloth_agent.session import AgentSession
    from cloth_agent.free_exploration import ExplorationProposal
    from cloth_agent.robot_api import ControllerTrajectoryValidation
    robot = RobotConfig(robot_ip="test", boundaries=WorkspaceBounds(x_min=0, x_max=600,
        y_min=-300, y_max=300, z_min=0, z_max=500), init_joints_deg=(0,)*6,
        init_pose_mm_deg=(300, 0, 200, 180, 0, 0), orientation_roll_deg=180, orientation_pitch_deg=0)
    session = AgentSession(tmp_path, tmp_path / "runs" / "test", robot, ExperimentConfig())
    pipe = FoldExplorationPipeline(session, perception_config=tmp_path / "perception.json",
        max_iterations=2, max_replans=0, unattended=True, record_video=False, retry_backoff_s=0)
    monkeypatch.setattr(module.PerceptionConfig, "load", lambda *a: SimpleNamespace(active_camera_labels=("A",)))
    monkeypatch.setattr(module, "assess_screen_visibility", lambda *a, **kw: {"visibility": "FULL"})
    monkeypatch.setattr(module, "validate_controller_trajectory", lambda *a: ControllerTrajectoryValidation(
        joint_targets_rad={}, controller_warning_code=0, tcp_offset_mm_deg=(0,)*6, validated_sample_count=1))
    pipe._single_view_execution_confirmation = lambda *a: True
    def capture(config, path, **kwargs):
        path.mkdir(parents=True)
        rgb = path / "camera_A_rgb_upright.png"
        Image.new("RGB", (16, 16), (150, 150, 150)).save(rgb)
        return {}, path / "perception.json", [rgb]
    pipe._capture_with_retries = capture
    def supervisor(*a, **kw):
        return {"status": "READY", "current_step": "left_sleeve", "completed_steps": [],
            "garment_visibility": "FULL", "trajectory_decision": "CONTINUE", "confidence": .9,
            "evidence": ["Unfolded shirt"], "reason": "Fold sleeve"}
    pipe._supervisor = supervisor
    pipe._locate_sleeve_with_molmo = lambda **kw: None
    pipe._resolve_fold_grasp_height = lambda p: (p, None)
    actions = []
    def move(x, z): return {"name": "move", "args": {"x": x, "y": 0., "z": z, "yaw": 0.}}
    actions = [move(300, 100), {"name": "open_gripper", "args": {}}, move(300, 20),
        {"name": "close_gripper", "args": {}}, move(300, 80), move(400, 80), move(400, 20),
        {"name": "open_gripper", "args": {}}, move(400, 100), {"name": "home", "args": {}}]
    proposal = ExplorationProposal(garment_observation="shirt", reveal_strategy="fold sleeve",
        confidence=.9, actions=tuple(actions), expected_observation="sleeve folds", safety_notes=("workspace checked",),
        selected_grasp={"camera": "A", "pixel_xy": [1, 1]})
    plans = []
    def plan(images, objective, history, **kw):
        plans.append(list(history))
        if len(plans) > 1:
            raise KeyboardInterrupt()  # observe next-iteration feedback, no second motion
        return proposal
    pipe._plan_fold_with_retries = plan
    def execute(source, config, iteration_dir, **kwargs):
        assert kwargs['grasp_capture']['lift_mm'] >= 30
        result = execution(kind, success=safe)
        result.pop('checkpoint')
        image = iteration_dir / 'hold_check' / 'camera_A_grasp_after_lift.png'
        image.parent.mkdir(parents=True)
        Image.new('RGB', (16, 16), 'white').save(image)
        return result, {'status': 'disabled', 'grasp_snapshots': {'after_lift': {
            'status': 'CAPTURED', 'image': str(image), 'asynchronous': True, 'requested_lift_mm': 60}}}
    pipe._execute = execute
    def evaluate(before, after, **kwargs):
        assert any(p.name == 'camera_A_grasp_after_lift.png' for p in kwargs['observer_images'])
        assert 'at least 30 mm' in kwargs['objective']
        assert 'not stationary pre-transport checkpoints' in kwargs['objective']
        return checkpoint_evaluation(execution(kind)['checkpoint']), None
    pipe._evaluate_with_retries = evaluate
    return pipe, plans


@pytest.mark.parametrize("kind", ["EMPTY", "UNKNOWN"])
def test_full_loop_final_evaluation_receives_lift_photo_and_next_plan_sees_failure(tmp_path, monkeypatch, kind):
    pipe, plans = make_loop(tmp_path, monkeypatch, kind=kind)
    with pytest.raises(KeyboardInterrupt):
        pipe.run()
    assert len(plans) == 2
    assert plans[1][-1]["failure_detection"]["category"] == ("EMPTY_GRASP" if kind == "EMPTY" else "GRASP_UNOBSERVABLE")
    assert plans[1][-1]["supervisor_after"]["current_step"] == "left_sleeve"


def test_full_loop_never_retries_unconfirmed_abort(tmp_path, monkeypatch):
    pipe, plans = make_loop(tmp_path, monkeypatch, safe=False)
    with pytest.raises(GraspCheckpointRejected):
        pipe.run()
    assert len(plans) == 1
    rows = pipe.experiences.history(limit=None)
    assert rows[0]["failure_detection"]["category"] == "EXECUTION_UNCONFIRMED"
    summary_path = next(pipe.session.results.glob("fold_exploration/*/summary.json"))
    assert json.loads(summary_path.read_text())["restart_safe"] is False


def test_full_success_path_stages_skill_and_persists_evidence(tmp_path, monkeypatch):
    pipe, plans = make_loop(tmp_path, monkeypatch)
    result = execution("GRASP_CONFIRMED")
    result["checkpoint"].update(executed_branch="CONTINUATION", continue_transport=True)
    pipe._execute = lambda *a, **kw: (result, {"status": "disabled"})
    evaluation = checkpoint_evaluation({})
    evaluation["grasp_acquisition"] = {"status": "SUCCESS", "confidence": .9, "evidence": ["Lifted fabric"]}
    evaluation["skill_update"] = failure_skill({"evaluation": checkpoint_evaluation(execution()["checkpoint"])}).as_dict()
    pipe._evaluate_with_retries = lambda *a, **kw: (evaluation, None)
    before = pipe._supervisor()
    after = {**before, "status": "COMPLETE", "current_step": "COMPLETE",
             "completed_steps": ["left_sleeve", "right_sleeve", "left_side", "right_side", "hem_up"]}
    states = iter([before, after])
    pipe._supervisor = lambda *a, **kw: next(states)
    assert pipe.run()["status"] == "COMPLETE"
    rows = pipe.experiences.history(limit=None)
    assert len(rows) == 1 and rows[0]["skill_review"]["status"] == "RUN_LOCAL_PENDING"
    assert rows[0]["failure_detection"]["category"] == "NONE"
    assert len(pipe.skill_ledger.candidates_path.read_text().splitlines()) == 1


def test_evaluation_retry_reuses_images_without_reexecuting_robot(tmp_path):
    pipe = learning_pipeline(tmp_path)
    pipe.unattended = True
    pipe.max_stage_retries = 0
    pipe.retry_backoff_s = 0
    pipe._debug = lambda *a, **kw: None
    calls = []
    def evaluate(before, after, **kwargs):
        calls.append((before, after))
        if len(calls) == 1:
            raise TimeoutError("transient provider timeout")
        return checkpoint_evaluation({})
    pipe.client = SimpleNamespace(evaluate=evaluate, last_evaluation_result=None)
    result, _ = pipe._evaluate_with_retries(["before.png"], ["after.png"], iteration=1)
    assert result["grasp_acquisition"]["status"] == "UNKNOWN"
    assert calls == [(["before.png"], ["after.png"])] * 2


def test_video_archival_keeps_failed_rollouts_and_does_not_append_twice(tmp_path, monkeypatch):
    from cloth_agent import fold_recovery as module
    iteration = tmp_path / "iteration_001"
    recording = iteration / "rollout_recording"
    recording.mkdir(parents=True)
    (recording / "camera_A_rgb.mp4").write_bytes(b"source")
    calls = []
    monkeypatch.setattr(module, "label_iteration_mp4", lambda *a, **kw: {})
    monkeypatch.setattr(module, "speed_up_mp4", lambda *a, **kw: {})
    monkeypatch.setattr(module, "append_mp4_to_cumulative", lambda *a, **kw: calls.append(a) or {})
    monkeypatch.setattr(module, "prune_rollout_video_files", lambda *a: pytest.fail("failure video must remain"))
    record = {"iteration": 1, "status": "GRASP_REJECTED", "execution": execution(),
              "recording": {"directory": str(recording)}}
    assert module.archive_iteration_video(iteration, record, prune=True)["status"] == "ARCHIVED"
    module.archive_iteration_video(iteration, record, prune=True)
    assert len(calls) == 1
    assert (recording / "camera_A_rgb.mp4").is_file()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required for video integration")
def test_real_video_archive_and_opt_in_pruning(tmp_path):
    from cloth_agent.fold_recovery import archive_iteration_video
    for i in (1, 2):
        iteration = tmp_path / f"iteration_{i:03d}"
        recording = iteration / "rollout_recording"
        recording.mkdir(parents=True)
        source = recording / "camera_A_rgb.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=30",
                        "-t", "2", "-c:v", "libx264", str(source)], check=True, capture_output=True)
        record = {"iteration": i, "status": "FOLD", "recording": {"directory": str(recording)}}
        result = archive_iteration_video(iteration, record, prune=True)
        assert Path(result["video"]).is_file()
        assert not source.exists()
        assert str(source) in result["pruned_files"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
                            str(tmp_path / "combined_rollout.mp4")], check=True, capture_output=True, text=True)
    assert float(json.loads(probe.stdout)["format"]["duration"]) > 0


def test_corrected_preflight_becomes_provisional_skill(tmp_path):
    pipe = learning_pipeline(tmp_path)
    record = {"iteration": 1, "status": "FOLD", "planning_attempts": [
        {"attempt": 1, "status": "REJECTED_BEFORE_EXECUTION", "error": "ExperimentValidationError: invalid sequence"},
        {"attempt": 2, "status": "ACCEPTED"}]}
    pipe._save_iteration_learning(tmp_path / "iteration_001", record)
    candidates = [json.loads(line) for line in pipe.skill_ledger.candidates_path.read_text().splitlines()]
    assert candidates[0]["source"] == "validated_preexecution_correction"
    assert candidates[0]["proposal"]["name"] == "preflight-contract-recovery"


def test_recovery_exhaustion_is_persisted_for_watchdog(tmp_path, monkeypatch):
    from cloth_agent.fold_recovery import RecoveryExhausted
    pipe, _ = make_loop(tmp_path, monkeypatch)
    def unavailable(*args, **kwargs):
        raise RuntimeError("perception unavailable")
    pipe._capture_with_retries = unavailable
    with pytest.raises(RecoveryExhausted):
        pipe.run()
    summaries = sorted(pipe.session.results.glob("fold_exploration/*/summary.json"))
    assert len(summaries) == 3
    summary = json.loads(summaries[-1].read_text())
    assert summary["restart_safe"] is False
    assert classify_child_exit(1, summary, {"stage": "run"}) == "STOP"
    assert len(pipe.experiences.history(limit=None)) == 3


def test_repeated_planning_failure_stops_after_records_are_saved(tmp_path, monkeypatch):
    from cloth_agent.fold_recovery import RecoveryExhausted
    pipe, _ = make_loop(tmp_path, monkeypatch)
    pipe.max_iterations = 0
    def invalid(*args, **kwargs):
        raise RuntimeError("no executable reference")
    pipe._plan_fold_with_retries = invalid
    with pytest.raises(RecoveryExhausted):
        pipe.run()
    summary = json.loads(next(pipe.session.results.glob("fold_exploration/*/summary.json")).read_text())
    assert len(summary["iterations"]) == 3
    assert len(pipe.experiences.history(limit=None)) == 3
    assert summary["restart_safe"] is False
