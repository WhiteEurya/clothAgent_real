from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from cloth_agent.experiments import goal_only_episode_continue as c


def test_continuation_keeps_only_completed_steps_and_detects_changes(tmp_path):
    source, root = tmp_path/"old", tmp_path/"new"
    (source/"goal").mkdir(parents=True)
    Image.new("RGB", (10, 10), "purple").save(source/"goal/target.png")
    executed = dict(status="EXECUTED", physical_execution=True)
    c.shared.save(source/"episode_summary.json", dict(
        condition="GOAL_ONLY_NO_TRAJECTORY_MEMORY", termination_reason="SAFETY_STOP",
        total_rounds=2, physical_actions=1, episode_wall_s=20,
        rounds=[dict(round="round_000", execution=executed, acquisition="UNKNOWN_PENDING_HUMAN_REVIEW"),
                dict(round="round_001", execution=dict(status="SAFETY_REJECTED", physical_execution=False))]))
    for name in ("before", "after"):
        (source/"round_000"/name).mkdir(parents=True)
        Image.new("RGB", (10, 10), "purple").save(source/"round_000"/name/"rgb.png")
    for name, data in [("execution.json", executed), ("plan.json", dict(decision="ACTION")), ("timing.json", {})]:
        c.shared.save(source/"round_000"/name, data)
    c.freeze_continuation(root, source, [])
    history = c.load_history(root)
    assert len(history) == 1 and history[0]["round"] == "prefix_round_000"
    assert not (root/"memory").exists()
    assert c.shared.read(root/"continuation.json")["skipped_rejected_rounds"] == ["round_001"]
    c.verify(root)
    (history[0]["directory"]/"plan.json").write_text("{}")
    with pytest.raises(c.shared.DualArmError, match="prefix changed"):
        c.verify(root)


def test_shift_planning_failure_never_dispatches(tmp_path, monkeypatch):
    runtime = object.__new__(c.ShiftedObservationRuntime)
    runtime.config = SimpleNamespace(arms={"left": SimpleNamespace(world_from_base=np.eye(4))})
    runtime.models = {"left": SimpleNamespace(forward=lambda q: [0, 0, 520, 0, 0, 0])}
    monkeypatch.setattr(c.shared.EpisodeCollectionRuntime, "move_to_perception", lambda *args: None)
    runtime.capture_current = lambda path: SimpleNamespace(meta=dict(
        joints_deg={"left": [0]*6}, world_from_camera_mm=np.eye(4).tolist()))
    dispatched = []
    runtime._execute_operations = lambda *args: dispatched.append(True)
    def reject(*args, **kwargs):
        raise c.shared.DualArmError("IK rejected")
    runtime.plan = reject
    with pytest.raises(c.shared.DualArmError, match="IK rejected"):
        runtime.move_to_perception(tmp_path)
    assert not dispatched
