import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from cloth_agent.experiments import goal_only_full_episode_v1 as goal
from cloth_agent.experiments import trajectory_memory_full_episode_v1 as shared


def test_freeze_only_target_rgb_and_detect_changes(tmp_path):
    source = tmp_path/"previous_episode"
    source.mkdir()
    Image.new("RGB", (100, 100), "purple").save(source/"rgb.png")
    (source/"plan.json").write_text('{"secret_previous_action":[10,20]}')
    root = tmp_path/"new"
    manifest = goal.prepare_goal(root, source/"rgb.png", [])
    assert manifest["records"] == []
    assert not (root/"memory").exists()
    assert list((root/"goal").iterdir()) == [root/"goal/target.png"]
    shared.verify_memory(root)
    Image.new("RGB", (100, 100), "blue").save(root/"goal/target.png")
    with pytest.raises(shared.DualArmError, match="changed"):
        shared.verify_memory(root)


def test_goal_agent_gets_two_images_and_no_memory_mount(tmp_path, monkeypatch):
    root = tmp_path/"run"
    source = tmp_path/"goal.png"
    Image.new("RGB", (100, 100), "purple").save(source)
    manifest = goal.prepare_goal(root, source, [])
    folder = root/"round_000"
    before_dir = folder/"before"
    before_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "blue").save(before_dir/"rgb.png")
    before = SimpleNamespace(directory=before_dir, image=before_dir/"rgb.png",
                             xyz=np.ones((100, 100, 3), dtype=float))
    observed = {}
    def isolate(work, memory):
        observed.update(work=work, memory=memory)
        return ["bwrap", "--bind", str(work), "/work"]
    class Process:
        returncode = 0
        def __init__(self, cmd, **kwargs):
            observed["cmd"] = cmd
            self.stdout = kwargs["stdout"]
        def communicate(self, prompt, timeout):
            observed["prompt"] = prompt
            # Synthetic actual read verifies the access monitor for the target.
            (root/"goal/target.png").read_bytes()
            shared.save(observed["work"]/"final_answer.json", dict(
                decision="ACTION", grasp_px=[20, 20], target_px=[50, 40],
                motion_type="lift_pull_place", supporting_experiences=[], short_rationale="target-directed"))
            for item in (dict(type="thread.started", thread_id="test-independent"),
                         dict(type="turn.completed", usage={})):
                self.stdout.write(json.dumps(item)+"\n")
            self.stdout.flush()
    monkeypatch.setattr(shared.probe, "isolated_command", isolate)
    monkeypatch.setattr(shared.subprocess, "Popen", Process)
    plan, metrics = goal.plan_round(root, folder, before, [], manifest, 10)
    assert observed["memory"] is None
    assert "/memory" not in observed["cmd"]
    assert observed["cmd"].count("--image") == 2
    assert "/goal/target.png" in observed["cmd"]
    assert str(root/"goal") in observed["cmd"]
    assert metrics["goal_image_file_read"]
    assert metrics["historical_source_images_read"] == 0
    assert plan["supporting_experiences"] == []
    assert observed["prompt"] == goal.PROMPT
    request = shared.read(folder/"agent_events/request.json")
    assert request["historical_memory_mounted"] is False
    assert request["prior_rounds"] == []


def test_reject_history_in_goal_only_input(tmp_path):
    (tmp_path/"memory").mkdir()
    with pytest.raises(ValueError, match="historical memory"):
        goal.plan_round(tmp_path, None, None, [], dict(records=[]), 10)
