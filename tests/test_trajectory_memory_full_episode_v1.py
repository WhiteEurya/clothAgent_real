from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest

from cloth_agent.experiments import trajectory_memory_full_episode_v1 as m


def plan(decision="ACTION"):
    return dict(decision=decision, grasp_px=[20, 20] if decision == "ACTION" else None,
                target_px=[50, 40] if decision == "ACTION" else None,
                motion_type="lift_pull_place" if decision == "ACTION" else None,
                supporting_experiences=[], short_rationale="visible edge")


def test_strict_decisions_and_primitive():
    for decision in ("ACTION", "DONE", "ABSTAIN"):
        m.validate_plan(plan(decision), [100, 100], [])
    for change in (dict(motion_type="custom_motion"), dict(grasp_px=[20.5, 20]),
                   dict(target_px=[120, 40]), dict(decision="DONE")):
        with pytest.raises(Exception):
            m.validate_plan(dict(plan(), **change), [100, 100], [])


def test_repeat_guard_and_acquisition_distinction(tmp_path):
    a, b = tmp_path/"a.png", tmp_path/"b.png"
    Image.new("RGB", (100, 100), "purple").save(a)
    Image.new("RGB", (100, 100), "black").save(b)
    past = dict(plan=plan(), before=a, acquisition="UNKNOWN_PENDING_HUMAN_REVIEW")
    assert m.repeated_action(plan(), a, [past])
    assert not m.repeated_action(plan(), b, [past])
    past["acquisition"] = "ACQUISITION_FAILURE"
    assert m.repeated_action(plan(), b, [past])
    assert not m.repeated_action(plan("DONE"), a, [past])


def test_policy_exactly_inherits_collection_runtime():
    assert m.EpisodeCollectionRuntime.load_config is m.CollectionRuntime.load_config
    assert m.EpisodeCollectionRuntime._plan_preparation is m.CollectionRuntime._plan_preparation
    assert m.EpisodeCollectionRuntime.plan is m.CollectionRuntime.plan
    assert m.EpisodeCollectionRuntime._tcp_segments is m.CollectionRuntime._tcp_segments
    assert m.EpisodeCollectionRuntime.move_to_perception is m.CollectionRuntime.move_to_perception


def test_inherited_collection_settings_not_overridden(monkeypatch):
    config = SimpleNamespace(path_validation_enabled=False, collision_envelopes_disabled_arms={"left"})
    monkeypatch.setattr(m.CollectionRuntime, "load_config", lambda *args: config)
    runtime = object.__new__(m.EpisodeCollectionRuntime)
    assert runtime.load_config("unused") is config
    assert config.path_validation_enabled is False
    assert config.collision_envelopes_disabled_arms == {"left"}
    sentinel = object()
    monkeypatch.setattr(m.CollectionRuntime, "_plan_preparation", lambda self, source, initial: sentinel)
    assert runtime._plan_preparation("source", {}) is sentinel


@pytest.mark.parametrize("reject", [True, False])
@pytest.mark.parametrize("skip_initialization", [False, True])
def test_closed_loop_safety_stop_or_done_without_extra_dispatch(tmp_path, monkeypatch, reject, skip_initialization):
    import cloth_agent.random_fold_reset as reset
    import cloth_agent.gamepad_collect as collection
    calls = []
    class Runtime:
        def __init__(self, config):
            self.cancel = threading.Event()
        def capture_current(self, path):
            path.mkdir()
            # A genuinely changed image following execution avoids the repeat guard.
            Image.new("RGB", (100, 100), "purple" if len(calls) == 0 else "blue").save(path/"rgb.png")
            return SimpleNamespace(directory=path, image=path/"rgb.png")
        def observe(self, path):
            return self.capture_current(path)
        def arrival(self, initial):
            pass
        def _execute_operations(self, operations, folder):
            calls.append("dispatch")
            return dict(status="EXECUTED", physical_execution=True)
        def move_to_perception(self, folder):
            calls.append("return")
        def stop(self):
            calls.append("stop")
            return {}
        def close(self):
            calls.append("close")
    class Recorder:
        def __init__(self, *args):
            pass
        def start(self):
            pass
        def stage(self, *args):
            pass
        def check(self):
            pass
        def close(self):
            pass
    def freeze(root, *_):
        result = dict(files={}, records=[], memory_steps=0, memory_sessions=0)
        m.save(root/"memory_manifest.json", result)
        return result
    def agent(root, folder, before, history, manifest, timeout):
        # Only fully executed previous rounds can reach the next planner.
        assert len(history) == (0 if not calls else 1)
        for past in history:
            assert (past["directory"]/"after/rgb.png").exists()
            assert m.read(past["directory"]/"execution.json")["status"] == "EXECUTED"
        p = plan("ACTION" if not history else "DONE")
        metrics = dict(wall_latency_s=1., memory_step_files_read=0, memory_sessions_read=0,
                       historical_source_images_read=0, cited_experience_read_checks=[], session_ids=[folder.name])
        m.save(folder/"plan.json", p)
        m.save(folder/"agent_events/metrics.json", metrics)
        return p, metrics
    def prepare(*args):
        if reject:
            raise m.DualArmError("collision rejection")
        return [], {}
    monkeypatch.setattr(m, "freeze_memory", freeze)
    monkeypatch.setattr(m, "EpisodeCollectionRuntime", Runtime)
    reset_calls = []
    def run_reset(*args, **kwargs):
        reset_calls.append(True)
        return dict(attempts=[dict(executed=True)], elapsed_s=1.)
    monkeypatch.setattr(reset, "run_reset", run_reset)
    monkeypatch.setattr(collection, "TrajectoryRecorder", Recorder)
    monkeypatch.setattr(m, "agent_round", agent)
    monkeypatch.setattr(m, "prepare_action", prepare)
    monkeypatch.setattr(m, "render", lambda *args: None)
    root = tmp_path/"episode"
    args = SimpleNamespace(output=root, config=Path("config/dual_arm.local.json"), data_root=tmp_path,
                           scope_dates=[], max_rounds=12, timeout_s=10, skip_initialization=skip_initialization)
    m.run(args)
    result = m.read(root/"episode_summary.json")
    assert result["termination_reason"] == ("SAFETY_STOP" if reject else "DONE")
    assert calls.count("dispatch") == (0 if reject else 1)
    assert result["physical_actions"] == (0 if reject else 1)
    assert result["total_rounds"] == (1 if reject else 2)
    assert result["human_fold_ready"] is None
    assert result["acquisition_failures"] == 0
    assert len(reset_calls) == (0 if skip_initialization else 1)
    assert result["initialization_physical_actions"] == (0 if skip_initialization else 1)
    if not reject:
        assert result["acquisition_unknown"] == 1
        assert (root/"round_000/after/rgb.png").exists()
