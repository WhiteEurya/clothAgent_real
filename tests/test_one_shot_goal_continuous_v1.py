import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.experiments import one_shot_goal_continuous_v1 as m


def arguments(tmp_path, **overrides):
    goal = tmp_path/"goal.png"
    Image.new("RGB", (20, 20), "purple").save(goal)
    defaults = dict(execute=True, output=tmp_path/"batch", goal_image=goal,
        config=m.ROOT/"config/dual_arm.local.json", episodes=0, max_rounds=12,
        between_episodes_s=0., timeout_s=600., gate_timeout_s=120.,
        review_initialization=False)
    return SimpleNamespace(**(defaults | overrides))


def result(reason="DONE", **extra):
    return dict(termination_reason=reason, finished_at=m.utc(), total_rounds=2,
        physical_actions=1, initialization_physical_actions=1, safety_rejections=0,
        total_planning_s=22., total_gate_wall_s=8.,
        total_robot_execution_s_including_initialization=90.,
        human_fold_ready=None, episode_wall_s=130.,
        rounds=[{"timing": {"fast_planner_wall_s": 22., "gate_wall_s": 8.}}]) | extra


def fake_runner(records, calls, *, after=None):
    def run(command, log_path, session):
        index = len(calls)
        calls.append(command)
        folder = Path(command[command.index("--output")+1])
        folder.mkdir()
        record = records[index]
        if record is not None:
            m.save(folder/"episode_summary.json", record)
        if after:
            after(index, session)
        return 0
    return run


@pytest.mark.parametrize("stop", ["SAFETY_STOP", "ABSTAIN", "ROUND_LIMIT", "USER_STOP"])
def test_only_done_can_initialize_again(tmp_path, stop):
    args = arguments(tmp_path)
    calls = []
    code = m.run(args, episode_runner=fake_runner([result(), result(), result(stop)], calls))
    assert len(calls) == 3 and code != 0
    assert all("--continue-from" not in c for c in calls)
    assert all("--codex-recovery" in c for c in calls)
    assert len({c[c.index("--output")+1] for c in calls}) == 3
    assert len({c[c.index("--goal-image")+1] for c in calls}) == 1
    summary = json.loads((args.output/"batch_summary.json").read_text())
    assert summary["stop_reason"] == stop
    assert summary["model_done_episodes"] == 2
    assert summary["human_fold_ready_episodes"] == 0
    assert summary["initialization_physical_actions"] == 3


def test_explicit_cap_and_zero_action_done_not_claimed_success(tmp_path):
    args = arguments(tmp_path, episodes=2)
    calls = []
    assert m.run(args, episode_runner=fake_runner(
        [result(physical_actions=0), result()], calls)) == 0
    summary = json.loads((args.output/"batch_summary.json").read_text())
    assert len(calls) == 2 and summary["stop_reason"] == "EPISODE_LIMIT"
    assert summary["episodes"][0]["zero_action_done"] is True
    assert summary["episodes"][0]["human_fold_ready"] is None


def test_recovery_can_be_disabled(tmp_path):
    args = arguments(tmp_path, episodes=1, codex_recovery=False)
    calls = []
    assert m.run(args, episode_runner=fake_runner([result()], calls)) == 0
    assert "--codex-recovery" not in calls[0]


@pytest.mark.parametrize("record", [
    None, result(finished_at=None), result(cleanup_error="disconnect failed"),
    result(report_error="artifact failure"),
])
def test_incomplete_or_error_done_never_restarts(tmp_path, record):
    args = arguments(tmp_path)
    calls = []
    assert m.run(args, episode_runner=fake_runner([record], calls)) == 1
    assert len(calls) == 1
    assert json.loads((args.output/"batch_summary.json").read_text())["stop_reason"] == "EPISODE_ERROR"


def test_stop_at_episode_boundary_prevents_next_initialization(tmp_path):
    args = arguments(tmp_path)
    calls = []
    def stop(_index, session):
        session.request_stop(signal.SIGINT)
    assert m.run(args, episode_runner=fake_runner([result()], calls, after=stop)) == 130
    assert len(calls) == 1


def test_input_change_stops_before_next_episode(tmp_path):
    args = arguments(tmp_path)
    calls = []
    def alter(_index, session):
        (session.root/"goal/target.png").write_bytes(b"changed")
    assert m.run(args, episode_runner=fake_runner([result()], calls, after=alter)) == 1
    assert len(calls) == 1
    assert json.loads((args.output/"batch_summary.json").read_text())["stop_reason"] == "BATCH_INPUT_CHANGED"


def test_interrupt_is_forwarded_and_child_cleanup_finishes(tmp_path):
    """Real subprocess; fake robot cleanup only. No camera/controller or API."""
    session = m.Session(tmp_path)
    ready, cleaned = tmp_path/"ready", tmp_path/"cleaned"
    code = (
        "import signal,time,pathlib\n"
        "def stop(*_): raise KeyboardInterrupt()\n"
        "signal.signal(signal.SIGINT,stop)\n"
        f"pathlib.Path({str(ready)!r}).write_text('ready')\n"
        "try:\n"
        " while True: time.sleep(.05)\n"
        "except KeyboardInterrupt:\n"
        f" pathlib.Path({str(cleaned)!r}).write_text('STOP_AND_CLEANUP_FINISHED')\n"
    )
    def stop_when_ready():
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        session.request_stop(signal.SIGINT)
    timer = threading.Thread(target=stop_when_ready)
    timer.start()
    assert m.launch_episode([sys.executable, "-u", "-c", code],
                            tmp_path/"child.log", session) == 0
    timer.join()
    assert cleaned.read_text() == "STOP_AND_CLEANUP_FINISHED"
    assert session.child is None and session.cancel.is_set()


def test_no_execute_cannot_start(tmp_path):
    args = arguments(tmp_path, execute=False)
    with pytest.raises(ValueError, match="execute"):
        m.run(args, episode_runner=lambda *a: pytest.fail("hardware launch"))
    assert not args.output.exists()
