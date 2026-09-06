from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.watch_fold_exploration import (
    build_parser,
    run_watchdog,
    _snapshot_experience,
    _prepare_default_child_args,
    classify_child_exit,
)
from cloth_agent.fold_exploration_pipeline import build_parser as build_fold_parser


def test_watchdog_defaults_match_overnight_launcher() -> None:
    args = build_parser().parse_args([])
    assert args.project_root == Path(".")
    assert args.run_prefix == "fold_night"
    assert args.restart_delay_s == 10.0
    assert args.max_restarts == 0
    assert args.dry_run is False
    assert args.no_unattended is False
    assert args.no_viser is False


def test_fold_parser_exposes_uncapped_grounding_timeout() -> None:
    args = build_fold_parser().parse_args([])
    assert args.grounding_timeout_s is None
    args = build_fold_parser().parse_args(["--grounding-timeout-s", "3600"])
    assert args.grounding_timeout_s == 3600


def test_watchdog_restarts_preexecution_ik_failure() -> None:
    assert (
        classify_child_exit(
            1,
            {
                "status": "FAILED",
                "error": "SafetyError: controller IK rejected action 2 code=10",
            },
            {"stage": "run", "message": "pipeline finished"},
        )
        == "RESTART"
    )


def test_watchdog_injects_continuous_unattended_real_defaults() -> None:
    result = _prepare_default_child_args([])
    assert result == [
        "--real",
        "--confirm-real",
        "--unattended",
        "--max-iterations",
        "0",
        "--viser",
    ]


def test_watchdog_explicit_disable_flags_remove_defaults() -> None:
    result = _prepare_default_child_args(
        ["--real", "--confirm-real", "--unattended", "--viser", "--viser-port", "9000"],
        real=False,
        unattended=False,
        viser=False,
    )
    assert result == ["--max-iterations", "0"]


def test_watchdog_dry_run_keeps_continuous_unattended_viser_defaults() -> None:
    result = _prepare_default_child_args([], real=False)
    assert "--real" not in result
    assert "--confirm-real" not in result
    assert result[:2] == ["--unattended", "--max-iterations"]
    assert result[-1] == "--viser"


def test_watchdog_preserves_user_child_overrides() -> None:
    result = _prepare_default_child_args(
        ["--max-iterations", "5", "--viser-port", "9010", "--no-video"]
    )
    assert result.count("--max-iterations") == 1
    assert result[result.index("--max-iterations") + 1] == "5"
    assert result[result.index("--viser-port") + 1] == "9010"
    assert "--real" in result and "--confirm-real" in result


def test_watchdog_stops_after_execution_failure() -> None:
    assert (
        classify_child_exit(
            1,
            {"status": "FAILED", "error": "RobotExecutionError: motion failed"},
            {"stage": "run", "message": "pipeline finished"},
        )
        == "STOP"
    )
    assert (
        classify_child_exit(
            1,
            None,
            {"stage": "execution", "message": "trajectory started"},
        )
        == "STOP"
    )


def test_watchdog_stops_terminal_statuses() -> None:
    assert classify_child_exit(0, {"status": "COMPLETE"}, None) == "STOP"
    assert classify_child_exit(0, {"status": "MAX_ITERATIONS_REACHED"}, None) == "STOP"
    assert (
        classify_child_exit(
            0,
            {"status": "MAX_ITERATIONS_REACHED"},
            None,
            restart_on_max_iterations=True,
        )
        == "RESTART"
    )


def test_watchdog_snapshots_experience(tmp_path: Path) -> None:
    source = tmp_path / "old" / "workspace" / "fold_experience"
    source.mkdir(parents=True)
    (source / "experiences.jsonl").write_text('{"iteration": 1}\n', encoding="utf-8")
    (source / "experience_summary.json").write_text(
        json.dumps({"experience_count": 1}), encoding="utf-8"
    )
    destination = _snapshot_experience(source, tmp_path / "snapshot")
    assert destination is not None
    assert (destination / "experiences.jsonl").read_text(encoding="utf-8") == '{"iteration": 1}\n'
    assert json.loads((destination / "experience_summary.json").read_text())[
        "experience_count"
    ] == 1


def test_watchdog_passes_previous_experience_to_next_child(tmp_path: Path) -> None:
    root = tmp_path / "project"
    child_script = root / "scripts" / "claude_fold_exploration.py"
    child_script.parent.mkdir(parents=True)
    child_script.write_text(
        """
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--project-root')
p.add_argument('--run-id')
p.add_argument('--experience-dir')
p.add_argument('--unattended', action='store_true')
args, _ = p.parse_known_args()
root = Path(args.project_root)
count_path = root / 'child_count'
count = int(count_path.read_text()) if count_path.exists() else 0
count += 1
count_path.write_text(str(count))
run = root / 'runs' / args.run_id
exp = run / 'workspace' / 'fold_experience'
exp.mkdir(parents=True)
(exp / 'experiences.jsonl').write_text(json.dumps({'iteration': count}) + '\\n')
(exp / 'experience_summary.json').write_text(json.dumps({'experience_count': count}))
result = 'FAILED' if count == 1 else 'COMPLETE'
out = run / 'results' / 'fold_exploration' / 'attempt'
out.mkdir(parents=True)
(out / 'summary.json').write_text(json.dumps({'status': result, 'inherited': args.experience_dir}))
if count == 2:
    (root / 'inherited_value').write_text(args.experience_dir or '')
raise SystemExit(1 if result == 'FAILED' else 0)
""".strip(),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        project_root=root,
        python_bin=sys.executable,
        run_prefix="test_watch",
        restart_delay_s=0.0,
        max_restarts=1,
        restart_on_max_iterations=False,
        viser_port_base=8765,
        watchdog_dir=tmp_path / "watchdog",
        child_args=[],
    )
    assert run_watchdog(args) == 0
    inherited = Path((root / 'inherited_value').read_text())
    assert inherited.name == "fold_experience"
    assert (args.watchdog_dir / "watchdog_events.jsonl").is_file()
