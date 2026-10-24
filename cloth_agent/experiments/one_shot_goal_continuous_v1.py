"""Supervise independent one-shot episodes; only clean DONE permits another reset."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOAL = ROOT / "reports/goal_only_full_episode_v1_restart02/goal/target.png"
SOURCE_FILES = [
    "cloth_agent/experiments/one_shot_goal_continuous_v1.py",
    "cloth_agent/experiments/one_shot_goal_full_episode_v1.py",
    "cloth_agent/experiments/one_shot_goal_planner_probe_v1.py",
    "cloth_agent/experiments/trajectory_memory_full_episode_v1.py",
    "cloth_agent/experiments/goal_only_full_episode_v1.py",
    "cloth_agent/experiments/trajectory_memory_agent_probe.py",
    "cloth_agent/pre_action_gate.py", "cloth_agent/unfolding_planners.py",
    "cloth_agent/table_target_depth.py", "cloth_agent/gamepad_collect_runtime.py",
    "cloth_agent/codex_recovery.py",
    "cloth_agent/gamepad_collect.py", "cloth_agent/random_fold_reset.py",
    "cloth_agent/dual_arm/observation.py", "cloth_agent/codex_native_runtime.py",
    "config/perception.free_exploration.json",
    "config/calibration/dual_arm_working_20261008/camA_intrinsics.yaml",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Session:
    def __init__(self, root):
        self.root = root
        self.cancel = threading.Event()
        self.child = None
        self.stop_signal = None
        self.forwarded = False

    def request_stop(self, signum=signal.SIGINT, _frame=None):
        self.stop_signal = int(signum)
        self.cancel.set()
        self.forward_stop()

    def forward_stop(self):
        if self.child is not None and self.child.poll() is None and not self.forwarded:
            try:
                # The child owns robot stop/cleanup; never terminate it with SIGKILL.
                os.killpg(self.child.pid, self.stop_signal or signal.SIGINT)
                self.forwarded = True
            except ProcessLookupError:
                pass

    def event(self, name, **fields):
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(at=utc(), event=name, **fields),
                                    ensure_ascii=False) + "\n")


def launch_episode(command, log_path, session):
    """Wait for the foreground episode to finish cleanup before any next launch."""
    if session.cancel.is_set():
        return None
    with log_path.open("x", buffering=1) as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 start_new_session=True)
        session.child, session.forwarded = child, False
        session.event("CHILD_STARTED", pid=child.pid, command=command)
        if session.cancel.is_set():
            session.forward_stop()

        def copy_output():
            for line in child.stdout:
                log.write(line)
                try:
                    print(line, end="", flush=True)
                except (BrokenPipeError, OSError):
                    # Logging and robot cleanup must continue if the terminal closes.
                    pass

        thread = threading.Thread(target=copy_output, daemon=True)
        thread.start()
        try:
            while child.poll() is None:
                if session.cancel.is_set():
                    session.forward_stop()
                try:
                    child.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            session.request_stop(signal.SIGTERM)
            child.wait()
            raise
        finally:
            thread.join()
            child.stdout.close()
            session.child = None
        return child.returncode


def _stats(values):
    if not values:
        return dict(n=0, median=None, mean=None, p95=None)
    import numpy as np
    return dict(n=len(values), median=float(np.median(values)),
                mean=float(np.mean(values)), p95=float(np.percentile(values, 95)))


def checkpoint(root, summary, started):
    rows = summary["episodes"]
    summary.update(
        wall_s=time.monotonic()-started,
        episodes_recorded=len(rows),
        model_done_episodes=sum(r["termination_reason"] == "DONE" for r in rows),
        total_rounds=sum(r["rounds"] for r in rows),
        physical_manipulation_dispatches=sum(r["physical_actions"] for r in rows),
        initialization_physical_actions=sum(r["initialization_physical_actions"] for r in rows),
        safety_rejections=sum(r["safety_rejections"] for r in rows),
        total_fast_planning_s=sum(r["fast_planning_s"] for r in rows),
        total_gate_s=sum(r["gate_s"] for r in rows),
        total_codex_recovery_s=sum(r["codex_s"] for r in rows),
        codex_recovery_calls=sum(r["codex_recovery_calls"] for r in rows),
        total_robot_execution_s_including_initialization=sum(r["robot_s"] for r in rows),
        human_fold_ready_episodes=sum(r["human_fold_ready"] is True for r in rows),
        fast_planning_latency_s=_stats([v for r in rows for v in r["fast_latencies_s"]]),
        gate_latency_s=_stats([v for r in rows for v in r["gate_latencies_s"]]))
    save(root/"batch_summary.json", summary)
    fields = ["episode", "termination_reason", "rounds", "physical_actions",
              "initialization_physical_actions", "safety_rejections", "fast_planning_s",
              "gate_s", "codex_s", "codex_recovery_calls", "robot_s", "wall_s", "human_fold_ready", "returncode"]
    with (root/"episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Continuous one-shot collection", "",
        f"Status: {summary['status']}; stop reason: {summary.get('stop_reason')}", "",
        f"Recorded episodes: {len(rows)}; model DONE: {summary['model_done_episodes']}; "
        f"unfold manipulation dispatches: {summary['physical_manipulation_dispatches']}; "
        f"initialization actions: {summary['initialization_physical_actions']}.", "",
        "Only a clean model DONE starts another independent initialization. DONE is not "
        "human-confirmed FOLD_READY. Initialization execution is not confirmed cloth acquisition. "
        "Zero-action DONE episodes are retained, not counted as verified successes. "
        "Physical-action counts include dispatches that may have partially executed.", "",
        "Allowlisted pre-dispatch errors may invoke bounded Codex recovery inside the episode. "
        "ABSTAIN / ROUND_LIMIT / unrecoverable errors or exhausted recovery / operator stop end the batch. "
        "No hardware retry or failure reset. Review exports remain separate from execution records.", "",
        "|Episode|Stop|Rounds|Unfold actions|Init actions|Human FOLD_READY|Review|",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        name = row["episode"]
        review = f"[review]({name}/review.html)" if (root/name/"review.html").exists() else "unavailable"
        lines.append(f"|[{name}]({name}/report.md)|{row['termination_reason']}|{row['rounds']}|"
                     f"{row['physical_actions']}|{row['initialization_physical_actions']}|"
                     f"{row['human_fold_ready'] if row['human_fold_ready'] is not None else 'PENDING'}|{review}|")
    (root/"report.md").write_text("\n".join(lines)+"\n")


def _episode_row(name, record, returncode):
    rounds = record.get("rounds", [])
    return dict(episode=name, termination_reason=record.get("termination_reason"),
        detail=record.get("termination_detail", ""), returncode=returncode,
        rounds=record.get("total_rounds", 0), physical_actions=record.get("physical_actions", 0),
        initialization_physical_actions=record.get("initialization_physical_actions", 0),
        safety_rejections=record.get("safety_rejections", 0),
        fast_planning_s=record.get("total_planning_s", 0.),
        gate_s=record.get("total_gate_wall_s", 0.),
        codex_s=record.get("total_deliberative_planning_s", 0.),
        codex_recovery_calls=record.get("codex_recovery_calls", 0),
        robot_s=record.get("total_robot_execution_s_including_initialization", 0.),
        wall_s=record.get("episode_wall_s", 0.), human_fold_ready=record.get("human_fold_ready"),
        zero_action_done=record.get("termination_reason") == "DONE" and record.get("physical_actions", 0) == 0,
        fast_latencies_s=[r["timing"]["fast_planner_wall_s"] for r in rounds
                          if r.get("timing", {}).get("fast_planner_wall_s", 0) > 0],
        gate_latencies_s=[r["timing"]["gate_wall_s"] for r in rounds
                          if r.get("timing", {}).get("gate_wall_s", 0) > 0],
        report_error=record.get("report_error"), cleanup_error=record.get("cleanup_error"))


def run(args, *, episode_runner=launch_episode):
    if not args.execute:
        raise ValueError("--execute is required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/"goal").mkdir()
    goal = root/"goal/target.png"
    shutil.copyfile(args.goal_image, goal)
    config = args.config.resolve()
    sources = {str(ROOT/p): digest(ROOT/p) for p in SOURCE_FILES}
    sources[str(config)] = digest(config)
    sources[str(goal)] = digest(goal)
    save(root/"protocol.json", dict(
        experiment="one_shot_goal_continuous_v1", created_at=utc(),
        episodes_limit=args.episodes, zero_limit_means="until operator stop or a non-DONE termination",
        max_rounds_per_episode=args.max_rounds, between_episodes_s=args.between_episodes_s,
        next_episode_only_after="clean model DONE", model_done_is_human_fold_ready=False,
        initializations="one existing center pickup/drop per episode; Home Y/yaw retained",
        fast_planner="unchanged gpt-6-astra/high one-shot", gate="unchanged SHADOW",
        history_between_episodes=False, hardware_retries=False,
        codex_recovery=dict(enabled=getattr(args, "codex_recovery", True),
            max_calls_per_episode=getattr(args, "max_codex_replans", 4),
            max_calls_per_incident=getattr(args, "max_replans_per_incident", 2)),
        source_sha256=sources))
    session = Session(root)
    summary = dict(status="RUNNING", started_at=utc(), stop_reason=None,
                   active_episode=None, episodes=[])
    started = time.monotonic()
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in handlers:
        signal.signal(s, session.request_stop)
    checkpoint(root, summary, started)
    try:
        index = 1
        while not session.cancel.is_set() and (args.episodes == 0 or index <= args.episodes):
            if any(digest(p) != value for p, value in sources.items()):
                summary["stop_reason"] = "BATCH_INPUT_CHANGED"
                break
            name = f"episode_{index:04d}"
            folder = root/name
            command = [
                sys.executable, "-u", "-m", "cloth_agent.experiments.one_shot_goal_full_episode_v1",
                "--execute", "--output", str(folder), "--goal-image", str(goal),
                "--config", str(config), "--max-rounds", str(args.max_rounds),
                "--timeout-s", str(args.timeout_s), "--gate-timeout-s", str(args.gate_timeout_s),
            ]
            if args.review_initialization:
                command.append("--review-initialization")
            if getattr(args, "codex_recovery", True):
                command += ["--codex-recovery", "--max-codex-replans", str(getattr(args, "max_codex_replans", 4)),
                    "--max-replans-per-incident", str(getattr(args, "max_replans_per_incident", 2)),
                    "--codex-timeout-s", str(getattr(args, "codex_timeout_s", 600.))]
            summary["active_episode"] = name
            checkpoint(root, summary, started)
            session.event("EPISODE_START", episode=name)
            print(f"\nCONTINUOUS {name}: initialize → unfold → save; Ctrl+C stops the batch", flush=True)
            returncode = episode_runner(command, root/(name+".run.log"), session)
            path = folder/"episode_summary.json"
            if path.exists():
                record = json.loads(path.read_text())
                summary["episodes"].append(_episode_row(name, record, returncode))
                save(root/(name+".completion.json"), dict(
                    summary_sha256=digest(path), returncode=returncode, recorded_at=utc()))
            else:
                record = {}
            summary["active_episode"] = None
            checkpoint(root, summary, started)
            reason = record.get("termination_reason")
            session.event("EPISODE_END", episode=name, termination_reason=reason, returncode=returncode)
            if session.cancel.is_set() or reason == "USER_STOP":
                summary["stop_reason"] = "USER_STOP"
                break
            # A zero subprocess exit code is insufficient: ABSTAIN and ROUND_LIMIT
            # also return zero in the existing runner.
            if (returncode != 0 or reason != "DONE" or not record.get("finished_at")
                    or record.get("cleanup_error") or record.get("report_error")):
                summary["stop_reason"] = reason if reason and reason != "DONE" else "EPISODE_ERROR"
                break
            if args.episodes and index >= args.episodes:
                summary["stop_reason"] = "EPISODE_LIMIT"
                break
            session.event("WAITING_BEFORE_RESET", seconds=args.between_episodes_s)
            print(f"Model DONE saved. Next initialization in {args.between_episodes_s:g}s; Ctrl+C cancels.", flush=True)
            if session.cancel.wait(args.between_episodes_s):
                summary["stop_reason"] = "USER_STOP"
                break
            index += 1
        if session.cancel.is_set():
            summary["stop_reason"] = "USER_STOP"
    except BaseException as exc:
        session.request_stop(signal.SIGTERM)
        summary.update(stop_reason="SUPERVISOR_ERROR", error=f"{type(exc).__name__}: {exc}")
        (root/"exception.txt").write_text(traceback.format_exc())
    finally:
        summary.update(status="STOPPED", finished_at=utc(), stop_signal=session.stop_signal)
        checkpoint(root, summary, started)
        session.event("BATCH_STOP", reason=summary["stop_reason"])
        for s, handler in handlers.items():
            signal.signal(s, handler)
    print(f"CONTINUOUS FINISHED: {summary['stop_reason']}; {root}", flush=True)
    return 130 if summary["stop_reason"] == "USER_STOP" else (
        0 if summary["stop_reason"] == "EPISODE_LIMIT" else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-image", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--config", type=Path, default=ROOT/"config/dual_arm.local.json")
    parser.add_argument("--episodes", type=int, default=0, help="0: continue until operator stop or non-DONE; positive: cap")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--between-episodes-s", type=float, default=5.)
    parser.add_argument("--timeout-s", type=float, default=600.)
    parser.add_argument("--gate-timeout-s", type=float, default=120.)
    parser.add_argument("--review-initialization", action="store_true")
    parser.add_argument("--codex-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-codex-replans", type=int, default=4)
    parser.add_argument("--max-replans-per-incident", type=int, default=2)
    parser.add_argument("--codex-timeout-s", type=float, default=600.)
    args = parser.parse_args(argv)
    if (args.episodes < 0 or not 1 <= args.max_rounds <= 50
            or not 0 <= args.between_episodes_s <= 3600
            or not 0 < args.timeout_s <= 1200 or not 0 < args.gate_timeout_s <= 1200):
        parser.error("invalid limits")
    if not 1 <= args.max_codex_replans <= 8 or not 1 <= args.max_replans_per_incident <= 4 or not 0 < args.codex_timeout_s <= 1200:
        parser.error("invalid recovery limits")
    if not args.goal_image.is_file() or not args.config.is_file() or args.output.exists():
        parser.error("goal/config missing or output already exists")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
