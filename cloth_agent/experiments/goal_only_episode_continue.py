"""Explicitly authorized continuation with a shifted observation and unchanged checks."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from . import goal_only_full_episode_v1 as goal
from . import trajectory_memory_full_episode_v1 as shared


class ShiftedObservationRuntime(shared.EpisodeCollectionRuntime):
    """Use the existing motion planner for a 20 mm horizontal camera translation."""

    def move_to_perception(self, directory):
        super().move_to_perception(directory)
        reference = self.capture_current(Path(directory)/"unshifted_observation")
        initial = {k: np.asarray(v) for k, v in reference.meta["joints_deg"].items()}
        base = shared.matrix_pose(self.config.arms["left"].world_from_base @
                                 shared.pose_matrix(self.models["left"].forward(initial["left"])))
        if base[2] < 450:
            raise shared.DualArmError("observation shift requires TCP height >=450 mm")
        axis = np.asarray(reference.meta["world_from_camera_mm"])[:3, 0].copy()
        axis[2] = 0.
        axis /= np.linalg.norm(axis)
        target = base.copy()
        target[:3] += 20. * axis
        values = np.r_[target[:3]/1000., np.radians(target[3:])].tolist()
        source = f"motion.move_tcp(left={values})\n"
        folder = Path(directory)/"right20mm"
        folder.mkdir()
        (folder/"motion.py").write_text(source)
        operations = self.plan(source, initial, preserve_command_boundaries=True)
        final = initial
        for kind, data in operations:
            if kind != "move":
                raise shared.DualArmError("observation shift must not command the gripper")
            final = data.end
        self.arrival(initial)
        started = time.monotonic()
        result = self._execute_operations(operations, folder)
        shared.save(folder/"execution.json", result)
        if result["status"] != "EXECUTED":
            raise shared.DualArmError("observation shift failed: "+str(result))
        self.arrival(final)
        shared.save(folder/"shift.json", dict(
            kind="observation_only", cloth_manipulation=False, offset_mm=20.,
            world_direction=axis.tolist(), target_world_tcp_mm_deg=target.tolist(),
            wall_s=time.monotonic()-started))


def freeze_continuation(root, source, _scope):
    source = source.resolve()
    previous = shared.read(source/"episode_summary.json")
    if previous["condition"] != "GOAL_ONLY_NO_TRAJECTORY_MEMORY":
        raise ValueError("continuation must use the same goal-only episode")
    if previous["termination_reason"] != "SAFETY_STOP":
        raise ValueError("expected the explicitly stopped source episode")
    manifest = goal.prepare_goal(root, source/"goal/target.png", [])
    history = []
    hashes = {}
    for item in previous["rounds"]:
        if item["execution"]["status"] != "EXECUTED":
            continue
        src = source/item["round"]
        if not item["execution"]["physical_execution"] or not (src/"after/rgb.png").is_file():
            raise ValueError("incomplete executed prefix")
        name = "prefix_"+item["round"]
        dest = root/"current_episode_prefix"/name
        for directory in ("before", "after"):
            shutil.copytree(src/directory, dest/directory)
        for filename in ("plan.json", "execution.json", "timing.json"):
            shutil.copyfile(src/filename, dest/filename)
        for path in dest.rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(root))] = shared.digest(path)
        history.append(dict(round=name, acquisition=item["acquisition"]))
    shared.save(root/"continuation.json", dict(
        source_episode=str(source), source_summary_sha256=shared.digest(source/"episode_summary.json"),
        source_termination=previous["termination_reason"],
        prior_total_rounds=previous["total_rounds"], prior_physical_actions=previous["physical_actions"],
        prior_episode_wall_s=previous["episode_wall_s"], history=history, files=hashes,
        skipped_rejected_rounds=[item["round"] for item in previous["rounds"]
                                 if item["execution"]["status"] != "EXECUTED"],
        authorization="User explicitly requested continuation after the shifted-view diagnostic.",
        observation_policy="Existing perception pose followed by image-right horizontal 20 mm; no reset.",
        prompt_changed=False, grasp_depth_threshold_changed=False))
    return manifest


def load_history(root):
    return [dict(round=item["round"], directory=root/"current_episode_prefix"/item["round"],
                 plan=shared.read(root/"current_episode_prefix"/item["round"]/"plan.json"),
                 before=root/"current_episode_prefix"/item["round"]/"before/rgb.png",
                 acquisition=item["acquisition"])
            for item in shared.read(root/"continuation.json")["history"]]


def verify(root):
    shared.verify_memory(root)
    for name, expected in shared.read(root/"continuation.json")["files"].items():
        if shared.digest(root/name) != expected:
            raise shared.DualArmError("current episode prefix changed: "+name)


def render(root, summary):
    continuation = shared.read(root/"continuation.json")
    summary.update(
        continuation_of=continuation["source_episode"], initialization_kind="CONTINUATION_NO_RESET",
        prior_total_rounds=continuation["prior_total_rounds"],
        cumulative_total_rounds=continuation["prior_total_rounds"]+summary["total_rounds"],
        cumulative_physical_actions=continuation["prior_physical_actions"]+summary["physical_actions"],
        active_episode_segments_wall_s=continuation["prior_episode_wall_s"]+summary["episode_wall_s"],
        observation_policy=continuation["observation_policy"])
    shifts = [shared.read(path) for path in root.glob("**/right20mm/shift.json")]
    summary["observation_shift_count"] = len(shifts)
    summary["observation_shift_execution_s"] = sum(item["wall_s"] for item in shifts)
    goal.render(root, summary)
    with (root/"report.md").open("a") as f:
        f.write("\n## 用户授权续跑\n\n")
        f.write(f"承接 {continuation['source_episode']}，原 SAFETY_STOP 保留；不重新初始化。"
                "目标图与 Agent prompt 保持不变，保留此前实际执行的两步 current-episode history。"
                "被拒绝的动作未作为已执行经验提供。\n\n"
                "每次到原观测位姿后，通过原 planner 小幅向图像右方平移 20 mm，再采集新的 RGB-D。"
                "新动作只使用新图选点；所有原检查和 8 mm 深度阈值不变。"
                "再次检查失败仍立即停止。\n\n"
                f"本段 {summary['total_rounds']} rounds、{summary['physical_actions']} 个布料动作；"
                f"累计 {summary['cumulative_total_rounds']} rounds、"
                f"{summary['cumulative_physical_actions']} 个布料动作。"
                "观测位移单独计数，不计作布料动作。两段总运行时间不包含暂停等待和独立诊断时间。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/dual_arm.local.json"))
    parser.add_argument("--timeout-s", type=float, default=1200.)
    args = parser.parse_args()
    previous = shared.read(args.source/"episode_summary.json")
    args.max_rounds = int(previous["max_rounds"])-int(previous["total_rounds"])
    if args.max_rounds < 1:
        parser.error("no rounds remain in original episode budget")
    if (args.source/"prompt.txt").read_text() != goal.PROMPT:
        parser.error("source prompt differs; do not silently change it")
    args.skip_initialization = True
    args.data_root, args.scope_dates = args.source, []
    return shared.run(args, experiment_name="goal_only_episode_continue", prompt=goal.PROMPT,
                      prepare_inputs=freeze_continuation, verify_inputs=verify, plan_round=goal.plan_round,
                      render_report=render, source_file=__file__,
                      runtime_factory=ShiftedObservationRuntime, initial_history=load_history)


if __name__ == "__main__":
    raise SystemExit(main())
