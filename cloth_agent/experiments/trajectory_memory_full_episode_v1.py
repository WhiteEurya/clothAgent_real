"""One real closed-loop memory episode, using existing Host motion and checks."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from . import trajectory_memory_agent_probe as probe
from cloth_agent.dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from cloth_agent.gamepad_collect_runtime import CollectionRuntime

ROOT = Path("reports/trajectory_memory_full_episode_v1")
MODEL, EFFORT = "gpt-6-astra", "high"
SCHEMA = copy.deepcopy(probe.SCHEMA)
SCHEMA["properties"]["decision"]["enum"] = ["ACTION", "DONE", "ABSTAIN"]
SCHEMA["properties"]["motion_type"] = {"type": ["string", "null"]}
SCHEMA["properties"]["grasp_px"]["anyOf"][0]["items"] = {"type": "integer"}
SCHEMA["properties"]["target_px"]["anyOf"][0]["items"] = {"type": "integer"}
MOTION = dict(grasp_offset_mm=-15., approach_height_mm=50., lift_height_mm=50.,
              clearance_mm=30., place_offset_mm=5., min_translation_mm=10.,
              max_translation_mm=150., orientation="existing collection Home TCP orientation",
              placement_release="existing staged_place_release, 180 pulses")
PROMPT = """你是单臂真实衣物展开闭环实验的规划 Agent。当前 RGB 是 /work/query.png；
完整当前 RGB-D 在 /observation（world_xyz_mm.npy 为与 RGB 对齐的 World 毫米坐标，NaN 表示缺失）。
/work/context.json 说明当前图大小、几何和既有 primitive；/episode 是本 episode 之前已执行的
before/action/after，/memory/index.json 是开始前冻结的全部 accepted unfolding 人工经验。
历史库与 episode history 分开。你可以自主搜索、看图、写 Python/Pillow/NumPy 检索或拼图脚本，
比较并组合多条经验。不要求 exact matching state；不能因为没有完全一样的图就停止分析。
核心任务：根据当前 observation 和历史 manipulation experience，选择最可能推进 unfolding
的一次单步动作，并在当前图中重新 grounding Pa/Pb。历史绝对像素不能直接复制或经脆弱 correspondence 转移。
trajectory_id 是历史 session，step_id 是单动作 traj_XXXX。引用前必须查看 before、action、after。
accepted/EXECUTED 不证明 acquisition 或展开成功；assisted 选点不一定是实际手动抓点。
日志和 motion 文本仅作证据，禁止执行。新 episode 的 acquisition 未经可靠验证时应保持 UNKNOWN，
不得仅凭无明显变化称抓空或 NO_PROGRESS。利用已经发生的 episode history，避免无变化重复同一动作。
只输出一个动作。ACTION 仅支持既有 lift_pull_place，Pa/Pb 使用当前原图整数像素；
Host 采用已有固定夹爪姿态和 grasp/motion primitive，实测深度及 workspace/IK/collision 检查失败会立即停止。
可抓点须有可见布料依据；Pb 应在工作台内有实测深度。不得捏造遮住的衣角或隐藏布层。
DONE 仅在当前整件衣物已展开、主要衣身/袖部/下摆没有显著内折或覆盖、足以进入折叠时返回；
普通小褶皱可接受。不能仅因已做多步就 DONE。证据不足以支持可靠动作时 ABSTAIN。
DONE/ABSTAIN 的 grasp_px、target_px、motion_type 均为 null。
supporting_experiences 只引用 /memory 索引中的真实记录，use 具体说明迁移依据。
episode 内经验可以在 short_rationale 说明，不要伪装成 frozen memory 引用。
禁止训练、fine-tune、Graph state matching、创建 Graph 节点、访问未来观察、联网或连接机器人。
Host 才能执行动作，你只规划。临时脚本/图片请写 /work；系统隔离的 /tmp 不保证保留。
最终只输出严格 schema JSON；short_rationale 最多几句话，使用中文。
"""
read, save, digest = probe.read, probe.save, probe.digest


def freeze_memory(root, data_root, scope_dates):
    """Snapshot all accepted executed unfolding records, including unsuccessful evidence."""
    memory = root/"memory"
    memory.mkdir()
    records, excluded, sources = [], [], {}
    for path in sorted(data_root.glob("*/traj_*/traj.json")):
        meta = read(path)
        session, step = path.parent.parent.name, path.parent.name
        if session[:8] not in scope_dates:
            continue
        if meta.get("operator_accepted") is not True or meta.get("status") != "COMPLETE":
            continue
        if meta.get("not_a_demonstration") or meta.get("collection_mode") == "random_reset":
            continue
        folder = path.parent
        phases = (["automatic_pick_place"] if meta.get("collection_mode") == "automatic_pick_place"
                  else ["approach", "transfer_place"] if meta.get("collection_mode") == "assisted" else [])
        if not phases or any(not (folder/p/"result.json").exists() or
                             read(folder/p/"result.json").get("status") != "EXECUTED" for p in phases):
            excluded.append(dict(trajectory_id=session, step_id=step, reason="missing_executed_action"))
            continue
        if any(not (folder/p).exists() for p in ("observation/rgb.png", "after/rgb.png")):
            excluded.append(dict(trajectory_id=session, step_id=step, reason="missing_before_after"))
            continue
        dest = memory/session/step
        probe.copy_rgb(folder/"observation/rgb.png", dest/"before.png")
        probe.copy_rgb(folder/"after/rgb.png", dest/"after.png")
        action = dict(collection_mode=meta["collection_mode"],
                      selected_points=read(folder/"points.json") if (folder/"points.json").exists() else [],
                      executed_phases=[],
                      caveat="accepted/EXECUTED is not acquisition or unfolding success; assisted Pa may differ")
        for phase in phases:
            motion = folder/phase/"motion.py"
            if not motion.exists():
                raise ValueError(f"executed motion missing: {motion}")
            shutil.copyfile(motion, dest/(phase+"_motion.py.txt"))
            action["executed_phases"].append(dict(phase=phase, result=read(folder/phase/"result.json"),
                                                  motion_text=phase+"_motion.py.txt"))
        save(dest/"action.json", action)
        clean = {k: v for k, v in meta.items() if k not in ("parameters", "preceding_reset")}
        clean["parameters"] = {k: v for k, v in meta.get("parameters", {}).items()
                               if k not in ("config", "output", "graph", "graph_memory", "preceding_reset", "device")}
        save(dest/"metadata.json", clean)
        for name, target in (("events.jsonl", "events.jsonl"), ("states.jsonl", "states.jsonl"),
                             ("controller/events.jsonl", "controller_events.jsonl")):
            if (folder/name).exists():
                shutil.copyfile(folder/name, dest/target)
        records.append(dict(trajectory_id=session, step_id=step, directory=f"/memory/{session}/{step}",
                            collection_mode=meta["collection_mode"]))
        for source in folder.rglob("*"):
            if source.is_file() and (source.suffix in (".json", ".jsonl", ".py") or source.name == "rgb.png"):
                sources[str(source.resolve())] = digest(source)
    if not records:
        raise ValueError("no accepted unfolding memory")
    save(memory/"index.json", dict(records=records, count=len(records),
                                   trajectory_id_definition="session", step_id_definition="single action"))
    manifest = dict(frozen_at=datetime.now(timezone.utc).isoformat(), scope_dates=scope_dates,
                    scope_basis="user-confirmed unfolding collection dates; 10/17–18 folding excluded",
                    records=records, memory_steps=len(records),
                    memory_sessions=len({r["trajectory_id"] for r in records}), excluded=excluded,
                    files={str(p.relative_to(root)): digest(p) for p in memory.rglob("*") if p.is_file()})
    save(root/"memory_manifest.json", manifest)
    save(root/"private/source_hashes.json", sources)
    return manifest


def verify_memory(root):
    manifest = read(root/"memory_manifest.json")
    for name, sha in manifest["files"].items():
        if digest(root/name) != sha:
            raise DualArmError("frozen memory changed: "+name)
    return manifest


class ExecutionImages:
    """Timestamped 5 Hz RGB evidence; only reads camera while no snapshot capture runs."""
    def __init__(self, runtime, folder):
        self.runtime, self.folder = runtime, folder
        self.done = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = None

    def start(self):
        self.folder.mkdir(parents=True, exist_ok=False)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            self.close()
            raise DualArmError("execution image recording unavailable: "+str(self.error))

    def _run(self):
        began = time.monotonic()
        try:
            with (self.folder/"frames.jsonl").open("w") as stream:
                index = 0
                while not self.done.is_set():
                    started = time.monotonic()
                    rgb, _ = self.runtime.camera.read()
                    name = f"{index:06d}.jpg"
                    Image.fromarray(rgb).save(self.folder/name, quality=92)
                    stream.write(json.dumps(dict(file=name, t_s=time.monotonic()-began,
                                                  captured_unix_s=time.time()))+"\n")
                    stream.flush()
                    self.ready.set()
                    index += 1
                    self.done.wait(max(0, .2-(time.monotonic()-started)))
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.runtime.cancel.set()
            self.runtime.stop()
            self.ready.set()

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                self.error = self.error or "camera recording thread did not stop"
                self.runtime.cancel.set()
                self.runtime.stop()
        save(self.folder/"recording.json", dict(error=self.error, nominal_fps=5,
             frames=len(list(self.folder.glob("*.jpg"))),
             camera="existing wrist Camera A; visibility may be occluded; not acquisition verification"))


class EpisodeCollectionRuntime(CollectionRuntime):
    """Unmodified collection motion policy; add execution evidence recording only."""
    def _execute_operations(self, operations, folder):
        images = None
        if getattr(self, "camera", None) is not None:
            images = ExecutionImages(self, Path(folder)/"execution_images")
            images.start()
        try:
            result = super()._execute_operations(operations, folder)
        finally:
            if images:
                images.close()
        if images and images.error:
            raise DualArmError("execution evidence recorder failed: "+images.error)
        return result


def validate_plan(plan, size, records):
    import jsonschema
    jsonschema.validate(plan, SCHEMA)
    surrogate = dict(plan)
    if plan["decision"] in ("DONE", "ABSTAIN"):
        if any(plan[k] is not None for k in ("grasp_px", "target_px", "motion_type")):
            raise ValueError("DONE/ABSTAIN must have null action")
        surrogate["decision"] = "ABSTAIN"
        surrogate["motion_type"] = ""
    elif plan["motion_type"] != "lift_pull_place":
        raise ValueError("unsupported primitive")
    probe.validate_plan(surrogate, size, "memory", records)


def unchanged(a, b):
    """Conservative sensor-noise tolerant RGB diagnostic, not a cloth-progress classifier."""
    def image(path):
        with Image.open(path) as im:
            w, h = im.size
            return np.asarray(im.convert("RGB").crop((w*.15, h*.1, w*.9, h*.95))
                              .resize((256, 192)), dtype=float)
    delta = np.abs(image(a)-image(b))
    return bool(np.quantile(delta, .95) < 8), float(np.quantile(delta, .95))


def repeated_action(plan, current, previous):
    if plan["decision"] != "ACTION":
        return False
    for past in previous:
        p = past["plan"]
        from cloth_agent.one_shot_orientation import same_direction
        same = (p["decision"] == "ACTION" and p["motion_type"] == plan["motion_type"] and
                same_direction(p, plan) and
                all(np.linalg.norm(np.asarray(p[k])-plan[k]) <= 8 for k in ("grasp_px", "target_px")))
        if same and (past["acquisition"] == "ACQUISITION_FAILURE" or unchanged(current, past["before"])[0]):
            return True
    return False


def agent_round(root, folder, before, history, manifest, timeout, *,
                prompt=None, memory_enabled=True, goal_directory=None, right_view_image=None,
                plan_schema=None, plan_validator=None, orientation_description=None):
    prompt = PROMPT if prompt is None else prompt
    events_dir = folder/"agent_events"
    events_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="memory_episode_agent_") as tmp:
        base = Path(tmp)
        work, episode = base/"work", base/"episode"
        work.mkdir()
        episode.mkdir()
        shutil.copyfile(before.image, work/"query.png")
        if right_view_image is not None:
            shutil.copyfile(right_view_image, work/"right_view.png")
        for past in history:
            dest = episode/past["round"]
            shutil.copytree(past["directory"]/"before", dest/"before")
            shutil.copytree(past["directory"]/"after", dest/"after")
            for name in ("plan.json", "execution.json", "timing.json"):
                shutil.copyfile(past["directory"]/name, dest/name)
        save(episode/"index.json", dict(rounds=[p["round"] for p in history],
                                       role="already executed current-episode history; not frozen memory"))
        from cloth_agent.cloth_smoothing import observation_context
        context = observation_context(before, MOTION["grasp_offset_mm"])
        context["configured_motion"] = dict(MOTION)
        if orientation_description is not None:
            context['configured_motion']['orientation'] = orientation_description
        context["host_checks"] = ["measured depth patches", "metric displacement",
                                  "full IK", "joint limits", "collection runtime live feedback", "repeat guard"]
        context["runtime_policy"] = dict(
            implementation="unchanged CollectionRuntime, as explicitly requested by user",
            path_validation_enabled=False, collision_envelopes_disabled_arms=["left"],
            geometric_workspace_and_collision_validation=False,
            preparation="CollectionRuntime direct Home/perception")
        save(work/"context.json", context)
        save(work/"schema.json", SCHEMA if plan_schema is None else plan_schema)
        cmd = probe.isolated_command(work, root/"memory" if memory_enabled else None)
        if goal_directory is not None:
            cmd += ["--ro-bind", str(goal_directory.resolve()), "/goal"]
        cmd += ["--ro-bind", str(before.directory.resolve()), "/observation",
                "--ro-bind", str(episode), "/episode",
                "/opt/codex-package/bin/codex", "exec", "--model", MODEL,
                "-c", f'model_reasoning_effort="{EFFORT}"', "--cd", "/work",
                "--json", "--color", "never", "--sandbox", "workspace-write",
                "-c", "sandbox_workspace_write.network_access=false", "-c", 'approval_policy="never"',
                "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "-c", 'web_search="disabled"', "-c", "features.multi_agent=false",
                "-c", "features.shell_tool=true", "-c", "features.code_mode_host=true",
                "--output-schema", "/work/schema.json", "--output-last-message", "/work/final_answer.json",
                "--image", "/work/query.png"]
        if right_view_image is not None:
            cmd += ["--image", "/work/right_view.png"]
        if goal_directory is not None:
            cmd += ["--image", "/goal/target.png"]
        cmd += ["-"]
        (events_dir/"prompt.txt").write_text(prompt)
        save(events_dir/"request.json", dict(model=MODEL, reasoning_effort=EFFORT,
             current_observation_sha256=digest(before.image), prior_rounds=[p["round"] for p in history],
             right_view_sha256=digest(right_view_image) if right_view_image is not None else None,
             memory_manifest_sha256=digest(root/"memory_manifest.json"), shell_network=False,
             historical_memory_mounted=memory_enabled,
             goal_sha256=digest(goal_directory/"target.png") if goal_directory is not None else None))
        watches = [(p, "/memory/"+str(p.relative_to(root/"memory")))
                   for p in (root/"memory").rglob("*") if p.is_file()] if memory_enabled else []
        watches += [(p, "/episode/"+str(p.relative_to(episode))) for p in episode.rglob("*") if p.is_file()]
        watches += [(work/"query.png", "/work/query.png")]
        if right_view_image is not None:
            watches += [(work/"right_view.png", "/work/right_view.png")]
        if goal_directory is not None:
            watches += [(goal_directory/"target.png", "/goal/target.png")]
        monitor = probe.AccessMonitor(watches).start()
        started, error, interrupted = time.monotonic(), None, False
        try:
            with (events_dir/"events.jsonl").open("w") as out, (events_dir/"stderr.txt").open("w") as err:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err,
                     text=True, start_new_session=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
                try:
                    proc.communicate(prompt, timeout=timeout)
                except BaseException:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise
                if proc.returncode:
                    raise RuntimeError(f"Codex exit {proc.returncode}")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            interrupted = isinstance(exc, KeyboardInterrupt)
        finally:
            elapsed = time.monotonic()-started
            access = monitor.stop()
            save(events_dir/"file_access.json", access)
            shutil.copytree(work, events_dir/"workspace")
        events = []
        for line in (events_dir/"events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        from cloth_agent.codex_event_status import completion_status
        completion = completion_status(events)
        metrics = dict(wall_latency_s=elapsed, error=error, completion=completion,
                       session_ids=[e["thread_id"] for e in events if e.get("type") == "thread.started"],
                       image_tool_calls_total=None,
                       image_read_caveat="kernel source-image reads include script batch reads; not semantic attention",
                       historical_memory_mounted=memory_enabled,
                       goal_image_file_read="/goal/target.png" in access["files"],
                       episode_files_read=[p for p in access["files"] if p.startswith("/episode/")])
        save(events_dir/"metrics.json", metrics)
        if interrupted:
            raise KeyboardInterrupt(error)
        if error or completion["errors"] or not completion["final_turn_completed"]:
            raise RuntimeError(error or "incomplete Agent turn")
        plan = read(work/"final_answer.json")
        (validate_plan if plan_validator is None else plan_validator)(
            plan, context["image_size_px"], manifest["records"])
        save(folder/"plan.json", plan)
        metrics.update(probe.access_summary(access, manifest["records"], plan))
        save(events_dir/"metrics.json", metrics)
        if access["queue_overflow"]:
            raise RuntimeError("file access audit overflow")
        if not all(all(ref[k] for k in ("before_read", "action_read", "after_read"))
                   for ref in metrics["cited_experience_read_checks"]):
            raise RuntimeError("cited experience not fully read")
        return plan, metrics


def prepare_action(runtime, before, plan, folder, *, target_sampler=None, orientation_resolver=None,
                   check_grasp_height_spread=True, smoothing=False):
    """Only adapt pixels to the existing automatic collection primitive."""
    from cloth_agent.gamepad_collect import automatic_program, staged_place_release
    before.validate_for(runtime.config, live=True, check_age=False)
    spread = runtime.config.limits["max_surface_spread_mm"]
    # Edge grasps can opt out of spread rejection, but still require valid measured depth.
    start = before.sample(plan["grasp_px"], spread if check_grasp_height_spread else None)
    save(folder/"grasp_depth_policy.json", dict(
        height_spread_check_enabled=check_grasp_height_spread,
        max_height_spread_mm=spread if check_grasp_height_spread else None,
        valid_center_required=True,min_valid_patch_pixels=18,patch_size_px=5,
        depth_estimator="median of valid measured XYZ"))
    target = (before.sample(plan["target_px"], spread) if target_sampler is None
              else target_sampler(before, plan["target_px"], plan["grasp_px"],
                                  spread, folder/"target_depth.json"))
    points = [dict(name=name, pixel_xy=plan[key], world_xyz_mm=value.tolist())
              for name, key, value in (("start", "grasp_px", start), ("target", "target_px", target))]
    if target_sampler is not None:
        provenance = read(folder/"target_depth.json")
        points[0].update(depth_source="MEASURED_DEPTH", measured_depth=True)
        points[1].update(depth_source=provenance["source"], measured_depth=provenance["measured_depth"])
    save(folder/"points.json", points)
    a, b = [np.asarray(p["world_xyz_mm"]) for p in points]
    distance = float(np.linalg.norm(b[:2]-a[:2]))
    if smoothing and not 10. <= distance <= 30.:
        save(folder/'motion_profile.json',dict(is_smoothing=True,status='REJECT',distance_mm=distance,
            min_translation_mm=10.,max_translation_mm=30.,lift_height_mm=10.))
        raise DualArmError(f'smoothing action rejected: measured XY displacement {distance:.2f} mm; required 10–30 mm. Choose a closer Pb along the intended direction; do not change to normal action merely to bypass the limit.')
    if not MOTION["min_translation_mm"] <= distance <= MOTION["max_translation_mm"]:
        raise DualArmError(f"metric displacement {distance:.2f} mm outside existing 10–150 mm range")
    arm = runtime.config.arms["left"]
    rpy = matrix_pose(arm.world_from_base @ pose_matrix(runtime.models["left"].forward(arm.home_joints)))[3:]
    if orientation_resolver is not None:
        rpy = orientation_resolver(runtime, before, plan, a, rpy, folder)
    source = automatic_program(runtime, a, b, rpy, approach_height=50., grasp_offset=-15.,
                               lift_height=10. if smoothing else 50., clearance=30., place_offset=5.)
    (folder/"motion.py").write_text(source)
    primitive = dict(MOTION)
    if smoothing:primitive.update(lift_height_mm=10.,max_translation_mm=30.)
    save(folder/"motion_profile.json",dict(is_smoothing=smoothing,status="GEOMETRY_VALIDATED",
        distance_mm=distance,lift_height_mm=primitive["lift_height_mm"],
        max_translation_mm=primitive["max_translation_mm"],
        transfer_z_mm=float(max(a[2],b[2])+primitive["lift_height_mm"]),
        lift_reference="max measured grasp / target surface Z; not rise from closed-gripper TCP",
        grasp_offset_mm=primitive["grasp_offset_mm"]))
    if orientation_resolver is not None:
        primitive['orientation'] = 'chosen before grasp; fixed throughout closure/lift/transfer/place'
    save(folder/"geometry.json", dict(distance_mm=distance, rpy_world_deg=rpy.tolist(), primitive=primitive))
    initial = {k: np.asarray(v["joints"]) for k, v in runtime.coordinator.snapshots().items()}
    # Snapshot age alone is not an expiry: certify current stationary pose against capture before dispatch.
    runtime.arrival({k: np.asarray(v) for k, v in before.meta["joints_deg"].items()})
    operations = runtime.plan(source, initial, preserve_command_boundaries=True)
    opens = [i for i, (kind, data) in enumerate(operations) if kind == "gripper" and data["action"] == "open"]
    if len(opens) != 2:
        raise DualArmError("unexpected automatic pick/place sequence")
    i = opens[-1]
    operations = operations[:i]+staged_place_release(operations[i:])
    save(folder/"safety_checks.json", dict(status="PASSED_EXISTING_COLLECTION_CHECKS",
         path_validation_enabled=runtime.config.path_validation_enabled,
         collision_envelopes_disabled_arms=sorted(runtime.config.collision_envelopes_disabled_arms),
         planning_stats=runtime.plan_stats))
    return operations, initial


def render(root, summary):
    detail = summary.get("termination_description", summary["termination_detail"])
    no_episode = summary["total_rounds"] == 0
    lines = ["# trajectory_memory_full_episode_v1", "",
             f"终止原因：**{summary['termination_reason']}**；{detail}", "",
             f"Rounds：{summary['total_rounds']}；physical manipulation dispatches：{summary['physical_actions']}；"
             f"已确认 acquisition failure：{summary['acquisition_failures']}。",
             f"总 Agent latency：{summary['total_agent_latency_s']:.1f} 秒；"
             f"总 episode robot execution：{summary['total_robot_execution_s']:.1f} 秒。",
             f"初始化模型调用耗时（另计）：{summary.get('initialization_agent_latency_s', 0.):.1f} 秒。",
             "初始化与 episode manipulation 分开计数。acquisition 未可靠验证，不把 0 个已确认失败解释为全部抓取成功。",
             "", ("**人工 FOLD_READY：未进行（null），episode 尚未开始。**" if no_episode else
                  "**人工 FOLD_READY：PENDING（null）**。Agent 的 DONE 不是人工结论。"),
             "", "|Round|Decision|Execution|GPT s|Memory steps / images|",
             "|---|---|---|---:|---:|"]
    panels = []
    for item in summary["rounds"]:
        folder = root/item["round"]
        plan = read(folder/"plan.json") if (folder/"plan.json").exists() else None
        metrics = read(folder/"agent_events/metrics.json") if (folder/"agent_events/metrics.json").exists() else {}
        lines.append(f"|[{item['round']}]({item['round']}/plan.json)|{plan['decision'] if plan else 'NO_VALID_PLAN'}|"
                     f"{item['execution']['status']}|{metrics.get('wall_latency_s', 0):.1f}|"
                     f"{metrics.get('memory_step_files_read', 0)} / {metrics.get('historical_source_images_read', 0)}|")
        panel = Image.new("RGB", (1280, 420), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((15, 8), item["round"]+" / "+item["execution"]["status"], font=probe.font(23), fill="black")
        for i, name in enumerate(("before", "after")):
            path = folder/name/"rgb.png"
            if path.exists():
                with Image.open(path) as im:
                    picture = im.convert("RGB")
                if name == "before" and plan:
                    picture = probe.draw_action(picture, plan["grasp_px"], plan["target_px"])
                picture.thumbnail((640, 360))
                panel.paste(picture, (640*i, 55))
            draw.text((640*i+15, 33), name, font=probe.font(17), fill="black")
        panel.save(folder/"before_after.png")
        panels.append(panel)
        if plan:
            probe.selected_sheet(root, plan, folder/"selected_experiences/contact_sheet.png")
            with Image.open(folder/"before/rgb.png") as im:
                probe.draw_action(im.convert("RGB"), plan["grasp_px"], plan["target_px"]).save(folder/"proposed_action_overlay.png")
    if panels:
        canvas = Image.new("RGB", (1280, 420*len(panels)), "white")
        for i, panel in enumerate(panels):
            canvas.paste(panel, (0, i*420))
        canvas.save(root/"episode_overview.png")
        lines += ["", "![before/after](episode_overview.png)"]
    elif (root/"initialization/before/rgb.png").exists():
        lines += ["", "**本次在初始化动作下发前停止，未进入 unfolding round。**"
                  "没有新初始化后的状态、episode plan 或 after，不能据此评价 memory Agent 的闭环能力。",
                  "", "[安全停止原始记录](exception.txt)；[初始化执行记录](initialization/result.json)。",
                  "", "![初始化前观察，未执行动作](initialization/before/rgb.png)"]
    lines += ["", "## 记录与边界", "",
              "- [完整 summary](episode_summary.json)；[冻结历史清单](memory_manifest.json)；[固定协议](protocol.json)。",
              "- 进入 round 后，round_XXX/agent_events 保留 prompt、CLI 事件、文件访问和 latency。",
              "- 有引用时，selected_experiences 保存引用记录副本和 before/action/after 拼图。",
              "- 有执行时，execution_images 保存约 5 Hz timestamped RGB；手腕相机可见性可能受遮挡。",
              "- 本轮没有训练、Graph matching、新 Graph 节点、memory 更新或 no-memory baseline。",
              "- 按用户要求，与 CollectionRuntime 原有设置一致：path_validation_enabled=false，"
              "left 附加碰撞余量关闭，Home/perception 使用采集脚本原有路径。"
              "没有为实验额外打开检查，也不把未启用的检查标成通过。",
              "- 每次只执行一个既有 lift_pull_place，完成后重新采 RGB-D。采集脚本已启用的检查一旦拒绝立即停止。",
              "- Memory 图片读取量包括脚本制作拼图的批量读取，不代表逐张语义检查。",
              "", "## 人工结论", "",
              ("初始化未完成，本次不产生展开能力或 FOLD_READY 结论。" if no_episode else
               "请查看最终 observation 与执行图像后，在 human_review.json 中填写 FOLD_READY 判断。"),
              "本实验不会自动重试、修改 prompt 或启动下一 episode。", ""]
    (root/"report.md").write_text("\n".join(lines))


def run(args, *, experiment_name="trajectory_memory_full_episode_v1", prompt=None,
        prepare_inputs=None, verify_inputs=None, plan_round=None, render_report=None, source_file=None,
        runtime_factory=None, initial_history=None):
    prompt = PROMPT if prompt is None else prompt
    prepare_inputs = freeze_memory if prepare_inputs is None else prepare_inputs
    verify_inputs = verify_memory if verify_inputs is None else verify_inputs
    plan_round = agent_round if plan_round is None else plan_round
    render_report = render if render_report is None else render_report
    source_file = __file__ if source_file is None else source_file
    runtime_factory = EpisodeCollectionRuntime if runtime_factory is None else runtime_factory
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/"private").mkdir()
    runtime = None
    summary = dict(experiment=experiment_name, started_at=datetime.now(timezone.utc).isoformat(),
                   termination_reason=None, termination_detail="", total_rounds=0, physical_actions=0,
                   acquisition_failures=0, acquisition_unknown=0, total_agent_latency_s=0.,
                   total_robot_execution_s=0., initialization_physical_actions=0,
                   human_fold_ready=None, human_review_status="PENDING", rounds=[], model=MODEL,
                   reasoning_effort=EFFORT, max_rounds=args.max_rounds, prompt_sha256=None)
    history = []
    began = time.monotonic()
    folder = None
    old_signal = signal.getsignal(signal.SIGTERM)
    def interrupted(*_):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        manifest = prepare_inputs(root, args.data_root, args.scope_dates)
        if initial_history is not None:
            history = initial_history(root)
        (root/"prompt.txt").write_text(prompt)
        summary["prompt_sha256"] = digest(root/"prompt.txt")
        save(root/"protocol.json", dict(experiment=experiment_name, model=MODEL, effort=EFFORT, schema=SCHEMA, prompt=prompt,
             max_rounds=args.max_rounds, motion=MOTION,
             initialization=("user-placed state; no reset" if getattr(args, "skip_initialization", False)
                             else "existing run_reset, max_retries=0"),
             acquisition_verification="UNRELIABLE; preserve images/feedback; no automatic failure inference",
             repeat_guard="same motion and both points within 8 px; ROI RGB p95 difference <8",
             runtime_policy="unchanged CollectionRuntime per explicit user request",
             path_validation_enabled=False, collision_envelopes_disabled_arms=["left"],
             preparation="CollectionRuntime._plan_preparation (inherited unchanged)",
             source_sha256=digest(source_file)))
        shutil.copyfile(source_file, root/"private/experiment_source.py")
        if str(source_file) != __file__:
            shutil.copyfile(__file__, root/"private/shared_episode_source.py")
        protected = ["cloth_agent/codex_native_runtime.py", "cloth_agent/gamepad_collect_runtime.py",
                     "cloth_agent/gamepad_collect.py", "cloth_agent/random_fold_reset.py",
                     "cloth_agent/experiments/graph_policy_replay.py", "config/dual_arm.local.json"]
        save(root/"private/protected_source_hashes.json", {p: digest(p) for p in protected})
        save(root/"episode_summary.json", summary)
        print(f"FROZEN {manifest['memory_steps']} steps / {manifest['memory_sessions']} sessions", flush=True)
        runtime = runtime_factory(args.config)
        if getattr(args, "skip_initialization", False):
            print("START FROM USER-PLACED STATE; no pick-up/drop initialization", flush=True)
            startup = time.monotonic()
            runtime.observe(root/"startup_observation")
            summary["startup_observation_wall_s"] = time.monotonic()-startup
            summary["initialization_wall_s"] = 0.
            summary["initialization_kind"] = "USER_PLACED_NO_RESET"
        else:
            from cloth_agent.random_fold_reset import run_reset
            print("INITIALIZE one existing pick-up/drop, no retries", flush=True)
            init = run_reset(runtime, root/"initialization", max_retries=0)
            summary["initialization_physical_actions"] = sum(bool(x.get("executed")) for x in init["attempts"])
            summary["initialization_wall_s"] = init["elapsed_s"]
        from cloth_agent.gamepad_collect import TrajectoryRecorder
        for index in range(args.max_rounds):
            folder = root/f"round_{index:03d}"
            folder.mkdir()
            (folder/"selected_experiences").mkdir()
            item = dict(round=folder.name, execution=dict(status="NOT_EXECUTED", physical_execution=False),
                        supporting_experiences=[], acquisition="NOT_ATTEMPTED")
            summary["rounds"].append(item)
            summary["total_rounds"] = len(summary["rounds"])
            round_started = time.monotonic()
            timing = dict(agent_wall_s=0., robot_execution_s=0., host_planning_s=0., observation_s=0.)
            save(folder/"execution.json", item["execution"])
            save(folder/"timing.json", timing)
            try:
                capture_started = time.monotonic()
                before = runtime.capture_current(folder/"before")
                timing["observation_s"] = time.monotonic()-capture_started
                verify_inputs(root)
                print(f"ROUND {index:03d} AGENT", flush=True)
                plan, metrics = plan_round(root, folder, before, history, manifest, args.timeout_s)
                timing["agent_wall_s"] = metrics["wall_latency_s"]
                item["memory_access"] = {k: metrics[k] for k in
                    ("memory_step_files_read", "memory_sessions_read", "historical_source_images_read",
                     "cited_experience_read_checks", "session_ids")}
                item["supporting_experiences"] = plan["supporting_experiences"]
                for ref in plan["supporting_experiences"]:
                    source = root/"memory"/ref["trajectory_id"]/ref["step_id"]
                    shutil.copytree(source, folder/"selected_experiences"/ref["trajectory_id"]/ref["step_id"])
                print(f"ROUND {index:03d} {plan['decision']} {plan['grasp_px']} -> {plan['target_px']}", flush=True)
                if plan["decision"] != "ACTION":
                    summary.update(termination_reason=plan["decision"], termination_detail=plan["short_rationale"])
                    item["execution"]["status"] = "NO_ACTION_"+plan["decision"]
                    runtime.capture_current(folder/"after")
                    break
                if repeated_action(plan, before.image, history):
                    summary.update(termination_reason="ABSTAIN", termination_detail="Host repetition guard: unchanged observation and same action")
                    item["execution"]["status"] = "REPETITION_GUARD"
                    runtime.capture_current(folder/"after")
                    break
                prepare_started = time.monotonic()
                operations, initial = prepare_action(runtime, before, plan, folder)
                timing["host_planning_s"] = time.monotonic()-prepare_started
                recorder = TrajectoryRecorder(runtime, folder)
                robot_started = None
                try:
                    recorder.start()
                    recorder.stage("memory_episode_single_manipulation")
                    recorder.check()
                    runtime.arrival(initial)
                    save(folder/"dispatch_intent.json", dict(one_manipulation=True, plan=plan,
                         observation_sha256=digest(before.image), safety="PASSED"))
                    summary["physical_actions"] += 1  # dispatched attempts include partial execution.
                    item["execution"] = dict(status="DISPATCHED", physical_execution=True)
                    save(folder/"execution.json", item["execution"])
                    save(root/"episode_summary.json", summary)
                    robot_started = time.monotonic()
                    execution = runtime._execute_operations(operations, folder)
                    item["execution"] = execution
                    save(folder/"execution.json", execution)
                    if execution["status"] != "EXECUTED":
                        raise DualArmError("execution failed: "+str(execution))
                    item["acquisition"] = "UNKNOWN_PENDING_HUMAN_REVIEW"
                    summary["acquisition_unknown"] += 1
                    recorder.check()
                    recorder.stage("return_perception")
                    runtime.open_at_home_on_return = True
                    try:
                        runtime.move_to_perception(folder/"return_perception")
                    finally:
                        runtime.open_at_home_on_return = False
                    recorder.check()
                finally:
                    if robot_started is not None:
                        timing["robot_execution_s"] = time.monotonic()-robot_started
                    recorder.close()
                recorder.check()
                after = runtime.capture_current(folder/"after")
                same, delta = unchanged(before.image, after.image)
                save(folder/"observation_change.json", dict(unchanged=same, roi_rgb_p95_abs_difference=delta,
                     interpretation="image diagnostic only; not acquisition or cloth NO_PROGRESS"))
                history.append(dict(round=folder.name, directory=folder, plan=plan,
                                    before=before.image, acquisition=item["acquisition"]))
                if index == args.max_rounds-1:
                    summary.update(termination_reason="ROUND_LIMIT", termination_detail="Reached configured maximum; final after captured")
            except BaseException as exc:
                if item["execution"]["status"] == "NOT_EXECUTED":
                    item["execution"]["status"] = "USER_INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "SAFETY_REJECTED"
                item["execution"]["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                metrics_path = folder/"agent_events/metrics.json"
                if metrics_path.exists():
                    timing["agent_wall_s"] = read(metrics_path)["wall_latency_s"]
                timing["round_wall_s"] = time.monotonic()-round_started
                summary["total_agent_latency_s"] += timing["agent_wall_s"]
                summary["total_robot_execution_s"] += timing["robot_execution_s"]
                item["timing"] = timing
                item["execution"]["acquisition"] = item["acquisition"]
                save(folder/"timing.json", timing)
                save(folder/"execution.json", item["execution"])
                save(root/"episode_summary.json", summary)
    except BaseException as exc:
        summary.update(termination_reason="USER_STOP" if isinstance(exc, KeyboardInterrupt) else "SAFETY_STOP",
                       termination_detail=f"{type(exc).__name__}: {exc}")
        (root/"exception.txt").write_text(traceback.format_exc())
        if runtime:
            runtime.cancel.set()
            try:
                summary["stop_results"] = runtime.stop()
            except Exception as stop_error:
                summary["stop_error"] = str(stop_error)
        if folder and not (folder/"after").exists():
            save(folder/"after/unavailable.json", dict(reason="Stopped; no repositioning/capture attempted after safety rejection"))
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        if runtime:
            try:
                runtime.close()
            except Exception as exc:
                summary["cleanup_error"] = str(exc)
        summary["episode_wall_s"] = time.monotonic()-began
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        if (root/"memory_manifest.json").exists():
            try:
                verify_inputs(root)
                summary["frozen_memory_unchanged"] = True
            except Exception as exc:
                summary["frozen_memory_unchanged"] = False
                summary.update(termination_reason="SAFETY_STOP", termination_detail=str(exc))
        if (root/"initialization/result.json").exists():
            initialization = read(root/"initialization/result.json")
            summary["initialization_result"] = initialization
            summary["initialization_physical_actions"] = sum(
                bool(x.get("executed")) for x in initialization.get("attempts", []))
        if (root/"initialization/model_metrics.json").exists():
            summary["initialization_agent_latency_s"] = read(root/"initialization/model_metrics.json")["reasoning_time_s"]
        summary["all_model_wall_latency_s"] = summary["total_agent_latency_s"]+summary.get("initialization_agent_latency_s", 0.)
        accessed_images, accessed_steps = set(), set()
        for metrics_file in root.glob("round_*/agent_events/metrics.json"):
            metrics = read(metrics_file)
            accessed_images.update(metrics.get("source_images", []))
            accessed_steps.update((r["trajectory_id"], r["step_id"]) for r in metrics.get("accessed_steps", []))
        summary["unique_historical_source_images_read"] = len(accessed_images)
        summary["unique_historical_steps_read"] = len(accessed_steps)
        summary["unique_historical_sessions_read"] = len({r[0] for r in accessed_steps})
        summary["sum_per_round_historical_image_reads"] = sum(
            item.get("memory_access", {}).get("historical_source_images_read", 0) for item in summary["rounds"])
        protected_path = root/"private/protected_source_hashes.json"
        if protected_path.exists():
            summary["protected_source_files_unchanged"] = {
                p: digest(p) == sha for p, sha in read(protected_path).items()}
        save(root/"human_review.json", dict(fold_ready=None, acquisition_by_round={}, reviewer=None, notes=""))
        save(root/"episode_summary.json", summary)
        try:
            render_report(root, summary)
        except Exception as exc:
            summary["report_error"] = str(exc)
            save(root/"episode_summary.json", summary)
        print("EPISODE FINISHED "+str(summary["termination_reason"])+": "+summary["termination_detail"], flush=True)
    return 1 if summary["termination_reason"] == "SAFETY_STOP" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("config/dual_arm.local.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/demonstrations"))
    parser.add_argument("--scope-dates", nargs="+", default=["20261019", "20261020"])
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=1200.)
    args = parser.parse_args()
    if not 1 <= args.max_rounds <= 50 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid limits")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
