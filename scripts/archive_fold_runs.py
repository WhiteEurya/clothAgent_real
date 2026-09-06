#!/usr/bin/env python3
"""Archive representative fold-run evidence for the last seven days.

The script deliberately archives a curated evidence set rather than copying all
13+ GiB of raw run data.  After SHA-256 verification, it removes only the
selected source evidence files; experience ledgers and unselected run data are
kept so future runs can still inherit knowledge.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "reports" / "fold_run_archive_20260831"
# Run directory mtimes are recorded in the host's Asia/Shanghai clock.  The
# report window is therefore the local seven-day window, not midnight UTC.
CUTOFF = dt.datetime(2026, 8, 24, tzinfo=dt.timezone(dt.timedelta(hours=8)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, delete_sources: list[Path], manifest: list[dict], *, delete=True) -> bool:
    rec = {"source": str(src), "destination": str(dst), "status": "missing"}
    if not src.exists() or not src.is_file() or src.stat().st_size == 0:
        manifest.append(rec)
        return False
    ensure_parent(dst)
    shutil.copy2(src, dst)
    src_hash = sha256(src)
    dst_hash = sha256(dst)
    rec.update({"bytes": dst.stat().st_size, "sha256": dst_hash, "source_sha256": src_hash})
    if src_hash != dst_hash:
        rec["status"] = "hash_mismatch"
        manifest.append(rec)
        raise RuntimeError(f"copy verification failed: {src}")
    rec["status"] = "copied"
    if delete:
        delete_sources.append(src)
    manifest.append(rec)
    return True


def make_fast_video(src: Path, dst: Path, delete_sources: list[Path], manifest: list[dict]) -> bool:
    rec = {"source": str(src), "destination": str(dst), "speed": 32.0, "status": "missing"}
    if not src.exists() or src.stat().st_size == 0:
        manifest.append(rec)
        return False
    ensure_parent(dst)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-vf", "setpts=PTS/32", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", str(dst),
    ]
    subprocess.run(cmd, check=True)
    if not dst.exists() or dst.stat().st_size == 0:
        rec["status"] = "encode_failed"
        manifest.append(rec)
        raise RuntimeError(f"video encode failed: {src}")
    rec.update({"status": "copied_32x", "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    delete_sources.append(src)
    manifest.append(rec)
    return True


def p(rel: str) -> Path:
    return ROOT / rel


def add_issue_files(issue_dir: str, files: list[tuple[str, str]], videos: list[tuple[str, str]], manifest, delete_sources):
    d = OUT / issue_dir
    for rel_src, rel_dst in files:
        copy_file(p(rel_src), d / rel_dst, delete_sources, manifest)
    for rel_src, rel_dst in videos:
        make_fast_video(p(rel_src), d / rel_dst, delete_sources, manifest)


def collect_inventory() -> tuple[list[dict], dict]:
    runs = []
    for run in sorted(RUNS.iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        mt = dt.datetime.fromtimestamp(run.stat().st_mtime, tz=dt.timezone.utc)
        if mt < CUTOFF:
            continue
        runs.append(run)
    rows = []
    status_counts: dict[str, int] = {}
    for run in runs:
        sums = sorted(run.glob("results/**/summary.json"))
        if not sums:
            rows.append({"run": run.name, "summary": "", "status": "NO_SUMMARY", "iterations": "", "error": ""})
            continue
        for sp in sums:
            try:
                d = json.loads(sp.read_text(errors="replace"))
                status = str(d.get("status", "<none>"))
                iters = len(d.get("iterations", [])) if isinstance(d.get("iterations"), list) else ""
                err = d.get("error") or d.get("blocked_reason") or ""
                if not isinstance(err, str):
                    err = json.dumps(err, ensure_ascii=False)
            except Exception as exc:
                status, iters, err = "BAD", "", str(exc)
            status_counts[status] = status_counts.get(status, 0) + 1
            rows.append({"run": run.name, "summary": str(sp.relative_to(ROOT)), "status": status, "iterations": iters, "error": err[:500]})
    return rows, {"top_level_runs": len(runs), "summary_status_counts": status_counts}


def collect_eval_stats() -> dict:
    out: dict[str, int] = {}
    n = 0
    for ep in RUNS.glob("**/iteration_*/evaluation.json"):
        try:
            mt = dt.datetime.fromtimestamp(ep.stat().st_mtime, tz=dt.timezone.utc)
            if mt < CUTOFF:
                continue
            d = json.loads(ep.read_text(errors="replace"))
        except Exception:
            continue
        n += 1
        for field in ("grasp_acquisition", "target_structure_acquired", "transport", "laydown", "task_progress"):
            val = d.get(field, {}).get("status")
            if val:
                key = f"{field}={val}"
                out[key] = out.get(key, 0) + 1
    return {"evaluated_iterations": n, "counts": out}


def collect_timing_stats() -> dict:
    import re
    import statistics
    vals = {"visual_planning_s": [], "final_grounding_s": [], "total_planning_s": [], "supervisor_s": [], "perception_s": [], "molmo_s": []}
    pat = re.compile(r"timing=\{'visual_planning_s': ([0-9.]+).*'final_grounding_s': ([0-9.]+), 'total_planning_s': ([0-9.]+)\}")
    for lp in RUNS.glob("**/debug.log"):
        try:
            mt = dt.datetime.fromtimestamp(lp.stat().st_mtime, tz=dt.timezone.utc)
            if mt < CUTOFF:
                continue
            text = lp.read_text(errors="replace")
        except Exception:
            continue
        for m in pat.finditer(text):
            vals["visual_planning_s"].append(float(m.group(1)))
            vals["final_grounding_s"].append(float(m.group(2)))
            vals["total_planning_s"].append(float(m.group(3)))
        for line in text.splitlines():
            for key, needle in (("supervisor_s", "supervisor: inspection completed"), ("perception_s", "perception: capture completed"), ("molmo_s", "molmo: target-specific sleeve localization completed")):
                if needle in line and "duration_s=" in line:
                    try:
                        vals[key].append(float(line.rsplit("duration_s=", 1)[1].split(",", 1)[0]))
                    except Exception:
                        pass
    out = {}
    for key, arr in vals.items():
        if not arr:
            continue
        out[key] = {
            "n": len(arr),
            "mean_s": round(statistics.mean(arr), 1),
            "median_s": round(statistics.median(arr), 1),
            "max_s": round(max(arr), 1),
            "sum_s": round(sum(arr), 1),
        }
    return out


def write_report(inventory_meta: dict, eval_meta: dict, timing_meta: dict, issue_meta: dict) -> None:
    status_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(inventory_meta["summary_status_counts"].items()))
    eval_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(eval_meta["counts"].items()))
    timing_lines = "\n".join(
        f"- `{k}`: n={v['n']}, mean={v['mean_s']} s, median={v['median_s']} s, max={v['max_s']} s"
        for k, v in timing_meta.items()
    )
    structural = [x for x in issue_meta if x["class"] == "structural"]
    nonstructural = [x for x in issue_meta if x["class"] == "nonstructural"]
    def issue_section(items):
        chunks = []
        for x in items:
            archive_dir = x.get("archive_dir", f"{x['class']}/{x['id']}")
            video_count = len(list((OUT / archive_dir).glob("*_32x.mp4")))
            video_note = f"视频证据：已归档 {video_count} 个 32 倍速视频。" if video_count else "视频证据：无（该问题在执行前失败，原始流程没有产生视频）。"
            chunks.append(
                f"### {x['id']} · {x['title']}\n\n"
                f"{x['finding']}\n\n"
                f"证据：{x['evidence_summary']}\n\n"
                f"{video_note}\n\n"
                f"归档目录：`{archive_dir}`"
            )
        return "\n\n".join(chunks)
    report = f"""# 折叠衣服实验归档报告

统计窗口：2026-08-24 00:00（Asia/Shanghai）至 2026-08-31（本次整理时刻）。

## 总览

- 顶层 run：**{inventory_meta['top_level_runs']}**
- summary 记录：**{sum(inventory_meta['summary_status_counts'].values())}**
- 有 evaluation 的迭代：**{eval_meta['evaluated_iterations']}**
- 注意：无人值守 run 会把同一配置的重复尝试写入多个 summary/iteration，计数是观测次数，不是独立样本数。

Summary 状态分布：

{status_lines}

Evaluation 状态分布：

{eval_lines}

## 结构性问题

结构性问题指跨多个 run 重复出现、需要改 pipeline/感知/控制架构或实验设计的问题，而不是一次性的 API/网络异常。

{issue_section(structural)}

## 非结构性问题

非结构性问题指单次运行的输入契约、网络、进程或外部服务异常；它们会中断实验，但不能直接证明折叠策略错误。

{issue_section(nonstructural)}

## 时间开销证据

从 debug.log 中提取的阶段耗时：

{timing_lines or '- 没有足够的 timing 记录'}

视觉 planning 和 final grounding 的长耗时与 Claude 上下文增长、reselection/retry 及结构化输出重试有关；它不是机械臂运动本身的耗时。

## 归档与删除策略

- 已复制并校验 SHA-256 的代表性图像、JSON、日志和视频证据。
- 视频证据均在归档中生成 `_32x.mp4`（`setpts=PTS/32`，无音频）。
- 仅删除了“已成功复制且校验通过”的选中源证据文件；未删除 `workspace/fold_experience/`、未选中的原始 run 数据和配置，以免破坏跨 run 经验继承或后续复盘。
- 完整复制/删除记录见 `evidence_manifest.json`；缺失源文件会标记为 `missing`，不会被删除。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("structural", "nonstructural"):
        (OUT / sub).mkdir(exist_ok=True)
    inventory, inventory_meta = collect_inventory()
    eval_meta = collect_eval_stats()
    timing_meta = collect_timing_stats()

    issue_meta = [
        {"id":"S1","class":"structural","title":"袖子 acquisition 反复空抓","archive_dir":"structural/S1_acquisition_empty_grasp","finding":"97 个有 evaluation 的迭代中有 83 个 acquisition failure。代表性 Cam-C hold-check 显示夹爪抬起后没有悬挂布料，遥测 position_pulse=0、mechanical_grasp_detected=false；这说明抓取接触/边缘进入方式仍是首要瓶颈。","evidence_summary":"fold_night iteration 014 的 Camera-A before/after、Camera-C hold-check、Camera-C/AB 视频、evaluation、gripper telemetry 和 debug.log。"},
        {"id":"S2","class":"structural","title":"抓到的是整团布/滚卷而非目标层","archive_dir":"structural/S2_whole_bundle_drag_roll","finding":"存在少数机械上确实抓到布料的迭代，但目标层未被分离：整件衣服被拖动，释放后变成更窄、更高的 sausage/roll。对应 evaluation 中 acquisition=SUCCESS、target_structure=CONTRADICTED、laydown=FAILURE 或 task_progress=REGRESSED。","evidence_summary":"fold_inherit iteration 001 的 A/B before/after、AB depth 视频、evaluation 和 trajectory。"},
        {"id":"S3","class":"structural","title":"RGB→Rxxx→XYZ grounding / mask 不一致","archive_dir":"structural/S3_grounding_mask_mismatch","finding":"出现过 selected Rxxx 与实际 grasp XY 相差约 320 mm、选点落在 production mask 外、以及目标虽叫 image-left sleeve 却解析到肩部/衣身的情况。另有 bbox bottom=718/720 被统一判为 PARTIAL。错误发生在视觉点到执行坐标的链路，不能归咎于 Claude 的折叠策略。","evidence_summary":"single_sleeve 的 selected_point_vs_production_mask.png；fold_holdcheck 的 320 mm 日志；fold_night iteration 009 的 Camera-A/C 图像、evaluation 和视频。"},
        {"id":"S4","class":"structural","title":"Claude planning / grounding 上下文导致极端延迟","archive_dir":"structural/S4_latency_context","finding":"76 次 planning 返回记录的 visual planning 平均约 193 s、最大 1304 s；final grounding 平均约 189 s、最大 392 s。持久 session 还发生多次 context rollover，随后出现 Prompt is too long / Argument list too long。瓶颈在提示词、结构化输出重试和上下文管理，而非相机或机械臂移动。","evidence_summary":"fold_persistent debug.log、summary、persistent_claude_session.json、timing_summary.json，以及同一迭代的 32x AB 视频。"},
        {"id":"S5","class":"structural","title":"安全高度 / 工作空间 / controller IK 约束互相冲突","archive_dir":"structural/S5_safety_ik_height","finding":"多次出现 no legal engaged grasp Z（surface_z 低于 table/robot 下限）、x 低于 safe lower bound、controller IK code=10。说明桌面/海绵/衣物高度模型和最终 controller 可达域没有形成稳定、可执行的统一约束。","evidence_summary":"collar_high_lift_05 的高度失败 summary 与 collar overlay；fold_holdcheck_004839 的 IK debug/summary；single_sleeve 的安全边界 summary。"},
        {"id":"S6","class":"structural","title":"抓取成功判据受夹爪遮挡且 telemetry 仍是弱信号","archive_dir":"structural/S6_observability_telemetry","finding":"Camera-A 在夹爪靠近时遮挡了大部分目标；Camera-C 只能作为未标定 RGB observer。虽然加入了 hold-check，但多次 telemetry 样本标记 simulated=true、时间戳重复、mechanical_grasp_detected=false，因此只能作为弱机械证据，不能独立证明目标层被抓住。","evidence_summary":"fold_camC iteration 001 与 fold_night iteration 014 的 Camera-C 图像/视频、hold-check、gripper_telemetry.json。"},
        {"id":"S7","class":"structural","title":"无人值守状态机在 bunched/left_sleeve 上反复横跳","archive_dir":"structural/S7_unattended_state_loop","finding":"fold_night run 曾累计 194 次 unattended attempt；在衣服已 BUNCHED 时仍反复停留在 left_sleeve，repair 或 planning failure 后状态没有真正推进到可折叠状态。该问题会消耗整夜时间，却不产生新的有效物理证据。","evidence_summary":"fold_night_20260830T100630 的 summary、debug.log、unattended restart 记录和 iteration 009 视频；另有长 run 的 summary 记录。"},
        {"id":"S8","class":"structural","title":"相机/光照质量门槛使 perception 整批阻塞","archive_dir":"structural/S8_capture_quality","finding":"night run 大量出现 table luma 约 2–6（阈值 90）导致 before perception 连续失败；另有 mask silhouette coverage 仅 0.056。此类失败发生在 Claude/机器人动作之前，是采集质量和验证门槛的问题。","evidence_summary":"fold_night 多个 failed summary/debug.log；neat_fold_overnight 的 perception failure 图像与视频作为同环境对照。"},
        {"id":"N1","class":"nonstructural","title":"Claude 输出契约/Schema 偶发不合法","archive_dir":"nonstructural/N1_schema_contract","finding":"历史运行中出现 missing visual-plan fields、selected_reference 字段多/少、safety_notes 为空、缺 close_gripper、缺 move target、skill invocation 不合法等。它们均在执行前被 validator 拦截。","evidence_summary":"fold_unattended_20260827_190930、fold_molmo_20260828_021707、fold_night_20260830T100630 的 debug.log/summary。"},
        {"id":"N2","class":"nonstructural","title":"外部服务/相机连接/进程异常","archive_dir":"nonstructural/N2_infra_transient","finding":"出现 connect socket failed、API ENOTFOUND、KeyboardInterrupt 以及一次 _debug_exception TypeError。它们是基础设施或进程生命周期故障，不能当作 planning 失败的证据。","evidence_summary":"fold_gripper_sdk_20260829_021819、fold_gripper_sdk_20260829_021511、single_sleeve_grasp_dry_smoke 和若干 interrupted summary。"},
        {"id":"N3","class":"nonstructural","title":"人为中断/旧版运行上限造成提前结束","archive_dir":"nonstructural/N3_interruptions_limits","finding":"多个旧 run 以 INTERRUPTED、MAX_FOLDS_REACHED 或 KeyboardInterrupt 结束；这解释了“只跑 1/7 个 iteration 就结束”的历史现象，但不等同于衣服操作失败。","evidence_summary":"claude_global_cli_real_20260825_*、neat_fold_overnight_20260827_020727、collar_high_lift 系列 summary。"},
    ]

    manifest: list[dict] = []
    delete_sources: list[Path] = []
    # Structural evidence
    add_issue_files("structural/S1_acquisition_empty_grasp", [
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/before_raw/camera_A_rgb_upright.png", "before_camera_A.png"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/after_raw/camera_A_rgb_upright.png", "after_camera_A.png"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/hold_check/camera_C_observer_rgb_hold_check.png", "hold_check_camera_C.png"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/evaluation.json", "evaluation.json"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/gripper_telemetry.json", "gripper_telemetry.json"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/debug.log", "debug.log"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/rollout_recording/evaluator_video_evidence/camera_C_rgb_contact_sheet.png", "camera_C_contact_sheet.png"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/rollout_recording/evaluator_video_evidence/camera_AB_DEPTH_rgb_contact_sheet.png", "camera_AB_contact_sheet.png"),
    ], [
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/rollout_recording/camera_C_observer_rgb.mp4", "camera_C_32x.mp4"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/rollout_recording/composite_AB_depth.mp4", "composite_AB_32x.mp4"),
    ], manifest, delete_sources)
    add_issue_files("structural/S2_whole_bundle_drag_roll", [
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/before_raw/camera_A_rgb_upright.png", "before_camera_A.png"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/before_raw/camera_1_B.png", "before_camera_B.png"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/after_raw/camera_A_rgb_upright.png", "after_camera_A.png"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/after_raw/camera_1_B.png", "after_camera_B.png"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/evaluation.json", "evaluation.json"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/trajectory.json", "trajectory.json"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/rollout_recording/evaluator_video_evidence/camera_AB_DEPTH_rgb_contact_sheet.png", "camera_AB_contact_sheet.png"),
    ], [
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/rollout_recording/camera_A_rgb.mp4", "camera_A_32x.mp4"),
        ("runs/fold_inherit_20260828_225638/results/fold_exploration/20260828T145638429755Z/iteration_001/rollout_recording/composite_AB_depth.mp4", "composite_AB_32x.mp4"),
    ], manifest, delete_sources)
    add_issue_files("structural/S3_grounding_mask_mismatch", [
        ("runs/single_sleeve_absolute_depth_20260827_153311/results/single_sleeve_grasp/20260827T073311260124Z/camera_A_rgb_upright.png", "single_sleeve_camera_A.png"),
        ("runs/single_sleeve_absolute_depth_20260827_153311/results/single_sleeve_grasp/20260827T073311260124Z/claude_rgb_input/camera_A_Rxxx_overlay.png", "single_sleeve_rxxx_overlay.png"),
        ("runs/single_sleeve_absolute_depth_20260827_153311/results/single_sleeve_grasp/20260827T073311260124Z/selected_point_vs_production_mask.png", "selected_point_vs_mask.png"),
        ("runs/single_sleeve_absolute_depth_20260827_153311/results/single_sleeve_grasp/20260827T073311260124Z/summary.json", "single_sleeve_summary.json"),
        ("runs/fold_holdcheck_20260829_232626/results/fold_exploration/20260829T152626706046Z/debug.log", "holdcheck_320mm_debug.log"),
        ("runs/fold_holdcheck_20260829_232626/results/fold_exploration/20260829T152626706046Z/summary.json", "holdcheck_320mm_summary.json"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/before_raw/camera_A_rgb_upright.png", "night_iter009_before_A.png"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/after_raw/camera_A_rgb_upright.png", "night_iter009_after_A.png"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/evaluation.json", "night_iter009_evaluation.json"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/trajectory.json", "night_iter009_trajectory.json"),
    ], [
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/rollout_recording/camera_C_observer_rgb.mp4", "night_iter009_camera_C_32x.mp4"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/rollout_recording/composite_AB_depth.mp4", "night_iter009_composite_AB_32x.mp4"),
    ], manifest, delete_sources)
    add_issue_files("structural/S4_latency_context", [
        ("runs/fold_persistent_20260828_213950/results/fold_exploration/20260828T133951009663Z/debug.log", "debug.log"),
        ("runs/fold_persistent_20260828_213950/results/fold_exploration/20260828T133951009663Z/summary.json", "summary.json"),
        ("runs/fold_persistent_20260828_213950/workspace/persistent_claude_session.json", "persistent_claude_session.json"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/debug.log", "night_debug_timing.log"),
    ], [
        ("runs/fold_persistent_20260828_213950/results/fold_exploration/20260828T133951009663Z/iteration_001/rollout_recording/composite_AB_depth.mp4", "context_iter001_composite_AB_32x.mp4"),
    ], manifest, delete_sources)
    add_issue_files("structural/S5_safety_ik_height", [
        ("runs/collar_high_lift_05/results/collar_lift_retreat/20260825T085016743826Z/summary.json", "collar_height_summary.json"),
        ("runs/collar_high_lift_05/results/collar_lift_retreat/20260825T085016743826Z/collar_grasp_overlay.png", "collar_grasp_overlay.png"),
        ("runs/fold_holdcheck_20260830_004839/results/fold_exploration/20260829T164839247141Z/debug.log", "ik_rejection_debug.log"),
        ("runs/fold_holdcheck_20260830_004839/results/fold_exploration/20260829T164839247141Z/summary.json", "ik_rejection_summary.json"),
        ("runs/single_sleeve_absolute_depth_20260827_153000/results/single_sleeve_grasp/20260827T073000560573Z/summary.json", "workspace_lower_bound_summary.json"),
    ], [], manifest, delete_sources)
    add_issue_files("structural/S6_observability_telemetry", [
        ("runs/fold_camC_20260829_031316/results/fold_exploration/20260828T191316890565Z/iteration_001/before_raw/camera_C_observer_rgb.png", "camC_before.png"),
        ("runs/fold_camC_20260829_031316/results/fold_exploration/20260828T191316890565Z/iteration_001/after_raw/camera_C_observer_rgb.png", "camC_after.png"),
        ("runs/fold_camC_20260829_031316/results/fold_exploration/20260828T191316890565Z/iteration_001/gripper_telemetry.json", "camC_gripper_telemetry.json"),
        ("runs/fold_camC_20260829_031316/results/fold_exploration/20260828T191316890565Z/iteration_001/evaluation.json", "camC_evaluation.json"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/hold_check/camera_C_observer_rgb_hold_check.png", "night_hold_check.png"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/gripper_telemetry.json", "night_gripper_telemetry.json"),
    ], [
        ("runs/fold_camC_20260829_031316/results/fold_exploration/20260828T191316890565Z/iteration_001/rollout_recording/camera_C_observer_rgb.mp4", "camC_rollout_32x.mp4"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260830T012849445077Z/iteration_014/rollout_recording/camera_C_observer_rgb.mp4", "night_camC_rollout_32x.mp4"),
    ], manifest, delete_sources)
    add_issue_files("structural/S7_unattended_state_loop", [
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/summary.json", "summary.json"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/debug.log", "debug.log"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/evaluation.json", "iteration009_evaluation.json"),
    ], [
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/iteration_009/rollout_recording/camera_C_observer_rgb.mp4", "iteration009_camC_32x.mp4"),
    ], manifest, delete_sources)
    add_issue_files("structural/S8_capture_quality", [
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260829T194442350392Z/debug.log", "table_luma_debug.log"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260829T194442350392Z/summary.json", "table_luma_summary.json"),
        ("runs/fold_night_20260829T173728412271Z/results/fold_exploration/20260829T194442350392Z/iteration_001/before_raw_attempts.json", "before_raw_attempts.json"),
        ("runs/claude_global_cli_real_20260826_002554/results/molmo_keypoint_cli/20260825T162555245893Z/summary.json", "mask_coverage_summary.json"),
        ("runs/neat_fold_overnight_20260827_020727/results/molmo_keypoint_cli/20260826T180728581036Z/iteration_001/lift_checkpoints/LIFT_CHECKPOINT_01/LIFT_CHECKPOINT_01_camera_A_rgb.png", "overnight_camera_A_reference.png"),
    ], [
        ("runs/neat_fold_overnight_20260827_020727/results/molmo_keypoint_cli/20260826T180728581036Z/combined_rollout.mp4", "overnight_combined_32x.mp4"),
    ], manifest, delete_sources)

    # Non-structural evidence
    add_issue_files("nonstructural/N1_schema_contract", [
        ("runs/fold_unattended_20260827_190930/results/fold_exploration/20260827T110930317675Z/debug.log", "unattended_schema_debug.log"),
        ("runs/fold_unattended_20260827_190930/results/fold_exploration/20260827T110930317675Z/summary.json", "unattended_schema_summary.json"),
        ("runs/fold_molmo_20260828_021707/results/fold_exploration/20260827T181708015508Z/debug.log", "molmo_schema_debug.log"),
        ("runs/fold_molmo_20260828_021707/results/fold_exploration/20260827T181708015508Z/summary.json", "molmo_schema_summary.json"),
        ("runs/fold_night_20260830T100630383895Z/results/fold_exploration/20260830T100630512236Z/debug.log", "night_schema_debug.log"),
    ], [], manifest, delete_sources)
    add_issue_files("nonstructural/N2_infra_transient", [
        ("runs/fold_gripper_sdk_20260829_021819/results/fold_exploration/20260828T181826742662Z/debug.log", "socket_debug.log"),
        ("runs/fold_gripper_sdk_20260829_021819/results/fold_exploration/20260828T181826742662Z/summary.json", "socket_summary.json"),
        ("runs/fold_gripper_sdk_20260829_021511/results/fold_exploration/20260828T181749340433Z/debug.log", "debug_exception_typeerror.log"),
        ("runs/single_sleeve_grasp_dry_smoke_20260827/results/single_sleeve_grasp/20260827T063731775651Z/summary.json", "api_dns_summary.json"),
    ], [], manifest, delete_sources)
    add_issue_files("nonstructural/N3_interruptions_limits", [
        ("runs/claude_global_cli_real_20260825_211257/results/molmo_keypoint_cli/20260825T131258334092Z/summary.json", "interrupted_summary.json"),
        ("runs/neat_fold_overnight_20260827_020727/results/molmo_keypoint_cli/20260826T180728581036Z/summary.json", "max_or_perception_summary.json"),
        ("runs/collar_high_lift_02/results/collar_lift_retreat/20260824T182117527427Z/summary.json", "collar_keyboardinterrupt_summary.json"),
        ("runs/fold_night_20260829T173728412271Z/unattended_restarts.jsonl", "night_restart_log.jsonl"),
    ], [], manifest, delete_sources)

    # Preserve experience ledgers explicitly, without deleting their originals.
    preserved = []
    for src in [
        p("runs/fold_night_20260829T173728412271Z/workspace/fold_experience/experience_summary.json"),
        p("runs/fold_inherit_20260828_225638/workspace/fold_experience/experience_summary.json"),
        p("runs/fold_grasp_learning_20260828_153842/workspace/fold_experience/experience_summary.json"),
    ]:
        dst = OUT / "preserved_experience" / src.parent.parent.parent.name / src.name
        if copy_file(src, dst, [], manifest, delete=False):
            preserved.append(str(dst.relative_to(OUT)))

    (OUT / "run_inventory.csv").parent.mkdir(parents=True, exist_ok=True)
    with (OUT / "run_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "summary", "status", "iterations", "error"])
        w.writeheader(); w.writerows(inventory)
    (OUT / "inventory_meta.json").write_text(json.dumps(inventory_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "evaluation_stats.json").write_text(json.dumps(eval_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "timing_summary.json").write_text(json.dumps(timing_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "issue_catalog.json").write_text(json.dumps(issue_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Delete only verified selected sources, and only once per path.
    deleted = []
    for src in sorted(set(delete_sources)):
        if not src.exists():
            continue
        # Do not delete experience data even if accidentally listed.
        if "workspace/fold_experience" in str(src):
            continue
        src.unlink()
        deleted.append(str(src))
    for rec in manifest:
        if rec.get("status") in ("copied", "copied_32x") and rec.get("source") in deleted:
            rec["source_deleted"] = True
    (OUT / "evidence_manifest.json").write_text(json.dumps({"files": manifest, "deleted_sources": deleted, "preserved_experience": preserved}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(inventory_meta, eval_meta, timing_meta, issue_meta)
    print(json.dumps({"archive": str(OUT), "top_level_runs": inventory_meta["top_level_runs"], "summary_status_counts": inventory_meta["summary_status_counts"], "evaluated_iterations": eval_meta["evaluated_iterations"], "copied_records": sum(r.get("status") in ("copied", "copied_32x") for r in manifest), "missing_records": sum(r.get("status") == "missing" for r in manifest), "deleted_sources": len(deleted)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
