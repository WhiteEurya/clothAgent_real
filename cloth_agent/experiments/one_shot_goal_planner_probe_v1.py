"""One Responses request per recorded query; no Agent, tools, or robot runtime."""
from __future__ import annotations

import argparse
import base64
import csv
import copy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jsonschema
import requests

ROOT = Path("reports/one_shot_goal_planner_probe_v1")
SOURCES = (Path("reports/goal_only_full_episode_v1_restart02"),
           Path("reports/goal_only_full_episode_v1_continue01"))
MODEL, EFFORT = "gpt-6-astra", "high"
ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
CHOICES = ("ONE_SHOT_BETTER", "ORIGINAL_BETTER", "EQUIVALENT", "BOTH_BAD")
# User-provided TCP boundary: data/robot/xarm_boundaries_new.json;
# shared base-frame Y/Z, X unrestricted per config/dual_arm.local.json.
GRIPPER_BOUNDARY_PROMPT = """全局夹爪边界（用户已给定，Arm6 基座坐标，单位毫米）：
-250.578537 <= Y <= 246.447205；Z >= -0.542602；X 未设限，Z 未设上限。
这是夹爪 TCP 的边界，不是图像像素或完整桌面轮廓；未设限不代表任意位置可达。
规划时遵守此边界，避免把衣物继续向边界外拉。仍输出当前 RGB 的 Pa/Pb，由 Host 做坐标转换和原有检查；不要把毫米值当像素或臆造坐标换算。"""
PROMPT = """你是单臂衣物展开的单步规划器。输入第一张是当前 RGB，第二张是目标 RGB。
任务：选择最可能使当前衣物接近目标状态的一个动作。目标是衣物主要部位的展开布局和折叠/遮挡状态接近参考；
允许整体平移或旋转、普通小褶皱，不追求背景或绝对桌面位置一致。
仅支持现有 lift_pull_place：固定 Home 夹爪姿态，抓取可见布料、抬升、平移、放下；抬升约50毫米，平移范围10–150毫米。
Pa 必须落在当前图可见、可抓的布料上；Pb 应在当前可见工作台内，选择有助展开的方向，不猜隐藏衣角或布层。
像素坐标以当前原图左上角为原点，x向右、y向下，必须为整数。不得把目标图坐标直接搬到当前图。
本次没有深度输入；坐标仅为待 Host 检查的像素建议，不声称已通过实测深度、IK或安全检查。
只有当前衣物已达到参考图的主要布局与展开程度才输出 DONE；无法提出有依据的单步动作时输出 ABSTAIN。
ACTION 时 motion_type 为 lift_pull_place；DONE/ABSTAIN 时 grasp_px、target_px、motion_type 全为 null。
只输出指定 JSON，short_reason 最多一句中文。"""
PROMPT += "\n\n" + GRIPPER_BOUNDARY_PROMPT
MULTIVIEW_PROMPT = PROMPT.replace(
    "输入第一张是当前 RGB，第二张是目标 RGB。",
    "输入第一张是 perception 当前 RGB，第二张是同一当前状态的 right view RGB，第三张是目标 RGB。"
) + """\n\n结合两个当前视角判断衣物部位、折叠层次和遮挡；right view 只辅助理解。
尽量抓取清楚可见、可夹住的衣物边缘或角点，优先能带动展开的外露边角，避免抓衣身中部或多层堆叠处。
不要为抓边角而猜测隐藏角点；仅在 right view 可见、perception 无法定位的点不能作为 Pa。
grasp_px 和 target_px 必须全部使用第一张 perception 原图的像素坐标；不得直接使用 right view 或 goal 坐标。
两个当前视角之间只移动相机机械臂，未执行衣物动作；请求期间机械臂正在返回 perception，返回并核验到位后才可能执行。
"""
POINT = {"anyOf": [{"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                   {"type": "null"}]}
SCHEMA = dict(type="object", additionalProperties=False,
    properties=dict(decision=dict(type="string", enum=["ACTION", "DONE", "ABSTAIN"]),
                    grasp_px=POINT, target_px=POINT,
                    motion_type={"type": ["string", "null"], "enum": ["lift_pull_place", None]},
                    short_reason=dict(type="string")),
    required=["decision", "grasp_px", "target_px", "motion_type", "short_reason"])

GRASP_ANGLE_PROMPT = """抓取前允许调整夹爪方向，夹住后保持该方向完成抬升、平移、放下，不允许持布旋转。
grasp_angle_deg 表示第一张 perception 图中两指之间的连线（夹爪开合轴），不是手指长轴或拉动方向。
以图像向右为 0°，向下为 90°，图像中顺时针增加，取 0–180°；0° 与 180° 等价。
根据 Pa 附近可见边角选择便于夹住布料的方向；无法可靠判断或无需调整时填 null，沿用 Home 方向。
Host 用实测抓点深度和相机标定转换角度，选择距 Home 最近的等价方向，只改变桌面平面内方向，保留原倾角。
不要把这个图像角度当作机器人 yaw 或 joint6 的绝对角度。DONE/ABSTAIN 时 grasp_angle_deg 必须为 null。
"""
ROTATING_MULTIVIEW_PROMPT = MULTIVIEW_PROMPT.replace(
    "固定 Home 夹爪姿态，", "抓取前可调整夹爪方向，夹住后保持姿态，") + "\n" + GRASP_ANGLE_PROMPT
ROTATION_SCHEMA = copy.deepcopy(SCHEMA)
ROTATION_SCHEMA['properties']['grasp_angle_deg'] = {
    'type': ['number', 'null'], 'minimum': 0, 'maximum': 180}
ROTATION_SCHEMA['required'].append('grasp_angle_deg')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def image_data(path):
    return "data:image/png;base64,"+base64.b64encode(Path(path).read_bytes()).decode()


def payload(current, target, size, *, right_view=None, allow_grasp_rotation=False):
    if allow_grasp_rotation and right_view is None:
        raise ValueError('rotating one-shot requires both current views')
    images = [current, target] if right_view is None else [current, right_view, target]
    labels = "第一张 current，第二张 goal。" if right_view is None else (
        "第一张 perception current（唯一 Pa/Pb 坐标基准），第二张 right view current，第三张 goal。")
    return dict(model=MODEL, instructions=(ROTATING_MULTIVIEW_PROMPT if allow_grasp_rotation
                else PROMPT if right_view is None else MULTIVIEW_PROMPT),
        input=[dict(role="user", content=[
            dict(type="input_text", text=f"当前 perception 图尺寸 {size[0]}×{size[1]} 像素。{labels}"),
            *[dict(type="input_image", image_url=image_data(path), detail="high") for path in images]])],
        tools=[], tool_choice="none", parallel_tool_calls=False,
        reasoning=dict(effort=EFFORT),
        text=dict(format=dict(type="json_schema", name="unfold_single_step", strict=True,
                             schema=ROTATION_SCHEMA if allow_grasp_rotation else SCHEMA)),
        store=False, stream=True)


def validate_plan(plan, size, *, require_grasp_rotation=False):
    rotated = require_grasp_rotation or 'grasp_angle_deg' in plan
    jsonschema.validate(plan, ROTATION_SCHEMA if rotated else SCHEMA)
    if rotated:
        angle = plan['grasp_angle_deg']
        if angle is not None and (not math.isfinite(angle) or plan['decision'] != 'ACTION'):
            raise ValueError('grasp angle must be finite for ACTION and null otherwise')
    if plan["decision"] == "ACTION":
        if plan["motion_type"] != "lift_pull_place":
            raise ValueError("ACTION primitive invalid")
        for name in ("grasp_px", "target_px"):
            point = plan[name]
            if point is None or not (0 <= point[0] < size[0] and 0 <= point[1] < size[1]):
                raise ValueError("ACTION coordinate invalid: "+name)
    elif any(plan[k] is not None for k in ("grasp_px", "target_px", "motion_type")):
        raise ValueError("non-ACTION coordinates must be null")
    # Validate, never repair or send a second request.
    if not plan["short_reason"].strip():
        raise ValueError("empty short_reason")
    import re
    if len([s for s in re.split(r"[。！？!?]+", plan["short_reason"]) if s.strip()]) > 1:
        raise ValueError("short_reason exceeds one sentence")


def overlay(path, plan, destination, title):
    with Image.open(path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    draw.rectangle((0, 0, image.width, 35), fill="#101828")
    draw.text((10, 4), title+" | "+plan.get("decision", "INVALID"), font=font, fill="white")
    if plan.get("decision") == "ACTION" and plan.get("grasp_px") and plan.get("target_px"):
        a, b = plan["grasp_px"], plan["target_px"]
        draw.line((*a, *b), fill="#ffd43b", width=5)
        angle = math.atan2(b[1]-a[1], b[0]-a[0])
        draw.polygon([tuple(b), (b[0]-20*math.cos(angle-.4), b[1]-20*math.sin(angle-.4)),
                      (b[0]-20*math.cos(angle+.4), b[1]-20*math.sin(angle+.4))], fill="#ffd43b")
        for point, color, label in ((a, "#f43f5e", "Pa"), (b, "#06b6d4", "Pb")):
            x, y = point
            draw.ellipse((x-8, y-8, x+8, y+8), fill=color, outline="white", width=2)
            draw.text((x+10, y+8), label, font=font, fill=color, stroke_width=2, stroke_fill="black")
        if plan.get('grasp_angle_deg') is not None:
            theta = math.radians(plan['grasp_angle_deg'])
            axis = np.array([math.cos(theta), math.sin(theta)])
            cross = np.array([-axis[1], axis[0]])
            ends = [np.asarray(a)+sign*30*axis for sign in (-1, 1)]
            draw.line([tuple(p) for p in ends], fill='#80ff80', width=3)
            for p in ends:
                draw.line([tuple(p-12*cross), tuple(p+12*cross)], fill='#80ff80', width=5)
            draw.text((10, 40), f"Jaw opening axis: {plan['grasp_angle_deg']:g} deg (image)",
                      font=font, fill='#80ff80', stroke_width=2, stroke_fill='black')
    image.save(destination)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    queries, hashes = [], {}
    target_sha = None
    for segment, source in enumerate(SOURCES):
        summary = read(source/"episode_summary.json")
        goal = source/"goal/target.png"
        target_sha = target_sha or sha(goal)
        if sha(goal) != target_sha:
            raise ValueError("goal images differ between segments")
        hashes[str(goal.resolve())] = sha(goal)
        for item in summary["rounds"]:
            folder = source/item["round"]
            if not (folder/"plan.json").exists() or not (folder/"agent_events/metrics.json").exists():
                continue
            query_id = ("initial_" if segment == 0 else "continue_")+item["round"]
            dest = out/"queries"/query_id
            dest.mkdir(parents=True)
            prep_start = time.perf_counter()
            original = read(folder/"plan.json")
            current = folder/"before/rgb.png"
            request = read(folder/"agent_events/request.json")
            if sha(current) != request["current_observation_sha256"]:
                raise ValueError("query differs from original Agent observation")
            for name, source_path in (("current.png", current), ("goal.png", goal)):
                with Image.open(source_path) as im:
                    im.convert("RGB").save(dest/name)
            with Image.open(dest/"current.png") as im:
                size = list(im.size)
            shutil.copyfile(folder/"plan.json", dest/"original_plan.json")
            shutil.copyfile(folder/"agent_events/metrics.json", dest/"original_metrics.json")
            original_prompt = folder/"agent_events/prompt.txt"
            shutil.copyfile(original_prompt, dest/"original_prompt.txt")
            body = payload(dest/"current.png", dest/"goal.png", size)
            # All serialization and image loading done before the single network call.
            (dest/"request_payload.json").write_bytes(json.dumps(body, ensure_ascii=False).encode())
            (dest/"prompt.txt").write_text(PROMPT+"\n"+body["input"][0]["content"][0]["text"])
            overlay(dest/"current.png", original, dest/"original_overlay.png", "ORIGINAL")
            meta = dict(query_id=query_id, source_episode=str(source.resolve()), source_round=item["round"],
                source_rgb=str(current.resolve()), original_decision=original["decision"],
                original_execution=item["execution"]["status"],
                original_wall_latency_s=read(folder/"agent_events/metrics.json")["wall_latency_s"],
                original_prompt_chars=len(original_prompt.read_text()),
                image_size_px=size, input_image_count=2,
                prompt_chars=len((dest/"prompt.txt").read_text()),
                prompt_utf8_bytes=len((dest/"prompt.txt").read_bytes()),
                schema_chars=len(json.dumps(SCHEMA, ensure_ascii=False)),
                request_payload_bytes=(dest/"request_payload.json").stat().st_size,
                request_payload_sha256=sha(dest/"request_payload.json"),
                current_png_sha256=sha(dest/"current.png"), goal_png_sha256=sha(dest/"goal.png"),
                host_preparation_s=time.perf_counter()-prep_start)
            save(dest/"input_manifest.json", meta)
            queries.append(meta)
            for path in (current, folder/"plan.json", folder/"agent_events/metrics.json", original_prompt):
                hashes[str(path.resolve())] = sha(path)
    if len(queries) != 10:
        raise ValueError("expected ten actual paired queries, including rejected ACTION and final DONE")
    save(out/"query_manifest.json", queries)
    save(out/"source_hashes.json", hashes)
    save(out/"protocol.json", dict(experiment="one_shot_goal_planner_probe_v1", model=MODEL, effort=EFFORT,
        created_at=utc(), endpoint=ENDPOINT, transport="direct HTTPS Responses using existing Codex ChatGPT account auth",
        model_calls_per_query=1, http_attempts_per_query=1, retries=0, tools=[], tool_choice="none",
        original_agent_rerun=False, robot=False, graph=False, trajectory_memory=False,
        query_count=10, schema=SCHEMA, prompt=PROMPT,
        image_order=["current RGB", "same goal RGB"], source_code_sha256=sha(__file__),
        latency="time from prebuilt HTTP request dispatch to completed response; includes transport, excludes host preparation",
        quality="operator review only; original plan is not ground truth",
        close_to_20_seconds="median <=20 s meets target; 20<median<=25 s near target; >25 s does not meet near-20 criterion",
        confounds=["Original Agent had RGB-D + own episode history and tools; one-shot has only RGB pair and fixed constraints.",
                   "Original latency includes CLI startup and Agent orchestration; direct request excludes those by design.",
                   "Calls occur at different times; service load/cache/tier can differ. No causal isolation or matched repeated trials."],
        docs=["https://developers.openai.com/api/docs/guides/structured-outputs",
              "https://developers.openai.com/api/reference/resources/responses/methods/create"]))
    shutil.copyfile(__file__, out/"experiment_source.py")
    save(out/"human_review.json", dict(experiment="one_shot_goal_planner_probe_v1", reviewer=None,
        ratings={q["query_id"]: dict(choice=None, note="") for q in queries}))
    print("PREPARED 10 queries; no model calls", flush=True)


def auth_headers():
    # Use the already authorized account; never serialize/log any auth value.
    auth = read(Path.home()/".codex/auth.json")
    tokens = auth.get("tokens", {})
    if auth.get("auth_mode") != "chatgpt" or not tokens.get("access_token"):
        raise RuntimeError("existing ChatGPT account auth unavailable; no credential fallback")
    headers = {"Authorization": "Bearer "+tokens["access_token"], "Content-Type": "application/json",
               "Accept": "text/event-stream", "originator": "codex_cli_rs",
               "User-Agent": "one_shot_goal_planner_probe_v1"}
    if tokens.get("account_id"):
        headers["ChatGPT-Account-ID"] = tokens["account_id"]
    return headers


def parse_events(events):
    created = {e["response"]["id"] for e in events if e.get("type") == "response.created"}
    completed = [e["response"] for e in events if e.get("type") == "response.completed"]
    tool_events = [e for e in events if "function_call" in e.get("type", "") or
                   e.get("item", {}).get("type") in ("function_call", "custom_tool_call", "web_search_call",
                                                    "file_search_call", "code_interpreter_call")]
    if len(created) != 1 or len(completed) != 1 or completed[0]["id"] not in created or tool_events:
        raise ValueError("not exactly one completed tool-free model response")
    response = completed[0]
    if response.get("status") != "completed":
        raise ValueError("response not completed")
    if any(item.get("type") not in ("message", "reasoning") for item in response.get("output", [])):
        raise ValueError("unexpected tool/action output")
    text = "".join(c.get("text", "") for item in response.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")
    return response, text


def call_once(folder, meta, headers):
    # The persistent marker forbids repeating a query after failure or interruption.
    body = (folder/"request_payload.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != meta["request_payload_sha256"]:
        raise ValueError("prepared request changed")
    with (folder/"attempt.json").open("x") as f:
        json.dump(dict(started_at=utc(), http_attempt=1, retries=0), f)
    result = dict(status="FAILED", model=MODEL, effort=EFFORT, model_response_count=0,
                  http_attempts=1, retry_count=0, tool_calls=0, input_image_count=2)
    events = []
    start = time.perf_counter()
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            with session.post(ENDPOINT, headers=headers, data=body, stream=True,
                              timeout=(30, 600), allow_redirects=False) as response:
                result["http_status"] = response.status_code
                result["request_id"] = response.headers.get("x-request-id")
                if response.status_code != 200:
                    (folder/"raw_http_error.txt").write_text(response.text)
                    raise RuntimeError(f"HTTP {response.status_code}; no retry")
                with (folder/"raw_response.sse").open("wb") as raw:
                    for line in response.iter_lines():
                        raw.write(line+b"\n")
                        raw.flush()
                        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                            events.append(json.loads(line[6:]))
                            if events[-1].get("type") in ("response.completed", "response.failed"):
                                break
        result["wall_latency_s"] = time.perf_counter()-start
        result["model_response_count"] = len({e["response"]["id"] for e in events if e.get("type") == "response.created"})
        complete, text = parse_events(events)
        save(folder/"raw_response.json", complete)
        (folder/"raw_output.txt").write_text(text)
        result["usage"] = complete.get("usage")
        result["returned_model"] = complete.get("model")
        result["service_tier"] = complete.get("service_tier")
        plan = json.loads(text)
        save(folder/"one_shot_plan.json", plan)
        validate_plan(plan, meta["image_size_px"])
        overlay(folder/"current.png", plan, folder/"one_shot_overlay.png", "ONE SHOT")
        result["status"] = "VALID"
        result["decision"] = plan["decision"]
    except BaseException as exc:
        result["wall_latency_s"] = result.get("wall_latency_s", time.perf_counter()-start)
        result["error"] = f"{type(exc).__name__}: {exc}"
        # This stores no request headers, cookies, bearer token or account identifier.
        if isinstance(exc, KeyboardInterrupt):
            result["status"] = "INTERRUPTED"
            save(folder/"result.json", result)
            raise
    finally:
        save(folder/"result.json", result)
    print(meta["query_id"]+" "+result["status"]+f" {result['wall_latency_s']:.2f}s", flush=True)
    return result


def run(out):
    if read(out/"protocol.json")["source_code_sha256"] != sha(__file__):
        raise ValueError("source changed after preparation")
    queries = read(out/"query_manifest.json")
    headers = auth_headers()
    for meta in queries:
        folder = out/"queries"/meta["query_id"]
        if (folder/"attempt.json").exists():
            continue  # Never repeat. Unfinished attempts remain missing/failed.
        result = call_once(folder, meta, headers)
        if result.get("http_status") != 200:
            print("TRANSPORT STOP; remaining queries not attempted", flush=True)
            break
    report(out)


def stats(values):
    return dict(n=len(values), median=float(np.median(values)), mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95))) if values else dict(n=0, median=None, mean=None, p95=None)


def report(out):
    queries = read(out/"query_manifest.json")
    rows, cards = [], []
    for q in queries:
        folder = out/"queries"/q["query_id"]
        result = read(folder/"result.json") if (folder/"result.json").exists() else {}
        original = read(folder/"original_plan.json")
        valid = result.get("status") == "VALID"
        plan = read(folder/"one_shot_plan.json") if valid else None
        both_action = valid and plan["decision"] == original["decision"] == "ACTION"
        rows.append(dict(query_id=q["query_id"], status=result.get("status", "NOT_COMPLETED"),
            original_decision=original["decision"], original_execution=q["original_execution"],
            one_shot_decision=plan["decision"] if plan else None,
            original_wall_s=q["original_wall_latency_s"], one_shot_wall_s=result.get("wall_latency_s"),
            speedup=q["original_wall_latency_s"]/result["wall_latency_s"] if valid else None,
            decision_agreement=int(plan["decision"] == original["decision"]) if valid else None,
            grasp_pixel_distance=float(np.linalg.norm(np.array(original["grasp_px"])-plan["grasp_px"])) if both_action else None,
            target_pixel_distance=float(np.linalg.norm(np.array(original["target_px"])-plan["target_px"])) if both_action else None,
            image_count=2, prompt_chars=q["prompt_chars"]))
        cards.append(dict(query_id=q["query_id"], valid=valid, original=original, one_shot=plan,
                          error=result.get("error")))
    valid_rows = [r for r in rows if r["status"] == "VALID"]
    old_stats = stats([r["original_wall_s"] for r in valid_rows])
    new_stats = stats([r["one_shot_wall_s"] for r in valid_rows])
    review = read(out/"human_review.json")
    counts = {choice: sum(r["choice"] == choice for r in review["ratings"].values()) for choice in CHOICES}
    summary = dict(query_count=len(queries), valid_one_shot_count=len(valid_rows),
        failures_or_unfinished=len(queries)-len(valid_rows), one_shot_latency_s=new_stats,
        original_paired_latency_s=old_stats, original_all_queries_latency_s=stats([r["original_wall_s"] for r in rows]),
        mean_latency_speedup=old_stats["mean"]/new_stats["mean"] if valid_rows else None,
        median_latency_speedup=old_stats["median"]/new_stats["median"] if valid_rows else None,
        paired_speedup_distribution=stats([r["speedup"] for r in valid_rows]),
        decision_agreement=float(np.mean([r["decision_agreement"] for r in valid_rows])) if valid_rows else None,
        grasp_pixel_distance=stats([r["grasp_pixel_distance"] for r in rows if r["grasp_pixel_distance"] is not None]),
        target_pixel_distance=stats([r["target_pixel_distance"] for r in rows if r["target_pixel_distance"] is not None]),
        human_quality=dict(counts=counts, rated=sum(counts.values()), total=len(queries),
                           pending=len(queries)-sum(counts.values()), preserved_quality=None),
        near_20_seconds=(new_stats["median"] <= 25) if valid_rows else None,
        at_or_below_20_seconds=sum(r["one_shot_wall_s"] <= 20 for r in valid_rows),
        original_is_ground_truth=False, input_source_hashes_unchanged=all(
            sha(path) == value for path, value in read(out/"source_hashes.json").items()))
    action_rows = [r for r in valid_rows if r["original_decision"] == "ACTION"]
    summary["original_action_queries_only"] = dict(
        one_shot=stats([r["one_shot_wall_s"] for r in action_rows]),
        original=stats([r["original_wall_s"] for r in action_rows]))
    save(out/"summary.json", summary)
    with (out/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_review(out, cards)
    lines = ["# one_shot_goal_planner_probe_v1", "",
        "本实验每 query 只发送一次 direct HTTPS Responses 请求：gpt-6-astra/high、两张 RGB、tools=[]、"
        "tool_choice=none、严格 JSON Schema、无 retry/repair/第二次模型调用。没有启动原 Agent、连接机器人或执行动作。", "",
        "**能否接近 20 秒且保持动作质量？**", ""]
    if valid_rows:
        lines += [f"- 速度：one-shot median **{new_stats['median']:.2f} s**，mean **{new_stats['mean']:.2f} s**，"
                  f"p95 **{new_stats['p95']:.2f} s**；{summary['at_or_below_20_seconds']}/{len(valid_rows)} 次 ≤20 s。"
                  + ("达到预先定义的接近 20 秒口径（median≤25 s）。" if summary["near_20_seconds"] else
                     "未达到预先定义的接近 20 秒口径（median≤25 s）。"),
                  f"- 质量：目前人工已评 {sum(counts.values())}/{len(queries)}，"
                  "未评不能当作 EQUIVALENT；像素差与 decision agreement 不能替代动作质量。",
                  "- 因此在人工评审完成前，不能确认“保持动作质量”。", "",
                  "|配对 latency|median s|mean s|p95 s|", "|---|---:|---:|---:|",
                  f"|One-shot|{new_stats['median']:.2f}|{new_stats['mean']:.2f}|{new_stats['p95']:.2f}|",
                  f"|原 Agent（保存值）|{old_stats['median']:.2f}|{old_stats['mean']:.2f}|{old_stats['p95']:.2f}|",
                  "", f"均值之比 speedup：**{summary['mean_latency_speedup']:.2f}×**；"
                  f"中位数之比：**{summary['median_latency_speedup']:.2f}×**。"
                  "单 query 比值另见 metrics.csv；失败/未完成调用不计入成功配对均值，但在下表全部保留。"]
    else:
        lines += ["尚无有效 one-shot 结果，无法回答速度与质量。"]
    lines += ["", f"样本：{len(queries)} queries；有效 one-shot {len(valid_rows)}，失败/未完成 "
              f"{len(queries)-len(valid_rows)}。原实验主段 3 轮与续跑 7 轮全部纳入，包括原深度拒绝 ACTION 和最终 DONE；"
              "排除从人工摆平状态直接 DONE 的零动作试跑。", "",
              "|Query|原 decision|One-shot|原 s|One-shot s|Pa 差 px|Pb 差 px|状态|",
              "|---|---|---|---:|---:|---:|---:|---|"]
    def number(value):
        return "—" if value is None else f"{value:.2f}"
    for r in rows:
        lines.append(f"|{r['query_id']}|{r['original_decision']}|{r['one_shot_decision'] or '—'}|"
                     f"{number(r['original_wall_s'])}|{number(r['one_shot_wall_s'])}|"
                     f"{number(r['grasp_pixel_distance'])}|{number(r['target_pixel_distance'])}|{r['status']}|")
    lines += ["", f"Decision agreement：{summary['decision_agreement']}。Pa/Pb 差只在双方均 ACTION 时计算，"
              "是原图坐标欧氏距离，不是 grounding error；原方案不是 ground truth。", "",
              "## 人工评审", "", "[打开本地评审页](review.html)。每轮选择 ONE_SHOT_BETTER / ORIGINAL_BETTER / "
              "EQUIVALENT / BOTH_BAD，可备注动作方向、抓点/落点和 DONE/ABSTAIN 是否合理。"
              "浏览器本地自动保存，并可导出 JSON；不自动写回报告目录。导出后用以下命令汇总：", "",
              "```bash", "/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.one_shot_goal_planner_probe_v1 "
              "review --ratings /path/to/exported_review.json", "```", "",
              "|人工判断|数量|", "|---|---:|"]
    lines += [f"|{choice}|{counts[choice]}|" for choice in CHOICES]
    lines += [f"|PENDING|{summary['human_quality']['pending']}|", "", "## 输入、计时和限制", "",
        "- Host 在调用前准备所有 PNG、prompt、schema 和完整 request body；每个 query 只有 current + goal 两图，"
        "原 plan、原 rationale、历史及深度均不进入请求。prompt 长度、字节数和图片尺寸见各 input_manifest.json。",
        "- Prompt 保留原目标布局/展开标准、允许旋转和普通褶皱、可见布料抓点、工作台落点、原 lift_pull_place "
        "约束与 DONE/ABSTAIN 条件；删去文件、shell、检索、crop/view、历史浏览和 Agent workflow 指令。",
        "- 使用现有 Codex 账号对应 Responses 服务，直接 HTTP 调用，没有 Codex CLI 启动、Agent 循环或工具。"
        "鉴权值不进入输出文件。HTTP/模型错误保留且不重试。",
        "- wall latency 包含模型服务与网络传输，不包含已完成的 Host 准备。原 Agent wall latency 是已保存的 "
        "CLI/Agent 总耗时；两者范围差异是待测 workflow 移除的一部分。",
        "- **并非严格只改变 orchestration 的消融**：原 Agent 还可用 RGB-D 和自己的已执行历史，"
        "本次按要求只给 RGB pair；prompt 内容和输入信息也减少。服务负载、缓存及调用时间不配对。"
        "不能把所有速度/质量变化单独归因于工具。",
        "- 只有单个 episode 的 10 个相关 observation，每 query 一次请求。人工画面评审不等同于机器人执行成功率。",
        "- 原 9 个 ACTION query 的独立 latency 子集见 summary.json；其中 1 个原动作被 Host 拒绝，未把它当作正确答案。",
        "", "## 文件", "",
        "query_manifest.json、protocol.json、summary.json、metrics.csv、human_review.json；"
        "queries/ 下每轮保存 original_plan、one_shot_plan、双 action overlay、输入图片/长度、"
        "单次调用计时和 raw_response.sse/json。无效输出保留 raw，未修复。",
        "", "接口参考：[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、"
        "[Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)。", ""]
    (out/"summary.md").write_text("\n".join(lines))


def make_review(out, cards):
    chunks = []
    for card in cards:
        qid = card["query_id"]
        folder = "queries/"+qid
        options = '<option value="">待评审</option>'+''.join(
            f'<option value="{choice}">{choice}</option>' for choice in CHOICES)
        one = f'<img src="{folder}/one_shot_overlay.png">' if card["valid"] else '<p class="bad">无有效 one-shot 输出</p>'
        chunks.append(f"""<section data-id="{qid}"><h2>{qid}</h2>
<div class="images"><figure><figcaption>当前原图</figcaption><img src="{folder}/current.png"></figure>
<figure><figcaption>目标图</figcaption><img src="{folder}/goal.png"></figure></div>
<div class="images"><figure><figcaption>原 Agent</figcaption><img src="{folder}/original_overlay.png">
<pre>{html.escape(json.dumps(card["original"],ensure_ascii=False,indent=2))}</pre></figure>
<figure><figcaption>One-shot</figcaption>{one}<pre>{html.escape(json.dumps(card["one_shot"],ensure_ascii=False,indent=2))}</pre></figure></div>
<label>动作质量比较 <select>{options}</select></label>
<textarea placeholder="备注：动作是否合理、Pa/Pb、DONE/ABSTAIN 判断等"></textarea></section>""")
    # No server, remote asset, API or runtime connection; standalone offline review artifact.
    script = """
const KEY='one_shot_goal_planner_probe_v1.review.v1';
let ratings={};try{ratings=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ratings))}catch(e){} update()}
function update(){let n=Object.values(ratings).filter(x=>x.choice).length;document.querySelector('#progress').textContent=n+'/10 已评'}
document.querySelectorAll('section').forEach(s=>{let id=s.dataset.id,sel=s.querySelector('select'),note=s.querySelector('textarea');
sel.value=ratings[id]?.choice||'';note.value=ratings[id]?.note||'';
function change(){ratings[id]={choice:sel.value||null,note:note.value};persist()}
sel.addEventListener('change',change);note.addEventListener('input',change);});
document.querySelector('#export').onclick=()=>{const all={};document.querySelectorAll('section').forEach(s=>all[s.dataset.id]=ratings[s.dataset.id]||{choice:null,note:''});
const obj={experiment:'one_shot_goal_planner_probe_v1',reviewer:document.querySelector('#reviewer').value||null,exported_at:new Date().toISOString(),ratings:all};
const url=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='one_shot_goal_planner_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};update();
"""
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One-shot 与原 Agent：动作评审</title><style>
body{font-family:system-ui,sans-serif;max-width:1440px;margin:auto;background:#f3f5f8;color:#142033;padding:20px}
header{background:#142033;color:white;padding:24px;border-radius:14px}section{background:white;padding:22px;margin:24px 0;border-radius:14px}
.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:10px 0}figcaption{font-weight:700;margin-bottom:8px}
img{width:100%;border-radius:8px}pre{white-space:pre-wrap;font-size:13px;background:#f3f5f8;padding:12px}
button,select,input,textarea{font:inherit;padding:10px;border:1px solid #bbb;border-radius:6px}
textarea{display:block;width:96%;height:65px;margin-top:14px}button{cursor:pointer;background:#2563eb;color:white}.bad{color:#b42318}
@media(max-width:720px){.images{grid-template-columns:1fr}}.toolbar{position:sticky;top:0;background:#fff;padding:12px;border-bottom:2px solid #2563eb;z-index:2}
</style><header><h1>One-shot 与原 Agent：单步动作评审</h1>
<p>原 plan 不是标准答案。请根据当前图和目标图，比较动作、抓点/落点以及 DONE/ABSTAIN 是否合理。红点 Pa，蓝点 Pb。</p>
<p>所有动作均为离线建议，本页面不连接机器人。未选择的项目保持待评审。浏览器本地保存，导出 JSON 后才能汇总到实验报告。</p></header>
<div class="toolbar"><span id="progress"></span> <input id="reviewer" placeholder="评审者（可选）"> <button id="export">导出评审 JSON</button></div>"""
    (out/"review.html").write_text(page+"\n".join(chunks)+"<script>"+script+"</script></html>")


def import_review(out, path):
    submitted = read(path)
    ids = {q["query_id"] for q in read(out/"query_manifest.json")}
    if submitted.get("experiment") != "one_shot_goal_planner_probe_v1" or set(submitted.get("ratings", {})) != ids:
        raise ValueError("review experiment/query set mismatch")
    for rating in submitted["ratings"].values():
        if rating.get("choice") not in (*CHOICES, None) or not isinstance(rating.get("note", ""), str):
            raise ValueError("invalid human label")
    save(out/"human_review.json", submitted)
    report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report", "review"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--ratings", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage == "prepare":
        prepare(out)
    elif args.stage == "run":
        run(out)
    elif args.stage == "review":
        if args.ratings is None:
            parser.error("--ratings required")
        import_review(out, args.ratings)
    else:
        report(out)


if __name__ == "__main__":
    main()
