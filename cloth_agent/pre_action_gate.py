"""Independent, pre-execution policy-quality probe. No robot or routing authority."""
from __future__ import annotations

import json
from pathlib import Path
import time

from PIL import Image

from .experiments import one_shot_goal_planner_probe_v1 as wire

GATE_PROMPT = """你是单步衣物展开动作的执行前质量评审器。
输入依次为当前 RGB、目标 RGB、当前图上的具体动作 Pa→Pb 标注；动作是 lift_pull_place，抬升约50毫米后平移并放下。
若具体动作带 grasp_angle_deg，它是当前图中两指连线的开合轴方向：向右0°、向下90°，0–180°，绿色标记表示此方向。
仅抓取前调整方向，夹住后保持姿态；角度为 null 或未提供时沿用 Home 方向。结合角度评估可见布片是否便于夹取。
只判断：这个已经提出的具体动作是否有足够依据直接尝试，以推进衣物展开并接近目标布局。
检查抓住的可见布片与移动方向是否有助解除覆盖，以及是否可能把布片搬到已经展开的区域、增加覆盖或聚拢；不要仅因目标点在空桌面就认定可展开。
TRUST：该具体动作有清楚的视觉依据，值得直接尝试。
DELIBERATE：存在具体可见的策略问题或明显需要进一步分析的布层/方向关系，应交给更仔细的规划。
UNCERTAIN：仅凭这些输入无法可靠判断。
不识别 Graph state，不检索相似历史，不评判 depth、IK 或机器人几何安全；这些由 Host 单独检查。
不提出替代动作，不猜执行结果；只输出严格 JSON 的 decision 字段，不输出理由。"""
GATE_SCHEMA = dict(type="object", additionalProperties=False,
    properties=dict(decision=dict(type="string", enum=["TRUST", "DELIBERATE", "UNCERTAIN"])),
    required=["decision"])


class PreActionGate:
    """One independent direct request. Errors are logged, never repaired.

    ``request_once`` is the existing no-retry transport, injected by the host.
    Only current/goal/overlay pixels and geometric action fields enter the request.
    """
    def __init__(self, folder: Path, headers, request_once, timeout_s=120.):
        self.folder = Path(folder)
        self.headers = headers
        self.request_once = request_once
        self.timeout_s = timeout_s
        self.metrics = {}

    def evaluate(self, current_rgb, goal_rgb, proposed_action):
        started = time.perf_counter()
        status = "ERROR"
        result = {"decision": "UNCERTAIN"}
        self.metrics = dict(mode="SHADOW", affects_execution=False, status="ERROR",
                            model=wire.MODEL, effort=wire.EFFORT, request_attempted=False)
        try:
            with Image.open(current_rgb) as image:
                size = list(image.size)
            wire.validate_plan({k:v for k,v in proposed_action.items() if k!="is_smoothing"}, size)
            if proposed_action["decision"] != "ACTION":
                raise ValueError("PreActionGate only evaluates ACTION proposals")
            overlay = self.folder/"gate_input_overlay.png"
            wire.overlay(current_rgb, proposed_action, overlay, "FastPlanner proposed action")
            # Build a fresh request; never include FastPlanner response IDs or its rationale.
            action = {k: proposed_action[k] for k in ("grasp_px", "target_px", "motion_type")}
            if 'grasp_angle_deg' in proposed_action:
                action['grasp_angle_deg'] = proposed_action['grasp_angle_deg']
            if 'is_smoothing' in proposed_action:
                action['is_smoothing']=proposed_action['is_smoothing']
                action['motion_limits_mm']={'lift_above_surface':10,'max_xy':30} if proposed_action['is_smoothing'] else {'lift_above_surface':50,'max_xy':150}
            body = wire.payload(current_rgb, goal_rgb, size)
            body["instructions"] = GATE_PROMPT
            body["input"][0]["content"][0]["text"] = (
                f"当前原图尺寸 {size[0]}×{size[1]}。图片顺序：current、goal、Pa→Pb overlay。\n"
                "具体动作："+json.dumps(action, ensure_ascii=False))
            body["input"][0]["content"].append(dict(type="input_image",
                image_url=wire.image_data(overlay), detail="high"))
            body["text"]["format"].update(name="pre_action_policy_gate", schema=GATE_SCHEMA)
            wire.save(self.folder/"gate_input_manifest.json", dict(
                mode="SHADOW", model=wire.MODEL, effort=wire.EFFORT,
                image_count=3, current_sha256=wire.sha(current_rgb), goal_sha256=wire.sha(goal_rgb),
                overlay_sha256=wire.sha(overlay), proposed_action=action,
                tools=[], tool_choice="none", retries=0,
                after_supplied=False, execution_supplied=False, history_supplied=False,
                graph_supplied=False, previous_rejection_supplied=False))
            self.metrics["request_attempted"] = True
            value, request_metrics = self.request_once(
                self.folder/"gate_request", body, self.headers, self.timeout_s)
            wire.jsonschema.validate(value, GATE_SCHEMA)
            result = value
            self.metrics["request_wall_s"] = request_metrics["wall_latency_s"]
            status = "VALID"
        except Exception as exc:
            self.metrics["error"] = f"{type(exc).__name__}: {exc}"
            # Synthetic UNCERTAIN is explicitly distinguishable from a model UNCERTAIN.
            self.metrics["fallback"] = "UNCERTAIN; shadow probe failed; original action retained"
        except BaseException:
            status = "INTERRUPTED"
            raise  # User stop is never swallowed by the shadow error handling.
        finally:
            self.metrics.update(status=status, gate_wall_s=time.perf_counter()-started,
                                finished_at=wire.utc())
            wire.save(self.folder/"gate_result.json", result)
            wire.save(self.folder/"gate_status.json", self.metrics)
        return result
