"""Goal-image-only free exploration, with the existing collection execution path."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from . import trajectory_memory_full_episode_v1 as shared
from . import trajectory_memory_agent_probe as visuals

ROOT = Path("reports/goal_only_full_episode_v1")
DEFAULT_GOAL = Path("reports/trajectory_memory_full_episode_v1_collection_aligned/round_006/after/rgb.png")
PROMPT = """你是单臂真实衣物操作的自由探索 Agent。本实验没有任何人工示范或历史 trajectory memory。
附图第一张 /work/query.png 是当前观察，第二张 /goal/target.png 是唯一目标状态。
完整当前 RGB-D 在 /observation：world_xyz_mm.npy 与当前 RGB 对齐，单位 World 毫米，NaN 表示缺深度。
/work/context.json 给出图大小、现用 primitive 和采集 runtime 的实际设置。
/episode 只包含本次探索此前已执行的 before/action/after，可用于判断自己的动作效果和避免重复；
没有上一场实验的过程、人工动作或未来观察。目标图只有 RGB，没有目标动作或目标深度。

任务：自主观察当前衣物与目标衣物的差异，选择最可能使当前衣物接近目标状态的一个动作。
目标是衣物主要部位的展开布局和折叠/遮挡状态接近参考；允许整体平移或旋转、普通小褶皱，
不追求背景、人物、像素或绝对桌面位置一致。目标不是一条固定动作序列，请用每步真实反馈决定下一步。
不使用 Graph、预定义状态匹配或示范检索，不用脆弱像素 correspondence 直接搬运目标图坐标。
可以看图、写 Python/Pillow/NumPy 分析/标注脚本。每次只输出一个在当前图重新 grounding 的 Pa/Pb。
ACTION 只支持现有 lift_pull_place；Pa/Pb 必须是当前原图整数像素。Host 沿用采集脚本的固定
Home 夹爪姿态、抓取/抬升/移动/放下 primitive；不能输出新 motion 代码或特殊动作。
抓点必须有可见布料依据；目标点应在当前工作台内有实测深度，不猜隐藏衣角或布层。
不得在观察无变化时重复同一动作。夹爪 feedback 不能可靠证明 acquisition，
未确认时保持 UNKNOWN，不将抓空或无视觉变化直接认定为 cloth action NO_PROGRESS。
只有当前衣物已达到参考图所表达的主要布局和展开程度才返回 DONE；
无法提出有依据的单步动作时 ABSTAIN。DONE/ABSTAIN 的 Pa、Pb、motion_type 均为 null。
supporting_experiences 必须为空数组，本条件没有历史经验可引用。
禁止训练/fine-tune、联网、连接机器人、创建 Graph、修改目标、查看其他实验或未来观察。
Host 才能执行动作。临时文件写 /work；/tmp 不保证保留。
最终只输出严格 schema JSON，short_rationale 用最多几句话中文说明当前差异和动作目的。
"""


def prepare_goal(root, goal_path, unused_scope):
    if not goal_path.is_file():
        raise ValueError(f"goal image missing: {goal_path}")
    visuals.copy_rgb(goal_path, root/"goal/target.png")
    manifest = dict(condition="GOAL_ONLY_NO_TRAJECTORY_MEMORY", frozen_unix_s=time.time(),
                    memory_steps=0, memory_sessions=0, records=[],
                    files={"goal/target.png": shared.digest(root/"goal/target.png")},
                    goal_pixel_sha256=visuals.pixel_digest(root/"goal/target.png"))
    # The shared loop uses this manifest filename for immutable input checks;
    # no memory directory or historical records are created or mounted.
    shared.save(root/"memory_manifest.json", manifest)
    shared.save(root/"goal_manifest.json", manifest)
    shared.save(root/"private/goal_source.json", dict(source=str(goal_path.resolve()),
                source_sha256=shared.digest(goal_path), exposed_to_agent=False))
    shared.save(root/"private/source_hashes.json", {str(goal_path.resolve()): shared.digest(goal_path)})
    return manifest


def plan_round(root, folder, before, history, manifest, timeout):
    if (root/"memory").exists() or manifest["records"]:
        raise ValueError("goal-only condition must not contain historical memory")
    plan, metrics = shared.agent_round(root, folder, before, history, manifest, timeout,
                                       prompt=PROMPT, memory_enabled=False, goal_directory=root/"goal")
    if plan["supporting_experiences"] or metrics["historical_source_images_read"]:
        raise ValueError("historical evidence in goal-only condition")
    if not metrics["goal_image_file_read"]:
        raise ValueError("goal image was not read by the isolated Agent process")
    return plan, metrics


def picture(path, size=(420, 236)):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail(size)
        return im


def render(root, summary):
    summary.update(condition="GOAL_ONLY_NO_TRAJECTORY_MEMORY",
                   target_image="goal/target.png",
                   target_sha256=shared.digest(root/"goal/target.png"),
                   target_unchanged=summary.pop("frozen_memory_unchanged", None),
                   human_target_reached=None)
    shared.save(root/"episode_summary.json", summary)
    review = shared.read(root/"human_review.json")
    review.update(target_reached=None, target_image="goal/target.png")
    shared.save(root/"human_review.json", review)
    lines = ["# goal_only_full_episode_v1", "",
             f"终止：**{summary['termination_reason']}**；{summary['termination_detail']}", "",
             "条件：只有目标 RGB、当前 RGB-D 和本次已执行历史；无人工 trajectory memory、无上一实验过程。",
             f"模型：{shared.MODEL} / {shared.EFFORT}；动作 primitive、runtime 设置与采集脚本一致。",
             f"Rounds：{summary['total_rounds']}；展开动作：{summary['physical_actions']}；"
             f"初始化动作：{summary['initialization_physical_actions']}。",
             f"总时长 {summary['episode_wall_s']:.1f} 秒；Agent {summary['total_agent_latency_s']:.1f} 秒；"
             f"机器人执行及回观测位姿 {summary['total_robot_execution_s']:.1f} 秒。",
             "**人工目标达成 / FOLD_READY 均未判定。Agent DONE 只表示其认为达到参考目标。**", "",
             "![target](goal/target.png)", "",
             "|Round|Decision|Execution|GPT s|Robot s|",
             "|---|---|---|---:|---:|"]
    panels, last_after = [], None
    for item in summary["rounds"]:
        folder = root/item["round"]
        plan = shared.read(folder/"plan.json") if (folder/"plan.json").exists() else None
        timing = item.get("timing", {})
        lines.append(f"|{item['round']}|{plan['decision'] if plan else 'NO_VALID_PLAN'}|"
                     f"{item['execution']['status']}|{timing.get('agent_wall_s',0):.1f}|"
                     f"{timing.get('robot_execution_s',0):.1f}|")
        panel = Image.new("RGB", (1280, 290), "white")
        draw = ImageDraw.Draw(panel)
        for col, (label, path) in enumerate((("TARGET", root/"goal/target.png"),
                (item["round"]+" BEFORE", folder/"before/rgb.png"),
                ("AFTER", folder/"after/rgb.png"))):
            draw.text((col*425+8, 8), label, font=visuals.font(19), fill="black")
            if path.exists():
                with Image.open(path) as im:
                    im = im.convert("RGB")
                if col == 1 and plan:
                    im = visuals.draw_action(im, plan["grasp_px"], plan["target_px"])
                    im.save(folder/"proposed_action_overlay.png")
                im.thumbnail((420, 236))
                panel.paste(im, (col*425, 42))
        if (folder/"after/rgb.png").exists():
            last_after = folder/"after/rgb.png"
        panel.save(folder/"comparison.png")
        panels.append(panel)
    if panels:
        overview = Image.new("RGB", (1280, 290*len(panels)), "white")
        for i, panel in enumerate(panels):
            overview.paste(panel, (0, i*290))
        overview.save(root/"episode_overview.png")
        lines += ["", "![episode](episode_overview.png)"]
    if last_after:
        comparison = Image.new("RGB", (1280, 290), "white")
        draw = ImageDraw.Draw(comparison)
        for col, (label, path) in enumerate((("TARGET", root/"goal/target.png"),
                ("INITIAL", root/"round_000/before/rgb.png"), ("FINAL", last_after))):
            draw.text((col*425+8, 8), label, font=visuals.font(23), fill="black")
            if path.exists():
                comparison.paste(picture(path), (col*425, 42))
        comparison.save(root/"final_comparison.png")
        lines += ["", "![target initial final](final_comparison.png)"]
    lines += ["", "## 记录", "",
              "- [目标冻结信息](goal_manifest.json)、[固定协议](protocol.json)、[汇总](episode_summary.json)。",
              "- 每轮保存 plan、执行结果、before/after、Agent 事件、访问统计及 timing。",
              "- 约 5 Hz 执行图像和关节/夹爪反馈用于人工判断 acquisition；未确认不计作抓取成功。",
              "- 目标图取自上一实验最后完整 after，但其路径和上一实验动作过程对 Agent 不可见。",
              "- 不训练、不使用 Graph、不自动修改 prompt、不根据结果追加探索预算。",
              "- 遵循用户要求的 CollectionRuntime 原有设置；未启用的几何检查不标为通过。",
              "- 两场初始化状态不同，因此只是能力观察，不能当作严格受控的 memory 因果对比。",
              "", "请在 human_review.json 中记录目标是否达成及 FOLD_READY 人工判断。", ""]
    (root/"report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--goal-image", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("config/dual_arm.local.json"))
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--skip-initialization", action="store_true",
                        help="start from the user-placed garment; no pick-up/drop reset")
    parser.add_argument("--timeout-s", type=float, default=1200.)
    args = parser.parse_args()
    if not 1 <= args.max_rounds <= 50 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("invalid limits")
    if not args.goal_image.is_file():
        parser.error("goal image does not exist")
    args.data_root, args.scope_dates = args.goal_image, []
    return shared.run(args, experiment_name="goal_only_full_episode_v1", prompt=PROMPT,
                      prepare_inputs=prepare_goal, verify_inputs=shared.verify_memory,
                      plan_round=plan_round, render_report=render, source_file=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
