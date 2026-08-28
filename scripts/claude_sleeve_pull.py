#!/usr/bin/env python3
"""Run a language-conditioned Claude sleeve-pull experiment.

This is intentionally separate from the collar-specific pipeline.  Claude is
asked to identify a visible, graspable sleeve and move it to the farthest safe
reachable location supported by the current RGB-D evidence and controller IK.
The wrapper supplies the task instruction; perception, optional Molmo evidence,
grounding, preflight, and execution remain owned by the language-skill runtime.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.language_skill_pipeline import main as language_skill_main


DEFAULT_INSTRUCTION = (
    "抓住当前衣服中可见且可抓的袖子，并沿 Claude 根据当前 RGB-D、reference、"
    "Molmo 证据和工作空间判断出的最远安全可达方向，把袖子拉到最远处。"
    "不要抓肩膀、衣身、标签或桌面；袖子、抓取点、抬升高度、拉伸方向和终点都由 Claude 决定，"
    "不要硬编码像素或坐标。"
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--instruction" not in args:
        args = ["--instruction", DEFAULT_INSTRUCTION, *args]
    if "--claude-timeout-s" not in args:
        args = [*args, "--claude-timeout-s", "900"]
    return language_skill_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
