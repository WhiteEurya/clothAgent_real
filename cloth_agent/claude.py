"""Constrained Claude Code integration for writing experiment files."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class ClaudeCodeError(RuntimeError):
    """Raised when Claude Code cannot complete a workspace-only turn."""


@dataclass(frozen=True)
class ClaudeResult:
    prompt: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ClaudeCodeClient:
    """Invoke the installed ``claude`` CLI with no shell or core write access."""

    def __init__(self, binary: str = "claude", timeout_s: int = 300):
        self.binary = binary
        self.timeout_s = timeout_s

    def invoke(self, prompt: str, workspace: Path) -> ClaudeResult:
        workspace = workspace.resolve()
        if not workspace.is_dir():
            raise ClaudeCodeError(f"workspace does not exist: {workspace}")
        binary = shutil.which(self.binary) if Path(self.binary).name == self.binary else self.binary
        if binary is None:
            raise ClaudeCodeError(f"Claude Code CLI not found: {self.binary}")
        system_prompt = (
            "You are the implementation assistant for one robotics research run. "
            "You may edit files only in the current run workspace. The core project, "
            "robot driver, safety layer, perception, evaluation, and agent runtime are "
            "read-only and unavailable for editing. If you see an infrastructure defect, "
            "write an ENGINEERING_ISSUE: report in your response and do not edit core code. "
            "Generate or modify a minimal experiment script with exactly one def run(): "
            "function. It must contain only calls to move(x, y, z, yaw), open_gripper(), "
            "close_gripper(), and home(), with no imports, no xarm SDK, no shell, no retry, "
            "no exception handling, and no dynamic code."
        )
        command = [
            binary,
            "--print",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            "Read",
            "Edit",
            "Write",
            "--tools",
            "Read,Edit,Write",
            "--add-dir",
            str(workspace),
            "--safe-mode",
            "--system-prompt",
            system_prompt,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClaudeCodeError(f"Claude Code invocation failed: {exc}") from exc
        result = ClaudeResult(prompt, command, completed.returncode, completed.stdout, completed.stderr, _now())
        log_dir = workspace / "results" / "claude"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (log_dir / f"{stamp}.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            raise ClaudeCodeError(
                f"Claude Code exited with {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return result
