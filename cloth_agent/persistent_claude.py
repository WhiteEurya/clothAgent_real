"""Run-local persistent Claude Code conversation support.

The CLI process may exit after each ``--print`` turn, but the conversation is
kept under one explicit session UUID and resumed on the next stage. This keeps
the folding agent's reasoning continuity while preserving host-side process
timeouts, JSON schemas, tool restrictions, and crash recovery.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class PersistentClaudeSession:
    """Persist and resume one Claude conversation for a complete robot run."""

    SYSTEM_PROMPT = (
        "You are one persistent garment-robotics reasoning agent for an entire "
        "closed-loop run. Each user turn names its current stage: visual planning, "
        "grounding/compiler, acquisition inspection, full evaluation, or fold-state "
        "supervision. Carry forward relevant physical outcomes and hypotheses across "
        "turns, while obeying the current turn's tool restrictions and JSON schema. "
        "You never control the robot directly: the host remains authoritative for "
        "calibration, workspace, grasp height, preflight, IK, and physical execution."
    )

    def __init__(
        self,
        run_dir: Path,
        *,
        name: str = "persistent-fold-agent",
        max_generation_turns: int = 8,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.state_path = (
            self.run_dir / "workspace" / "persistent_claude_session.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {}
        if self.state_path.is_file():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except (OSError, json.JSONDecodeError):
                state = {}
        raw_session_id = state.get("session_id")
        try:
            session_id = str(uuid.UUID(str(raw_session_id)))
        except (ValueError, TypeError, AttributeError):
            session_id = str(uuid.uuid4())
        self.session_id = session_id
        self.name = str(state.get("name") or name)
        self.max_generation_turns = max(2, int(max_generation_turns))
        self.started = bool(state.get("started", False))
        self.turn_count = max(0, int(state.get("turn_count", 0)))
        self.generation = max(1, int(state.get("generation", 1)))
        self.generation_turn_count = max(
            0,
            int(state.get("generation_turn_count", self.turn_count)),
        )
        self.stage_counts = {
            str(key): max(0, int(value))
            for key, value in dict(state.get("stage_counts") or {}).items()
        }
        self.rollovers = [
            dict(item)
            for item in list(state.get("rollovers") or [])[-20:]
            if isinstance(item, dict)
        ]
        self._save()

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "name": self.name,
                    "started": self.started,
                    "turn_count": self.turn_count,
                    "generation": self.generation,
                    "generation_turn_count": self.generation_turn_count,
                    "max_generation_turns": self.max_generation_turns,
                    "stage_counts": self.stage_counts,
                    "rollovers": self.rollovers,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def debug_file(self, stage: str) -> Path:
        """Allocate a run-local Claude CLI debug log for one invocation."""

        safe_stage = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(stage)
        ).strip("_") or "claude"
        directory = self.run_dir / "results" / "claude_debug"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return directory / f"{stamp}_{safe_stage}_{uuid.uuid4().hex[:8]}.log"

    @staticmethod
    def _remove_option(command: list[str], option: str, *, takes_value: bool) -> None:
        while option in command:
            index = command.index(option)
            del command[index]
            if takes_value and index < len(command):
                del command[index]

    def prepare_command(self, command: Sequence[str], *, stage: str) -> list[str]:
        """Attach this run's resumable session contract to one CLI command."""

        if self.started and self.generation_turn_count >= self.max_generation_turns:
            self.rollover(
                reason="preemptive_turn_budget",
                stage=stage,
            )
        result = [str(item) for item in command]
        if not result:
            raise ValueError("Claude command cannot be empty")
        self._remove_option(result, "--no-session-persistence", takes_value=False)
        for option in (
            "--session-id",
            "--resume",
            "--name",
            "--autocompact",
            "--debug-file",
        ):
            self._remove_option(result, option, takes_value=True)
        self._remove_option(result, "--system-prompt", takes_value=True)
        session_args = (
            ["--resume", self.session_id]
            if self.started
            else ["--session-id", self.session_id, "--name", self.name]
        )
        initial_system_args = (
            ["--system-prompt", self.SYSTEM_PROMPT] if not self.started else []
        )
        # Insert after the binary. Keep all stage-specific schemas and tool
        # restrictions intact.
        result[1:1] = [
            *session_args,
            "--autocompact",
            "100k",
            "--debug-file",
            str(self.debug_file(stage)),
            *initial_system_args,
        ]
        return result

    @staticmethod
    def is_context_limit_error(*messages: str) -> bool:
        combined = "\n".join(str(message) for message in messages).lower()
        return "prompt is too long" in combined or "blocking_limit" in combined

    @staticmethod
    def is_session_conflict_error(*messages: str) -> bool:
        """Recognize Claude CLI errors for a UUID that is already active."""

        combined = "\n".join(str(message) for message in messages).lower()
        return (
            ("session id" in combined or "session with id" in combined)
            and "already in use" in combined
        ) or "session_id_already_in_use" in combined

    def rollover(self, *, reason: str, stage: str | None = None) -> dict[str, Any]:
        """Start a fresh Claude session while retaining run-local experience."""

        previous_session_id = self.session_id
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "reason": str(reason),
            "stage": str(stage) if stage is not None else None,
            "previous_session_id": previous_session_id,
            "previous_generation": self.generation,
            "previous_generation_turn_count": self.generation_turn_count,
            "total_turn_count": self.turn_count,
        }
        self.rollovers.append(record)
        self.rollovers = self.rollovers[-20:]
        self.session_id = str(uuid.uuid4())
        self.generation += 1
        self.generation_turn_count = 0
        self.started = False
        self._save()
        return record

    def record_success(self, *, stage: str, stdout: str = "") -> None:
        """Commit a successful turn so later invocations use ``--resume``."""

        returned_session_id = None
        try:
            payload = json.loads(stdout)
            if isinstance(payload, dict):
                returned_session_id = payload.get("session_id")
        except (json.JSONDecodeError, TypeError):
            pass
        if returned_session_id:
            try:
                self.session_id = str(uuid.UUID(str(returned_session_id)))
            except (ValueError, TypeError, AttributeError):
                pass
        self.started = True
        self.turn_count += 1
        self.generation_turn_count += 1
        key = str(stage)
        self.stage_counts[key] = self.stage_counts.get(key, 0) + 1
        self._save()

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "started": self.started,
            "turn_count": self.turn_count,
            "generation": self.generation,
            "generation_turn_count": self.generation_turn_count,
            "max_generation_turns": self.max_generation_turns,
            "stage_counts": dict(self.stage_counts),
            "rollover_count": len(self.rollovers),
            "last_rollover": self.rollovers[-1] if self.rollovers else None,
            "state_path": str(self.state_path),
        }
