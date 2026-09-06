from __future__ import annotations

import json
from pathlib import Path

from cloth_agent.persistent_claude import PersistentClaudeSession


def test_persistent_session_starts_then_resumes_and_survives_reload(
    tmp_path: Path,
) -> None:
    session = PersistentClaudeSession(tmp_path)
    base = [
        "/usr/bin/claude",
        "--print",
        "hello",
        "--no-session-persistence",
        "--system-prompt",
        "test",
    ]

    first = session.prepare_command(base, stage="visual_planning")
    assert "--session-id" in first
    assert "--resume" not in first
    assert "--no-session-persistence" not in first
    assert first[first.index("--autocompact") + 1] == "100k"
    assert "--debug-file" in first
    assert Path(first[first.index("--debug-file") + 1]).parent.name == "claude_debug"
    assert "one persistent garment-robotics reasoning agent" in first[
        first.index("--system-prompt") + 1
    ]

    session.record_success(
        stage="visual_planning",
        stdout=json.dumps({"session_id": session.session_id}),
    )
    second = session.prepare_command(base, stage="evaluation")
    assert "--resume" in second
    assert second[second.index("--resume") + 1] == session.session_id
    assert "--session-id" not in second
    assert "--system-prompt" not in second
    assert "--debug-file" in second

    restored = PersistentClaudeSession(tmp_path)
    third = restored.prepare_command(base, stage="fold_supervisor")
    assert "--resume" in third
    assert restored.turn_count == 1
    assert restored.stage_counts == {"visual_planning": 1}
    assert restored.state_path.is_file()


def test_session_conflict_detection_and_rollover_changes_uuid(tmp_path: Path) -> None:
    session = PersistentClaudeSession(tmp_path)
    old_session_id = session.session_id

    assert session.is_session_conflict_error(
        f"Session ID {old_session_id} is already in use"
    )
    assert session.is_session_conflict_error(
        f"Session with ID {old_session_id} is already in use"
    )
    assert session.is_session_conflict_error(
        '{"subtype":"session_id_already_in_use"}'
    )
    assert not session.is_session_conflict_error("prompt is too long")

    session.record_success(
        stage="visual_planning",
        stdout=json.dumps({"session_id": old_session_id}),
    )
    record = session.rollover(reason="claude_session_conflict", stage="visual_planning")
    assert record["previous_session_id"] == old_session_id
    assert session.session_id != old_session_id
    assert session.started is False
    command = session.prepare_command(
        ["/usr/bin/claude", "--print", "hello"], stage="visual_planning"
    )
    assert command[command.index("--session-id") + 1] == session.session_id
    assert "--resume" not in command


def test_rollover_records_timeout_without_reusing_session(tmp_path: Path) -> None:
    session = PersistentClaudeSession(tmp_path)
    old_session_id = session.session_id
    record = session.rollover(reason="claude_timeout", stage="final_grounding")

    assert record["reason"] == "claude_timeout"
    assert record["stage"] == "final_grounding"
    assert record["previous_session_id"] == old_session_id
    assert session.session_id != old_session_id
    assert session.started is False
    state = json.loads(session.state_path.read_text(encoding="utf-8"))
    assert state["rollovers"][-1] == record
