from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_claude_timing import build_report


def test_analyzer_extracts_api_and_first_byte_timing(tmp_path: Path) -> None:
    debug_dir = tmp_path / "results" / "claude_debug"
    debug_dir.mkdir(parents=True)
    (debug_dir / "20260828T120000000000Z_visual_planning_ab12cd34.log").write_text(
        "\n".join(
            [
                "2026-08-28T12:00:00.000Z [DEBUG] [API:timing] dispatching to firstParty model=claude-opus-5",
                "2026-08-28T12:00:01.000Z [DEBUG] [API REQUEST] /v1/messages source=sdk",
                "2026-08-28T12:00:03.500Z [DEBUG] [API:timing] first byte after 3500ms",
                "2026-08-28T12:00:03.600Z [DEBUG] [MCP] test tool ready",
            ]
        ),
        encoding="utf-8",
    )
    report = build_report(tmp_path)
    assert len(report["debug_logs"]) == 1
    parsed = report["debug_logs"][0]
    assert parsed["stage"] == "visual_planning"
    assert parsed["api_request_count"] == 1
    assert parsed["dispatch_count"] == 1
    assert parsed["first_byte_ms"] == [3500.0]
    assert parsed["first_api_to_first_byte_s"] == 2.5
    assert parsed["tool_or_mcp_line_count"] == 1
    assert parsed["wall_s_from_debug_timestamps"] == 3.6


def test_analyzer_joins_saved_timeout_result(tmp_path: Path) -> None:
    result_dir = tmp_path / "results" / "claude_exploration"
    result_dir.mkdir(parents=True)
    (result_dir / "20260828T120000000000Z_failed.json").write_text(
        json.dumps(
            {
                "stage": "final_grounding",
                "duration_s": 400.1,
                "error": "timeout",
                "returncode": None,
                "command": ["claude", "--debug-file", str(tmp_path / "x.log")],
            }
        ),
        encoding="utf-8",
    )
    report = build_report(tmp_path)
    assert report["saved_stage_results"][0]["stage"] == "final_grounding"
    assert report["saved_stage_results"][0]["duration_s"] == 400.1
