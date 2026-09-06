#!/usr/bin/env python3
"""Summarize Claude CLI timing/debug logs for one cloth-agent run.

The folding pipeline writes one ``--debug-file`` per Claude invocation under
``results/claude_debug``. This report separates process startup, API request,
time-to-first-byte, tool/MCP activity, and shutdown. It also joins the saved
stage result files so a timeout is visible even when Claude emitted no final
JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_LINE_RE = re.compile(
    r"^(?P<timestamp>[^ ]+) \[(?P<level>[^]]+)\] (?P<message>.*)$"
)
_API_RE = re.compile(r"\[API REQUEST\] (?P<message>.*)$")
_DISPATCH_RE = re.compile(r"\[API:timing\] dispatching to (?P<message>.*)$")
_FIRST_BYTE_RE = re.compile(r"\[API:timing\] first byte after (?P<ms>[0-9.]+)ms")
_TOOL_RE = re.compile(r"(?:Tool|tool|MCP|mcp)")


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_debug_log(path: Path) -> dict[str, Any]:
    first: datetime | None = None
    last: datetime | None = None
    api_requests: list[dict[str, Any]] = []
    dispatches: list[dict[str, Any]] = []
    first_bytes_ms: list[float] = []
    first_byte_events: list[dict[str, Any]] = []
    tool_lines: list[str] = []
    timestamped_lines: list[tuple[datetime, str]] = []
    api_timestamps: list[datetime] = []
    all_lines = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        all_lines += 1
        match = _LINE_RE.match(raw)
        if not match:
            continue
        timestamp = _parse_timestamp(match.group("timestamp"))
        if timestamp is not None:
            first = first or timestamp
            last = timestamp
        message = match.group("message")
        if timestamp is not None:
            timestamped_lines.append((timestamp, message))
        api = _API_RE.search(message)
        if api:
            if timestamp is not None:
                api_timestamps.append(timestamp)
            api_requests.append(
                {
                    "timestamp": match.group("timestamp"),
                    "message": api.group("message"),
                }
            )
        dispatch = _DISPATCH_RE.search(message)
        if dispatch:
            dispatches.append(
                {
                    "timestamp": match.group("timestamp"),
                    "message": dispatch.group("message"),
                }
            )
        byte = _FIRST_BYTE_RE.search(message)
        if byte:
            elapsed_ms = float(byte.group("ms"))
            first_bytes_ms.append(elapsed_ms)
            first_byte_events.append(
                {
                    "timestamp": match.group("timestamp"),
                    "elapsed_ms": elapsed_ms,
                }
            )
        if _TOOL_RE.search(message):
            tool_lines.append(raw[:1000])
    wall_s = (last - first).total_seconds() if first and last else None
    gaps: list[dict[str, Any]] = []
    for (left_time, left_message), (right_time, right_message) in zip(
        timestamped_lines, timestamped_lines[1:]
    ):
        gap_s = (right_time - left_time).total_seconds()
        if gap_s >= 1.0:
            gaps.append(
                {
                    "gap_s": gap_s,
                    "from": left_message[:300],
                    "to": right_message[:300],
                    "from_timestamp": left_time.isoformat(),
                    "to_timestamp": right_time.isoformat(),
                }
            )
    gaps.sort(key=lambda item: float(item["gap_s"]), reverse=True)
    first_api_to_first_byte_s = None
    if api_timestamps and first_byte_events:
        first_byte_time = _parse_timestamp(str(first_byte_events[0]["timestamp"]))
        if first_byte_time is not None:
            first_api_to_first_byte_s = (
                first_byte_time - api_timestamps[0]
            ).total_seconds()
    diagnosis: list[str] = []
    if not api_requests:
        diagnosis.append("no API request was recorded; inspect CLI startup/auth/config")
    elif not first_byte_events:
        diagnosis.append(
            "API request recorded but no first byte; likely API/network/model wait before response"
        )
    else:
        if first_api_to_first_byte_s is not None and first_api_to_first_byte_s > 30:
            diagnosis.append(
                f"first API response byte took {first_api_to_first_byte_s:.1f}s"
            )
        if gaps and float(gaps[0]["gap_s"]) > 30:
            diagnosis.append(
                f"largest event gap is {float(gaps[0]['gap_s']):.1f}s; inspect its from/to events"
            )
    if tool_lines:
        diagnosis.append(
            f"{len(tool_lines)} tool/MCP debug lines were emitted; correlate them with event gaps"
        )
    if not diagnosis:
        diagnosis.append("no long internal gap detected in the available debug events")
    stem_parts = path.stem.split("_")
    stage_name = "_".join(stem_parts[1:-1]) if len(stem_parts) >= 3 else path.stem
    return {
        "stage": stage_name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "line_count": all_lines,
        "wall_s_from_debug_timestamps": wall_s,
        "api_request_count": len(api_requests),
        "api_requests": api_requests,
        "dispatch_count": len(dispatches),
        "dispatches": dispatches,
        "first_byte_ms": first_bytes_ms,
        "first_byte_events": first_byte_events,
        "first_api_to_first_byte_s": first_api_to_first_byte_s,
        "long_gaps": gaps[:20],
        "tool_or_mcp_line_count": len(tool_lines),
        "tool_or_mcp_lines": tool_lines[:40],
        "diagnosis": diagnosis,
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
    }


def _saved_stage_results(run_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    patterns = (
        "results/claude_visual/*.json",
        "results/claude_exploration/*.json",
        "results/claude_auto/*.json",
    )
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            command = payload.get("command")
            debug_path = None
            if isinstance(command, list) and "--debug-file" in command:
                index = command.index("--debug-file")
                if index + 1 < len(command):
                    debug_path = str(command[index + 1])
            results.append(
                {
                    "result_file": str(path.resolve()),
                    "stage": payload.get("stage")
                    or ("visual_planning" if "visual_plan" in path.name else None),
                    "duration_s": payload.get("duration_s"),
                    "error": payload.get("error"),
                    "returncode": payload.get("returncode"),
                    "debug_file": debug_path,
                }
            )
    return results


def build_report(run_dir: Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    debug_dir = root / "results" / "claude_debug"
    logs = []
    if debug_dir.is_dir():
        logs = [_parse_debug_log(path) for path in sorted(debug_dir.glob("*.log"))]
    return {
        "schema_version": 1,
        "run_dir": str(root),
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "debug_logs": logs,
        "saved_stage_results": _saved_stage_results(root),
        "notes": [
            "A missing final JSON with a growing debug log indicates an in-flight or timed-out Claude invocation.",
            "first_byte_ms measures API time-to-first-byte; debug wall time includes CLI startup, tools, and shutdown.",
            "tool_or_mcp_lines are diagnostic excerpts, not an authoritative safety record.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    output = args.output or (args.run_dir / "results" / "claude_timing_report.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "debug_logs": len(report["debug_logs"]), "saved_stage_results": len(report["saved_stage_results"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
