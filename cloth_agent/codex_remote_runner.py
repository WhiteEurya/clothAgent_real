"""Standalone SSH job adapter: Codex JSONL -> audited planner events.

Only the standard library is required. Codex loads its own named profile.
No camera, depth, kinematics or robot code is deployed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def output_schema(schema):
    """Provider-compatible shape; original constraints are validated locally."""
    result = {k: v for k, v in schema.items() if k in {
        "type", "description", "enum", "const", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "pattern",
        "minItems", "maxItems", "$ref"}}
    if "const" in result:
        result["enum"] = [result.pop("const")]
    if "type" not in result and "enum" in result:
        types = {"null" if v is None else "boolean" if isinstance(v, bool) else
                 "string" if isinstance(v, str) else "number" for v in result["enum"]}
        result["type"] = sorted(types) if len(types) > 1 else types.pop()
    for key in ("$defs", "definitions"):
        if key in schema:
            result[key] = {name: output_schema(value) for name, value in schema[key].items()}
    branches = schema.get("oneOf", schema.get("anyOf"))
    if branches is not None:
        result["anyOf"] = [output_schema(branch) for branch in branches]
    if schema.get("type") == "object":
        properties = {}
        for key, child in schema.get("properties", {}).items():
            properties[key] = output_schema(child)
            if key not in schema.get("required", []):
                properties[key] = {"anyOf": [properties[key], {"type": "null"}],
                                   "description": "Use null when this optional field is not applicable."}
        result.update(properties=properties, required=list(properties), additionalProperties=False)
    if "items" in schema:
        result["items"] = output_schema(schema["items"])
    return result


def restore_optional_fields(value, schema):
    """Remove only provider-added null placeholders before host validation."""
    from jsonschema import Draft202012Validator
    for branch in schema.get("oneOf", schema.get("anyOf", [])):
        candidate = restore_optional_fields(value, branch)
        if Draft202012Validator(branch).is_valid(candidate):
            return candidate
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return {key: restore_optional_fields(child, properties.get(key, {}))
                for key, child in value.items()
                if not (key in properties and key not in schema.get("required", [])
                        and child is None and not Draft202012Validator(properties[key]).is_valid(None))}
    if isinstance(value, list):
        return [restore_optional_fields(child, schema.get("items", {})) for child in value]
    return value


def toml_value(value):
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + "=" + toml_value(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(toml_value(v) for v in value) + "]"
    if isinstance(value, (str, bool, int, float)):
        return json.dumps(value, allow_nan=False)
    raise ValueError("unsupported Codex provider configuration value")


def codex_command(job, request):
    mcp = json.loads((job / "image_tools.mcp.json").read_text())["mcpServers"]["cloth_image"]
    config = {
        "model_reasoning_effort": request["reasoning_effort"],
        "developer_instructions": request["system_prompt"] +
            "\nUse only cloth_image MCP tools. Inspect image_0 and other relevant supplied images "
            "with view_image before deciding. Tool metadata alone is not visual evidence. "
            f"At most {request['max_tool_calls']} tool calls are allowed. Return one JSON object.",
        "approval_policy": "never",
        "web_search": "disabled",
        "features.shell_tool": False,
        "features.multi_agent": False,
        "features.apps": False,
        "features.skills": False,
        "tools.view_image": False,
        "project_doc_max_bytes": 0,
        "mcp_servers": {"cloth_image": {
            "command": mcp["command"], "args": mcp["args"] +
                ["--call-limit", str(min(64, request["max_tool_calls"]))],
            "required": True, "startup_timeout_sec": 30,
        }},
    }
    command = ["codex", "exec", "--json", "--ephemeral", "-p", request["profile"],
               "--ignore-rules", "--skip-git-repo-check", "--sandbox", "read-only",
               "--model", request["model"], "--cd", str(job),
               "--output-schema", str(job / "response_schema.json")]
    for key, value in config.items():
        command.extend(["-c", key + "=" + toml_value(value)])
    return [*command, "-"]


class CodexEvents:
    """Translate only evidence actually present in the Codex output stream."""

    def __init__(self, max_tool_calls):
        self.max_tool_calls = max_tool_calls
        self.calls = set()
        self.completed_calls = set()
        self.messages = []
        self.completed = False
        self.failed = False
        self.usage = None
        self.errors = []
        self.last_event_type = None

    def consume(self, event):
        kind = event.get("type")
        self.last_event_type = kind
        if kind in {"turn.failed", "error"}:
            self.failed = True
            # Codex reports API/auth/model/schema failures on stdout as JSONL,
            # often with empty stderr. Keep their reason in the terminal error
            # envelope; the host otherwise only displays our generic failure.
            detail = {key: event[key] for key in
                      ("type", "message", "error", "code", "status", "status_code")
                      if key in event}
            self.errors.append(detail)
            self.errors = self.errors[-8:]
        if kind == "turn.completed":
            if self.completed:
                raise ValueError("multiple Codex terminal turns")
            self.completed = True
            self.usage = event.get("usage")
        item = event.get("item", {})
        if kind not in {"item.started", "item.completed", "item.updated"}:
            return []
        if self.completed:
            raise ValueError("Codex item after terminal turn")
        item_type = item.get("type")
        if item_type in {"command_execution", "file_change", "web_search", "collab_tool_call"}:
            raise ValueError("unexpected non-image Codex tool")
        if item_type == "agent_message" and kind == "item.completed":
            self.messages.append(item.get("text", ""))
        if item_type != "mcp_tool_call":
            return []
        from image_tools import TOOLS
        if item.get("server") != "cloth_image" or item.get("tool") not in {t["name"] for t in TOOLS}:
            raise ValueError("unexpected Codex MCP server/tool")
        identity = item.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Codex MCP call has no identity")
        result = []
        if identity not in self.calls:
            self.calls.add(identity)
            if len(self.calls) > self.max_tool_calls:
                raise ValueError("Codex tool-call budget exceeded")
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result.append({"type": "assistant", "message": {"content": [{
                "type": "tool_use", "id": identity,
                "name": "mcp__cloth_image__" + item["tool"], "input": arguments}]}})
        if kind == "item.completed":
            if identity in self.completed_calls:
                raise ValueError("duplicate Codex MCP completion")
            self.completed_calls.add(identity)
            payload = item.get("result") or {}
            # No reading source files to fabricate missing CLI image content.
            # MCP uses mimeType; image_content_summary understands this format.
            result.append({"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": identity,
                "content": payload.get("content", []),
                "is_error": bool(item.get("error") or payload.get("isError") or
                                 item.get("status") != "completed")}]}})
        return result

    def final(self, returncode):
        if returncode or self.failed or not self.completed or not self.messages:
            upstream = json.dumps(self.errors, ensure_ascii=False)
            raise ValueError(
                "Codex did not return a successful terminal turn and final message; "
                f"cli_exit_code={returncode}, turn_completed={self.completed}, "
                f"final_messages={len(self.messages)}, last_event={self.last_event_type}; "
                f"upstream_errors={upstream[:8000]}"
            )
        if self.calls != self.completed_calls:
            raise ValueError("Codex ended with unfinished MCP calls")
        value = json.loads(self.messages[-1])
        if not isinstance(value, dict):
            raise ValueError("Codex final response is not a JSON object")
        return {"type": "result", "subtype": "success", "is_error": False,
                "structured_output": value, "provider": "codex",
                "usage": self.usage, "num_tool_calls": len(self.calls)}


def emit(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = args.job.resolve(strict=True)
    request = json.loads((job / "codex_request.json").read_text())
    state = CodexEvents(request["max_tool_calls"])
    process = None
    try:
        (job / "response_schema.json").write_text(json.dumps(output_schema(request["schema"])))
        command = codex_command(job, request)
        # Inherit the prompt pipe, avoiding a large write-before-read deadlock.
        process = subprocess.Popen(command, stdin=sys.stdin, stdout=subprocess.PIPE,
                                   stderr=sys.stderr, text=True, cwd=job)
        for line in process.stdout:
            event = json.loads(line)
            emit({"type": "system", "subtype": "codex_event", "codex_event": event})
            for translated in state.consume(event):
                emit(translated)
            if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "mcp_tool_call":
                from image_tools import read_hook
                item = event["item"]
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                read_hook(job, {"hook_event_name": "PostToolUse",
                    "tool_name": "mcp__cloth_image__" + item["tool"],
                    "tool_use_id": item["id"], "tool_input": arguments,
                    "tool_response": item.get("result")})
        final = state.final(process.wait())
        if request.get("orientation_correction"):
            from image_tools import orientation_guard
            orientation_guard(job, {"hook_event_name": "Stop",
                                    "last_assistant_message": json.dumps(final["structured_output"])})
        emit({**final, "model": request["model"], "reasoning_effort": request["reasoning_effort"],
              "profile": request["profile"]})
        return 0
    except Exception as exc:
        emit({"type": "result", "subtype": "error_codex", "is_error": True,
              "provider": "codex", "model": request["model"],
              "profile": request["profile"],
              "reasoning_effort": request["reasoning_effort"],
              "result": f"{type(exc).__name__}: {exc}", "errors": state.errors})
        return 1
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
