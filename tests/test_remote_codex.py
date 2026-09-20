"""Transport/protocol tests use a fake CLI, never credentials or robot motion."""
import json
import os
import sys

import pytest
from PIL import Image

from cloth_agent import image_tools_mcp
from cloth_agent.codex_remote_runner import CodexEvents, codex_command, provider_config, output_schema, restore_optional_fields
from cloth_agent.planner_backend import RemoteCodexBackend, PlannerBackendError, parse_claude_json


@pytest.fixture(autouse=True)
def staged_tool_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "image_tools", image_tools_mcp)


def mcp_event(content=None, **overrides):
    return {"type": "item.completed", "item": {
        "id": "call_1", "type": "mcp_tool_call", "server": "cloth_image",
        "tool": "view_image", "arguments": {"image_id": "image_0"},
        "status": "completed", "result": {"content": content or []}, **overrides}}


def finish(state, text='{"ok":true}'):
    state.consume({"type": "item.completed", "item": {"type": "agent_message", "text": text}})
    state.consume({"type": "turn.completed", "usage": {"input_tokens": 10}})


def test_terminal_contract_and_missing_pixels():
    state = CodexEvents(2)
    events = state.consume(mcp_event([{"type": "text", "text": "metadata only"}]))
    assert not image_tools_mcp.image_content_summary(events[1]["message"]["content"])["image_count"]
    with pytest.raises(ValueError, match="terminal"):
        state.final(0)
    finish(state)
    assert state.final(0)["structured_output"] == {"ok": True}
    state.consume({"type": "error", "message": "upstream failure"})
    with pytest.raises(ValueError, match="terminal"):
        state.final(0)


@pytest.mark.parametrize("text", ["refused", "```json\n{}\n```", "[]"])
def test_non_json_final_is_rejected(text):
    state = CodexEvents(1)
    finish(state, text)
    with pytest.raises(ValueError):
        state.final(0)


def test_budget_and_unexpected_tools():
    state = CodexEvents(1)
    state.consume(mcp_event())
    with pytest.raises(ValueError, match="budget"):
        state.consume(mcp_event(id="call_2"))
    with pytest.raises(ValueError, match="server/tool"):
        CodexEvents(1).consume(mcp_event(server="robot"))
    with pytest.raises(ValueError, match="non-image"):
        CodexEvents(1).consume({"type": "item.started", "item": {"type": "command_execution"}})


def test_output_schema_preserves_original_validation():
    from jsonschema import Draft202012Validator
    from cloth_agent.remote_fold import MOTION_SCHEMA, VISUAL_PLAN_JSON_SCHEMA
    schema = output_schema(MOTION_SCHEMA)
    assert "anyOf" in schema["properties"]["actions"]["items"]
    optional = output_schema(VISUAL_PLAN_JSON_SCHEMA)
    assert "skill_invocations" in optional["required"]
    assert restore_optional_fields({"skill_invocations": None}, VISUAL_PLAN_JSON_SCHEMA) == {}
    original = {"type": "object", "properties": {"evidence": {
        "type": "array", "uniqueItems": True, "items": {"type": "string"}}},
        "required": ["evidence"], "additionalProperties": False}
    duplicate = {"evidence": ["same", "same"]}
    assert Draft202012Validator(output_schema(original)).is_valid(duplicate)
    assert not Draft202012Validator(original).is_valid(restore_optional_fields(duplicate, original))


def test_image_server_enforces_call_limit_across_restart(tmp_path):
    Image.new("RGB", (20, 20), "blue").save(tmp_path / "image_0.png")
    tools = image_tools_mcp.ImageTools(tmp_path, 1, call_limit=1)
    tools.call("view_image", {"image_id": "image_0"})
    restarted = image_tools_mcp.ImageTools(tmp_path, 1, call_limit=1)
    with pytest.raises(ValueError, match="call budget"):
        restarted.call("rotate_image", {"image_id": "image_0", "degrees_clockwise": 90})
    assert len(restarted.views) == 1


def test_command_isolates_tools_and_keeps_provider(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('model="old"\nmodel_provider="company"\n'
        '[model_providers.company]\nname="company"\nbase_url="https://example.invalid/v1"\n'
        'env_key="COMPANY_API_KEY"\nwire_api="responses"\n'
        '[mcp_servers.unrelated]\ncommand="unwanted"\n')
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    provider = provider_config()
    assert "model" not in provider and "mcp_servers" not in provider
    assert provider["model_provider"] == "company"
    (tmp_path / "image_tools.mcp.json").write_text(json.dumps({"mcpServers": {
        "cloth_image": {"command": "python3", "args": ["image_tools.py"]}}}))
    command = codex_command(tmp_path, {"model": "gpt-6-astra", "reasoning_effort": "medium",
        "system_prompt": "RGB only", "max_tool_calls": 6}, provider)
    assert command[:3] == ["codex", "exec", "--json"]
    assert "--ignore-user-config" in command
    assert 'model_reasoning_effort="medium"' in command
    assert 'features.shell_tool=false' in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-6-astra"
    assert "claude" not in command


FAKE_CODEX = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from image_tools import ImageTools
def emit(x):
    print(json.dumps(x), flush=True)
assert sys.argv[1:3] == ['exec', '--json']
assert sys.argv[sys.argv.index('--model')+1] == 'gpt-6-astra'
assert 'model_reasoning_effort="medium"' in sys.argv
prompt = sys.stdin.read()
tools = ImageTools(Path.cwd(), 1)
for i, (tool, arguments) in enumerate([
    ('view_image', {'image_id':'image_0'}),
    ('rotate_image', {'image_id':'image_0', 'degrees_clockwise':90}),
]):
    value = tools.call(tool, arguments)
    payload = tools.image_result(value)
    if 'METADATA_ONLY' in prompt:
        payload['content'] = [c for c in payload['content'] if c['type'] == 'text']
    emit({'type':'item.completed','item':{'id':str(i),'type':'mcp_tool_call',
        'server':'cloth_image','tool':tool,'arguments':arguments,'status':'completed','result':payload}})
emit({'type':'item.completed','item':{'type':'agent_message', 'text':json.dumps({'ok':True})}})
if 'NO_TERMINAL' not in prompt:
    emit({'type':'turn.completed', 'usage':{'input_tokens':1}})
'''


@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text(FAKE_CODEX)
    executable.chmod(0o700)
    launcher = tmp_path / "ssh"
    launcher.write_text('#!/bin/sh\nfor cloth_arg do :; done\nexec /bin/sh -c "$cloth_arg"\n')
    launcher.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + str(os.path.dirname(sys.executable)) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CLOTH_REMOTE_IMAGE_PYTHON", sys.executable)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty_auth"))
    image = tmp_path / "rgb.png"
    Image.new("RGB", (48, 32), "red").save(image)
    backend = RemoteCodexBackend(ssh_binary=str(launcher), timeout_s=15)
    monkeypatch.setattr(backend, "_upload", lambda path: path.as_uri())
    return backend, image


@pytest.mark.parametrize("metadata_only", [False, True])
def test_real_shell_runner_image_audit_and_cleanup(fake_remote, tmp_path, metadata_only):
    backend, image = fake_remote
    result = backend.invoke(prompt="METADATA_ONLY" if metadata_only else "Inspect RGB",
        image_paths=[image], schema={"type": "object"}, system_prompt="RGB only",
        image_edit_limit=2, orientation_correction=True, debug_dir=tmp_path / "debug")
    assert parse_claude_json(result.stdout) == {"ok": True}
    assert json.loads(result.stdout)["provider"] == "codex"
    assert len(result.image_sources) == 2
    for source in result.image_sources:
        assert (source["image_delivery_status"] == "VERIFIED") is not metadata_only
        assert not os.path.exists(source.get("remote_path", "/tmp/absent_cloth_test_path"))
    assert any(e.get("kind") == "orientation_guard" for e in result.image_tool_events)


def test_missing_terminal_fails_even_with_json(fake_remote, tmp_path):
    backend, image = fake_remote
    with pytest.raises(PlannerBackendError, match="terminal"):
        backend.invoke(prompt="NO_TERMINAL", image_paths=[image], schema={},
                       system_prompt="RGB", debug_dir=tmp_path / "debug")


def test_original_schema_rejection_after_successful_cli(fake_remote, tmp_path):
    backend, image = fake_remote
    with pytest.raises(PlannerBackendError, match="original schema"):
        backend.invoke(prompt="Inspect", image_paths=[image],
            schema={"type": "object", "properties": {"ok": {"const": False}}, "required": ["ok"]},
            system_prompt="RGB", debug_dir=tmp_path / "debug")
