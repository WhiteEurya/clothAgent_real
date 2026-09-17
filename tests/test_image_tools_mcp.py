import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.image_tools_mcp import ImageTools, TOOLS, main, transform_point
from cloth_agent.planner_backend import RemoteClaudeBackend, parse_claude_json


@pytest.fixture
def scene(tmp_path):
    image = Image.new("RGB", (13, 9))
    image.putdata([(x * 17, y * 23, (x + y) * 9) for y in range(9) for x in range(13)])
    image.save(tmp_path / "image_0.png")
    return ImageTools(tmp_path, 1), image


@pytest.mark.parametrize("degrees", [0, 90, -90, 180, 270, 360])
def test_rotation_pixels_and_inverse_match_pillow(scene, degrees):
    tools, original = scene
    rotated = tools.call("rotate_image", {"image_id": "image_0", "degrees_clockwise": degrees})
    with Image.open(rotated["path"]) as image:
        assert image.tobytes() == original.rotate(-degrees, expand=True).tobytes()
        for point in ([0, 0], [image.width-1, image.height-1], [3, 2]):
            mapped = tools.call("map_point", {"image_id": rotated["image_id"], "pixel_xy": point})
            source = tuple(round(v) for v in mapped["pixel_xy"])
            assert image.getpixel(tuple(point)) == original.getpixel(source)


def test_crop_rotate_zoom_chain_maps_back_to_original(scene):
    tools, _ = scene
    crop = tools.call("crop_image", {"image_id": "image_0", "box": [2, 1, 10, 7]})
    rotated = tools.call("rotate_image", {"image_id": crop["image_id"], "degrees_clockwise": 90})
    zoom = tools.call("resize_image", {"image_id": rotated["image_id"], "scale": 2})
    point = [5., 7.]
    mapped = tools.call("map_point", {"image_id": zoom["image_id"], "pixel_xy": point})
    assert mapped["original_image_index"] == 0
    assert mapped["pixel_xy"] == pytest.approx([5.25, 3.75])
    assert mapped["pixel_xy"] == pytest.approx(transform_point(zoom["to_original"], point))


def test_arbitrary_rotation_and_padding(scene):
    tools, _ = scene
    crop = tools.call("crop_image", {"image_id": "image_0", "box": [3, 2, 10, 7]})
    view = tools.call("rotate_image", {"image_id": crop["image_id"], "degrees_clockwise": 37})
    center = [(v-1)/2 for v in view["size"]]
    assert tools.call("map_point", {"image_id": view["image_id"], "pixel_xy": center})["pixel_xy"] == pytest.approx([6, 4])
    # This corner projects inside the original but outside the crop: reject.
    with pytest.raises(ValueError, match="padding"):
        tools.call("map_point", {"image_id": view["image_id"], "pixel_xy": [0, 0]})
    radians = math.radians(37)
    assert view["to_parent"][0] == pytest.approx(math.cos(radians))


@pytest.mark.parametrize("name,args", [
    ("image_info", {"image_id": "../../secret"}),
    ("crop_image", {"image_id": "image_0", "box": [-1, 0, 5, 5]}),
    ("crop_image", {"image_id": "image_0", "box": [0, 0, 5, 5], "output": "/tmp/overwrite"}),
    ("resize_image", {"image_id": "image_0", "scale": 9}),
    ("rotate_image", {"image_id": "image_0", "degrees_clockwise": True}),
    ("map_point", {"image_id": "image_0", "pixel_xy": [99, 1]}),
])
def test_invalid_requests_are_logged_and_cannot_overwrite_sources(scene, name, args):
    tools, original = scene
    with pytest.raises(ValueError):
        tools.call(name, args)
    assert Image.open(tools.job / "image_0.png").tobytes() == original.tobytes()
    event = json.loads((tools.job / "image_tool_calls.jsonl").read_text().splitlines()[-1])
    assert event["status"] == "error"
    assert not list(tools.job.glob("view_*.png"))


def test_tool_bootstrap_and_real_stdio_protocol(scene):
    tools, _ = scene
    main(["--job", str(tools.job), "--image-count", "1", "--prepare"])
    config = json.loads((tools.job / "image_tools.mcp.json").read_text())
    server = config["mcpServers"]["cloth_image"]
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "rotate_image", "arguments": {"image_id": "image_0", "degrees_clockwise": 90}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "run_robot", "arguments": {}}},
    ]
    result = subprocess.run([server["command"], *server["args"]],
        input="\n".join(map(json.dumps, requests)) + "\n", text=True, capture_output=True, check=True)
    responses = list(map(json.loads, result.stdout.splitlines()))
    assert len(responses) == 4
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert responses[1]["result"]["tools"] == TOOLS
    assert json.loads(responses[2]["result"]["content"][0]["text"])["size"] == [9, 13]
    assert responses[3]["result"]["isError"] is True


def test_six_shared_edits_then_inspection_only_with_restart_persistence(scene):
    original_tools, _ = scene
    tools = ImageTools(original_tools.job, 1, edit_limit=6)
    operations = [('rotate_image', {'degrees_clockwise': 90}),
                  ('crop_image', {'box': [0, 0, 5, 5]}), ('resize_image', {'scale': 2}),
                  ('rotate_image', {'degrees_clockwise': 180}),
                  ('crop_image', {'box': [0, 0, 6, 6]}), ('resize_image', {'scale': 1})]
    for index, (name, args) in enumerate(operations):
        result = tools.call(name, {'image_id': 'image_0', **args})
        assert result['edit_budget']['remaining'] == 5-index
    count = len(list(tools.job.glob('view_*.png')))
    with pytest.raises(ValueError, match='budget exhausted'):
        tools.call('rotate_image', {'image_id': 'image_0', 'degrees_clockwise': 45})
    assert len(list(tools.job.glob('view_*.png'))) == count
    assert tools.call('image_info', {'image_id': result['image_id']})['edit_budget']['remaining'] == 0
    assert tools.call('map_point', {'image_id': result['image_id'], 'pixel_xy': [1, 1]})['pixel_xy'] == [1, 1]
    with Image.open(result['path']) as image:
        assert image.width > 0  # all existing files remain readable
    restarted = ImageTools(tools.job, 1, edit_limit=6)
    with pytest.raises(ValueError, match='budget exhausted'):
        restarted.call('resize_image', {'image_id': 'image_0', 'scale': 3})
    assert restarted.call('image_info', {'image_id': 'image_0'})['edit_budget']['used'] == 6


def test_invalid_edit_attempts_consume_budget_but_info_does_not(scene):
    base, _ = scene
    tools = ImageTools(base.job, 1, edit_limit=6)
    for remaining in range(5, -1, -1):
        with pytest.raises(ValueError):
            tools.call('crop_image', {'image_id': 'image_0', 'box': [-1, 0, 2, 2]})
        assert tools.call('image_info', {'image_id': 'image_0'})['edit_budget']['remaining'] == remaining
    assert tools.created == 0
    with pytest.raises(ValueError, match='budget exhausted'):
        tools.call('rotate_image', {'image_id': 'image_0', 'degrees_clockwise': 90})


def test_saved_inventory_reads_and_duplicate_edits_survive_restart(scene):
    from cloth_agent.image_tools_mcp import audit
    base, _ = scene
    tools = ImageTools(base.job, 1, edit_limit=1)
    args = {'image_id': 'image_0', 'box': [1, 1, 8, 6]}
    first = tools.call('crop_image', args)
    assert first['inspection_history'][-1]['completed_reads'] == 0
    for status in ('started', 'completed'):
        audit(tools.job, {'kind': 'read', 'tool': 'Read', 'status': status,
              'tool_use_id': 'read-1', 'arguments': {'file_path': Path(first['path']).name}})
    assert tools.inspection_history()[-1]['completed_reads'] == 0
    audit(tools.job, {'kind': 'read', 'tool': 'Read', 'status': 'completed',
          'tool_use_id': 'read-verified', 'arguments': {'file_path': Path(first['path']).name},
          'image_content': {'identity_status': 'VERIFIED'}})
    duplicate = tools.call('crop_image', args)
    assert duplicate['reused'] is True
    assert duplicate['image_id'] == first['image_id']
    assert duplicate['edit_budget']['used'] == 1
    restarted = ImageTools(tools.job, 1, edit_limit=1)
    same = restarted.call('crop_image', args)
    assert same['path'] == first['path']
    assert len(list(tools.job.glob('view_*.png'))) == 1
    inventory = restarted.call('list_images', {})['inspection_history']
    assert len(inventory) == 2
    assert inventory[-1]['completed_reads'] == 1
    assert inventory[-1]['arguments'] == args
    assert json.loads((tools.job / 'inspection_history.json').read_text())['images'] == inventory
    with pytest.raises(ValueError, match='budget exhausted'):
        restarted.call('crop_image', {**args, 'box': [2, 1, 8, 6]})


def test_tool_call_budget_not_reset_on_restart(scene):
    from cloth_agent.image_tools_mcp import MAX_CALLS
    tools, _ = scene
    for _ in range(MAX_CALLS):
        tools.call('image_info', {'image_id': 'image_0'})
    restarted = ImageTools(tools.job, 1)
    with pytest.raises(ValueError, match='call budget exhausted'):
        restarted.call('list_images', {})


def test_budget_bootstrap_and_stdio_enforce_seventh_edit_denial(scene):
    tools, _ = scene
    main(['--job', str(tools.job), '--image-count', '1', '--edit-limit', '6', '--prepare'])
    server = json.loads((tools.job / 'image_tools.mcp.json').read_text())['mcpServers']['cloth_image']
    assert server['args'][-2:] == ['--edit-limit', '6']
    requests = [{'jsonrpc': '2.0', 'id': i, 'method': 'tools/call', 'params': {
        'name': 'rotate_image', 'arguments': {'image_id': 'image_0', 'degrees_clockwise': i*10}}}
        for i in range(7)]
    requests.append({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
                     'params': {'name': 'image_info', 'arguments': {'image_id': 'image_0'}}})
    result = subprocess.run([server['command'], *server['args']],
        input='\n'.join(map(json.dumps, requests))+'\n', text=True, capture_output=True, check=True)
    responses = [json.loads(line)['result'] for line in result.stdout.splitlines()]
    assert all(not row.get('isError') for row in responses[:6])
    assert responses[6]['isError']
    assert 'remaining' in responses[6]['content'][0]['text']
    assert json.loads(responses[7]['content'][0]['text'])['edit_budget']['remaining'] == 0


def test_backend_passes_per_call_limit_without_leaking_to_other_stages(scene, monkeypatch):
    tools, _ = scene
    from cloth_agent.planner_backend import BackendResult
    backend = RemoteClaudeBackend()
    setups = []
    def invoke(**kwargs):
        setups.append(backend._image_tool_setup('/tmp/cloth_remote_test', 1))
        return BackendResult('{}', '', 0, ())
    monkeypatch.setattr(backend, '_invoke', invoke)
    request = dict(prompt='test', image_paths=[tools.job / 'image_0.png'], schema={}, system_prompt='test')
    backend.invoke(**request, image_edit_limit=6)
    backend.invoke(**request)
    assert '--edit-limit 6' in setups[0][0]
    assert 'at most 6 edit attempts' in setups[0][2]
    assert '--edit-limit' not in setups[1][0]


@pytest.mark.parametrize("claude_fails", [False, True])
@pytest.mark.parametrize("live_debug", [False, True])
def test_real_bridge_bootstrap_tool_call_and_cleanup(scene, monkeypatch, claude_fails, live_debug):
    tools, _ = scene
    # The shell and MCP server are real; only transport and the Claude model
    # process are substituted. The stub must discover tools via generated config.
    stub = tools.job / "claude_stub.py"
    stub.write_text('''import json, subprocess, sys, os, time, shlex
from pathlib import Path
config = json.loads(Path(sys.argv[sys.argv.index("--mcp-config") + 1]).read_text())
server = config["mcpServers"]["cloth_image"]
assert sys.argv[sys.argv.index('--output-format') + 1] == 'stream-json'
assert '--verbose' in sys.argv
assert '--include-partial-messages' in sys.argv
print(json.dumps({'type': 'assistant', 'message': {'content': [
    {'type': 'text', 'text': 'Inspecting the sleeve before selecting a point.'}]}}), flush=True)
request = {"jsonrpc":"2.0", "id":1, "method":"tools/call", "params":{
    "name":"rotate_image", "arguments":{"image_id":"image_0", "degrees_clockwise":90}}}
completed = subprocess.run([server["command"], *server["args"]], input=json.dumps(request)+"\\n",
    capture_output=True, text=True, check=True)
result = json.loads(completed.stdout)["result"]
assert not result.get("isError"), result
assert Path(json.loads(result["content"][0]["text"])["path"]).is_file()
image_path = json.loads(result["content"][0]["text"])["path"]
settings = json.loads(Path(sys.argv[sys.argv.index("--settings") + 1]).read_text())
for event in ("PreToolUse", "PostToolUse"):
    hook = settings["hooks"][event][0]["hooks"][0]["command"]
    subprocess.run(shlex.split(hook), input=json.dumps({"hook_event_name":event,
        "tool_name":"Read", "tool_use_id":"read-image", "tool_input":{"file_path":image_path},
        "tool_response":{"base64":"MUST_NOT_BE_LOGGED"}}), text=True, check=True)
if os.environ.get("TEST_DEBUG_MANIFEST"):
    for _ in range(60):
        debug = json.loads(Path(os.environ["TEST_DEBUG_MANIFEST"]).read_text())
        if any(v.get("read_status") == "READ_COMPLETED" for v in debug["views"]):
            assert debug["status"] == "RUNNING"
            break
        time.sleep(.05)
    else:
        raise RuntimeError("debug did not become visible before Claude finished")
print(json.dumps({"type": "result", "result": "{\\"ok\\":true}"}))
''')
    if claude_fails:
        with stub.open("a") as stream:
            stream.write("sys.exit(7)\n")
    real_run = subprocess.run
    real_popen = subprocess.Popen
    monkeypatch.setenv("CLOTH_REMOTE_IMAGE_PYTHON", sys.executable)
    def substitute(command):
        remote = command[-1]
        if not remote.startswith("rm -rf"):
            remote = re.sub(r"curl -fsSL --connect-timeout 20 --max-time 120 \S+ -o (\S+)",
                lambda m: f"cp {shlex.quote(str(tools.job / 'image_0.png'))} {m[1]}", remote)
            remote = remote.replace("claude -p", f"{shlex.quote(sys.executable)} {shlex.quote(str(stub))} -p")
        return ["sh", "-c", remote]
    def run(command, **kwargs):
        if command[0] == "curl":
            return SimpleNamespace(returncode=0, stdout='{"id":"relay"}', stderr="")
        return real_run(substitute(command), **kwargs)
    def popen(command, **kwargs):
        return real_popen(substitute(command) if command[0] == "ssh" else command, **kwargs)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    backend = RemoteClaudeBackend()
    debug = tools.job / "claude_image_tools" / "test"
    kwargs = {"debug_dir": debug} if live_debug else {}
    if live_debug:
        monkeypatch.setenv("TEST_DEBUG_MANIFEST", str(debug / "image_debug.json"))
    if claude_fails:
        from cloth_agent.planner_backend import PlannerBackendError
        with pytest.raises(PlannerBackendError, match="exited with 7") as error:
            backend.invoke(prompt="inspect", image_paths=[tools.job / "image_0.png"],
                           schema={}, system_prompt="inspect RGB", **kwargs)
        rotation = next(e for e in error.value.image_tool_events if e["tool"] == "rotate_image")
        assert not Path(rotation["result"]["path"]).parent.exists()
        if live_debug:
            saved = json.loads((debug / "image_debug.json").read_text())
            assert saved["status"] == "FAILED"
            assert saved["views"][1]["read_status"] == "READ_COMPLETED"
            assert saved["audit_complete"] is True
        return
    result = backend.invoke(prompt="inspect", image_paths=[tools.job / "image_0.png"],
                            schema={}, system_prompt="inspect RGB", **kwargs)
    assert parse_claude_json(result.stdout) == {"ok": True}
    payload = next(e for e in result.image_tool_events if e['tool'] == 'image_payload')
    assert payload['image_content']['status'] == 'VALID_IMAGE'
    rotation = next(e for e in result.image_tool_events if e["tool"] == "rotate_image")
    assert rotation["result"]["size"] == [9, 13]
    remote_path = Path(rotation["result"]["path"])
    assert not remote_path.parent.exists()
    assert "--strict-mcp-config" in result.command[-1]
    assert "mcp__cloth_image__list_images" in result.command[-1]
    assert "mcp__cloth_image__image_info" in result.command[-1]
    assert "mcp__cloth_image__view_image" in result.command[-1]
    assert "--max-turns 16" in result.command[-1]
    if live_debug:
        saved = json.loads((debug / "image_debug.json").read_text())
        assert saved["status"] == "COMPLETED"
        assert saved["audit_complete"] is True
        assert saved["views"][1]["verification"] == "VERIFIED"
        assert saved["views"][1]["read_status"] == "READ_COMPLETED"
        assert saved['views'][1]['image_delivery_status'] == 'UNAVAILABLE'
        assert Path(saved["views"][1]["path"]).is_file()
        assert "MUST_NOT_BE_LOGGED" not in (debug / "events.jsonl").read_text()


def test_standalone_offline_smoke_produces_views_and_replay(scene):
    from scripts.remote_image_tools_test import main as smoke, replay
    tools, _ = scene
    output = tools.job / "smoke"
    assert smoke([str(tools.job / "image_0.png"), "--offline", "--output-dir", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["claude_tested"] is False
    events = [json.loads(line) for line in (output / "local_tools/image_tool_calls.jsonl").read_text().splitlines()]
    views = replay(tools.job / "image_0.png", events, output / "replay")
    assert len(views) == 3
    for view, event in zip(views, events):
        with Image.open(view["local_replayed_path"]) as actual, Image.open(event["result"]["path"]) as expected:
            assert actual.tobytes() == expected.tobytes()


def test_smoke_generates_direction_chart_without_camera(tmp_path):
    from scripts.remote_image_tools_test import main as smoke
    output = tmp_path / "chart_smoke"
    assert smoke(["--offline", "--output-dir", str(output)]) == 0
    with Image.open(output / "synthetic_rgb.png") as image:
        assert image.size == (512, 384)
        assert image.getpixel((10, 10)) == (255, 0, 0)
        assert image.getpixel((10, 250)) == (0, 0, 255)
    result = json.loads((output / "result.json").read_text())
    assert result["claude_tested"] is False
