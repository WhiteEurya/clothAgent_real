import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.image_tools_mcp import ImageTools
from cloth_agent.planner_backend import RemoteClaudeBackend
from cloth_agent.fold_exploration_viser import _FoldViserState, _iter_images, _image_tool_summary


def prepare(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    Image.new("RGB", (40, 30), "green").save(remote / "image_0.png")
    Image.new("RGB", (30, 40), "blue").save(remote / "image_1.png")
    tools = ImageTools(remote, 2)
    directory = tmp_path / "iteration_001" / "claude_image_tools" / "planning_call"
    debug = ImageDebugSession(directory, [remote / "image_0.png", remote / "image_1.png"],
                              {"prompt": "Inspect current and reference RGB"})
    debug.consume({"kind": "session", "images": list(tools.views.values()), "tool": "ready", "status": "ok"})
    return tools, debug


def latest(tools):
    return json.loads((tools.job / "image_tool_calls.jsonl").read_text().splitlines()[-1])


def test_multi_image_replay_read_status_point_overlays_and_hash_failure(tmp_path):
    tools, debug = prepare(tmp_path)
    rotated = tools.call("rotate_image", {"image_id": "image_1", "degrees_clockwise": 90})
    debug.consume(latest(tools))
    view = debug.state["views"][-1]
    assert view["original_image_index"] == 1
    assert view["verification"] == "VERIFIED"
    assert view["read_status"] == "UNKNOWN"
    for status, timestamp in [("started", 100), ("failed", 1_000_000_100)]:
        debug.consume({"kind": "read", "tool": "Read", "tool_use_id": "failed_read", "status": status,
            "timestamp_ns": timestamp, "arguments": {"file_path": rotated["path"]}, "error": "read error"})
    assert view["read_status"] == "READ_FAILED"
    assert view["reads"][0]["duration_s"] == 1
    debug.consume({"kind": "read", "tool": "Read", "tool_use_id": "retry", "status": "completed",
                   "arguments": {"file_path": Path(rotated["path"]).name}})
    assert view["read_status"] == "READ_COMPLETED"
    tools.call("map_point", {"image_id": rotated["image_id"], "pixel_xy": [3, 4]})
    debug.consume(latest(tools))
    assert len(debug.state["point_overlays"]) == 2
    assert debug.state["point_overlays"][1]["source_image_id"] == "image_1"
    tools.call("resize_image", {"image_id": rotated["image_id"], "scale": 2})
    event = latest(tools)
    event["result"]["rgb_sha256"] = "wrong pixels"
    debug.consume(event)
    assert debug.state["views"][-1]["verification"] == "UNVERIFIED_REPLAY"
    debug.consume({"tool": "audit_finished", "status": "ok"})
    debug.finish("FAILED", "model failed after operations")
    assert debug.state["views"][0]["read_status"] == "NO_READ_RECORDED"
    saved = json.loads((debug.directory / "image_debug.json").read_text())
    assert saved["status"] == "FAILED"
    assert saved["views"][2]["read_status"] == "READ_COMPLETED"
    assert all(Path(v["path"]).exists() for v in saved["views"])


def test_incomplete_replay_does_not_invent_image_or_read_success(tmp_path):
    _, debug = prepare(tmp_path)
    debug.consume({"tool": "crop_image", "status": "ok",
        "arguments": {"image_id": "missing_parent", "box": [0, 0, 2, 2]},
        "result": {"image_id": "missing_child"}})
    debug.consume({"tool": "rotate_image", "status": "error", "error": "bad rotation", "arguments": {}})
    debug.finish("INTERRUPTED", "KeyboardInterrupt")
    assert len(debug.state["views"]) == 2
    assert debug.state["audit_complete"] is False
    assert all(v["read_status"] == "UNKNOWN" for v in debug.state["views"])
    assert debug.state["errors"]
    summary = _image_tool_summary(debug.state)
    assert "incomplete" in summary and "bad rotation" in summary and "INTERRUPTED" in summary


def test_cached_edit_does_not_duplicate_debug_views(tmp_path):
    tools, debug = prepare(tmp_path)
    args = {'image_id': 'image_0', 'degrees_clockwise': 90}
    first = tools.call('rotate_image', args)
    debug.consume(latest(tools))
    tools.call('rotate_image', args)
    debug.consume(latest(tools))
    assert len(debug.state['views']) == 3
    assert debug.state['views'][-1]['image_id'] == first['image_id']
    assert debug.state['views'][-1]['verification'] == 'VERIFIED'
    assert not debug.state['errors']


class Folder:
    removed = False

    def remove(self):
        self.removed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Gui:
    def __init__(self):
        self.images = []
        self.markdowns = []

    def add_folder(self, *args, **kwargs):
        return Folder()

    def add_markdown(self, content):
        result = Folder()
        result.content = content
        self.markdowns.append(result)
        return result

    def add_html(self, content):
        result = Folder()
        result.content = content
        return result

    def add_image(self, image, label):
        result = Folder()
        result.label = label
        self.images.append(result)
        return result


def test_viser_renders_live_views_once_and_updates_read_status(tmp_path):
    tools, debug = prepare(tmp_path)
    tools.call("rotate_image", {"image_id": "image_0", "degrees_clockwise": 90})
    event = latest(tools)
    debug.consume(event)
    gui = Gui()
    viewer = _FoldViserState(SimpleNamespace(gui=gui), tmp_path)
    iteration = debug.directory.parent.parent
    debug.flush(force=True)  # Render the next persisted snapshot, now batched in production.
    viewer._render_image_tools(iteration)
    assert len(gui.images) == 3
    assert _iter_images(iteration) == []  # managed images must not appear twice
    debug.consume({"kind": "read", "tool": "Read", "status": "completed", "tool_use_id": "r",
                   "arguments": {"file_path": event["result"]["path"]}})
    debug.flush(force=True)
    viewer._render_image_tools(iteration)
    assert len(gui.images) == 3
    assert any("READ_COMPLETED" in p.content for p in viewer.tool_image_panels.values())
    debug.append_stream("stdout.log", "full model output\n" * 1000)
    viewer._render_image_tools(iteration)
    panel = viewer.tool_raw_panels[(debug.directory / "image_debug.json", "stdout.log")]
    assert panel.content.count("full model output") == 1000


def test_viser_third_iteration_evicts_all_first_iteration_handles_keeps_files(tmp_path):
    tools, debug = prepare(tmp_path)
    gui = Gui()
    viewer = _FoldViserState(SimpleNamespace(gui=gui), tmp_path)
    first = tmp_path / 'iteration_001'
    second = tmp_path / 'iteration_002'
    second.mkdir()
    viewer.update()
    assert viewer.visible_iterations == {first, second}
    old_images = list(viewer.tool_image_handles.values())
    old_panels = list(viewer.tool_panels.values())
    old_folders = list(viewer.image_folders.values()) + list(viewer.tool_raw_folders.values())
    trajectory = Folder()
    viewer.path_handles[first] = trajectory
    viewer.path_mtimes[first] = 123
    third = tmp_path / 'iteration_003'
    third.mkdir()
    viewer.update()
    assert viewer.visible_iterations == {second, third}
    assert old_images and all(handle.removed for handle in old_images)
    assert all(handle.removed for handle in old_panels + old_folders)
    assert trajectory.removed
    assert first not in viewer.path_mtimes
    assert not viewer.tool_image_handles and not viewer.tool_raw_mtimes
    assert (debug.directory / 'image_debug.json').is_file()
    assert all(Path(view['path']).exists() for view in debug.state['views'])
    viewer.update()
    assert not viewer.tool_image_handles  # never re-add the oldest iteration


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_audit_survives_timeout_and_interrupt(tmp_path, monkeypatch, interrupted):
    tools, _ = prepare(tmp_path)
    tools.call("rotate_image", {"image_id": "image_0", "degrees_clockwise": 90})
    event = latest(tools)
    backend = RemoteClaudeBackend()
    def invoke(**kwargs):
        if interrupted:
            backend._remote_timings("__CLOTH_IMAGE_TOOL__ " + json.dumps(event))
            raise KeyboardInterrupt()
        command = [sys.executable, "-c",
            "import sys,time,subprocess; print(sys.argv[1], file=sys.stderr, flush=True); "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(5)",
            "__CLOTH_IMAGE_TOOL__ " + json.dumps(event)]
        return backend._run_streaming(command, "", .3)
    monkeypatch.setattr(backend, "_invoke", invoke)
    directory = tmp_path / "partial_debug"
    with pytest.raises(KeyboardInterrupt if interrupted else subprocess.TimeoutExpired):
        backend.invoke(prompt="inspect", image_paths=[tools.job / "image_0.png"], schema={},
                       system_prompt="inspect", debug_dir=directory)
    saved = json.loads((directory / "image_debug.json").read_text())
    assert saved["status"] == ("INTERRUPTED" if interrupted else "FAILED")
    assert saved["audit_complete"] is False
    assert len(saved["views"]) == 2
    assert Path(saved["views"][1]["path"]).exists()
    assert saved["views"][1]["read_status"] == "UNKNOWN"
    assert (directory / "exception.log").is_file()
