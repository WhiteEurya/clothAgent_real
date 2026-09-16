"""Live, pixel-verified local replay of remote Claude image inspection."""
from __future__ import annotations

import json
import posixpath
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from .image_tools_mcp import ImageTools


def debug_directory(images, fallback, stage):
    import uuid
    root = Path(fallback)
    for image in images:
        iteration = next((p for p in Path(image).resolve().parents
                          if re.fullmatch(r"iteration_\d+", p.name)), None)
        if iteration is not None:
            root = iteration
            break
    return root / "claude_image_tools" / f"{stage}_{uuid.uuid4().hex[:12]}"


class ImageDebugSession:
    def __init__(self, directory, images, request):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.image_dir = self.directory / "images"
        self.image_dir.mkdir()
        self.images = [Path(p).resolve() for p in images]
        for i, path in enumerate(self.images):
            shutil.copyfile(path, self.image_dir / f"image_{i}.png")
        self.tools = ImageTools(self.image_dir, len(self.images))
        self.ids = {f"image_{i}": f"image_{i}" for i in range(len(self.images))}
        self.remote_paths = {}
        self.reads = {}
        self.started = time.monotonic()
        self.request = request
        self.last_message_at = None
        self.claude_event_count = 0
        self.phase_durations = {}
        self.transfer_images = [dict(image_index=i, path=str(p), bytes=p.stat().st_size)
                                for i, p in enumerate(self.images)]
        self.state = {"schema_version": 1, "status": "RUNNING", "audit_complete": False,
            "stage": self.directory.name.rsplit("_", 1)[0], "events": [], "progress": [],
            "point_overlays": [], "views": [{**view, "source_local_path": str(self.images[i]),
                       "verification": "AWAITING_REMOTE_HASH", "read_status": "UNKNOWN"}
                      for i, view in enumerate(self.tools.views.values())], "errors": []}
        self.write("request.json", request)
        self.append_stream('claude_transcript.md',
            '# Claude public conversation\n\nOnly CLI-exposed messages are recorded; hidden reasoning and '
            'provider-internal requests are unavailable. Receipt gaps include model, tools, queueing and network time.\n\n')
        self.flush()

    def save_prompt(self, prompt):
        """Exact application-supplied input, not a reconstruction of API internals."""
        (self.directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
        (self.directory / 'system_prompt.txt').write_text(self.request['system_prompt'], encoding='utf-8')

    def consume_claude_line(self, line):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return  # Raw stdout.log still preserves incomplete/non-JSON lines.
        if not isinstance(event, dict):
            return
        elapsed = time.monotonic() - self.started
        gap = None if self.last_message_at is None else elapsed - self.last_message_at
        self.last_message_at = elapsed
        self.claude_event_count += 1
        row = {'sequence': self.claude_event_count, 'received_at': datetime.now(timezone.utc).isoformat(),
               'received_elapsed_s': elapsed, 'since_previous_event_s': gap, 'event': event}
        self.append_stream('claude_events.jsonl', json.dumps(row, ensure_ascii=False) + '\n')
        self.state['last_claude_event'] = {k: v for k, v in row.items() if k != 'event'} | {
            'type': event.get('type'), 'subtype': event.get('subtype')}
        self.state['claude_event_count'] = self.claude_event_count
        header = f"## {self.claude_event_count}. {event.get('type', 'message')} at +{elapsed:.3f}s"
        if gap is not None:
            header += f" (receipt gap {gap:.3f}s)"
        parts = [header]
        message = event.get('message') or {}
        content = message.get('content', []) if isinstance(message, dict) else []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get('type')
                if kind == 'text':
                    parts.append(str(block.get('text', '')))
                elif kind == 'thinking':
                    parts.append('Provider-exposed reasoning (may be partial):\n' + str(block.get('thinking', '')))
                elif kind == 'redacted_thinking':
                    parts.append('Provider withheld reasoning; no text available.')
                elif kind == 'tool_use':
                    parts.append('Tool input: ' + json.dumps(block, ensure_ascii=False))
                elif kind == 'tool_result':
                    parts.append('Tool result: ' + json.dumps(block, ensure_ascii=False))
        if event.get('type') == 'result':
            self.write('claude_result.json', event)
            self.state['claude_metrics'] = {key: event[key] for key in (
                'duration_ms', 'duration_api_ms', 'num_turns', 'usage', 'modelUsage',
                'total_cost_usd', 'stop_reason', 'subtype', 'is_error') if key in event}
            parts.append(json.dumps(event, ensure_ascii=False, indent=2))
        if len(parts) == 1:
            parts.append(json.dumps(event, ensure_ascii=False))
        self.append_stream('claude_transcript.md', '\n\n'.join(parts) + '\n\n')
        self.flush()
        return self.state['last_claude_event']

    def save_timings(self):
        summary = {'phases_s': self.phase_durations, 'images': self.transfer_images,
            'note': 'SSH includes download, hash, setup and Claude; phases overlap. Missing durations are unknown, not zero. '
                    'Claude message receipt gaps are not pure inference durations.',
            'claude_metrics': self.state.get('claude_metrics', {})}
        self.write('timing.json', summary)
        lines = ['# Transfer and invocation timing', '', summary['note'], '',
            '| Image | Bytes | Upload (s) | Download (s) | SHA256 (s) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for row in self.transfer_images:
            values = [f"{row[k]:.3f}" if k in row else 'unknown' for k in ('upload_s', 'download_s', 'hash_s')]
            lines.append(f"| image_{row['image_index']} / {Path(row['path']).name} | {row['bytes']} | " + ' | '.join(values) + ' |')
        lines += ['', '| Phase | Duration (s) |', '| --- | ---: |']
        lines += [f'| {key} | {value:.3f} |' for key, value in self.phase_durations.items()]
        (self.directory / 'timing.md').write_text('\n'.join(lines), encoding='utf-8')

    def _point_overlay(self, view, point, label, sequence):
        with Image.open(view["path"]) as original:
            image = original.convert("RGB")
        x, y = point
        draw = ImageDraw.Draw(image)
        radius = max(5, min(image.size)//70)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="yellow", width=3)
        draw.line((x-radius*2, y, x+radius*2, y), fill="red", width=2)
        draw.line((x, y-radius*2, x, y+radius*2), fill="red", width=2)
        draw.text((5, 5), label + " (debug annotation)", fill="yellow", stroke_width=1, stroke_fill="black")
        directory = self.directory / "points"
        directory.mkdir(exist_ok=True)
        path = directory / f"{sequence:03d}_{label}.png"
        image.save(path)
        self.state["point_overlays"].append({"path": str(path), "label": label,
            "source_image_id": view["image_id"], "pixel_xy": point,
            "source_verification": view["verification"], "event_sequence": sequence,
            "note": "Local debug annotation; this overlay was NOT sent to Claude."})

    def write(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def flush(self):
        self.state["elapsed_s"] = time.monotonic() - self.started
        self.write("image_debug.json", self.state)

    def progress(self, stage, event, duration_s, details):
        self.state["progress"].append(dict(stage=stage, event=event, duration_s=duration_s,
                                          elapsed_s=time.monotonic()-self.started, **details))
        if duration_s is not None and event in {'finished', 'measured', 'completed', 'failed'}:
            self.phase_durations[stage] = duration_s
            match = re.fullmatch(r'(upload|remote_download|remote_hash)_(\d+)', stage)
            if match and int(match[2]) < len(self.transfer_images):
                self.transfer_images[int(match[2])][match[1].removeprefix('remote_') + '_s'] = duration_s
            self.save_timings()
        self.flush()

    def append_stream(self, name, text):
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(text)

    def _view(self, image_id):
        return next(v for v in self.state["views"] if v["image_id"] == image_id)

    def _match_reads(self):
        for view in self.state["views"]:
            reads = [r for r in self.reads.values() if self.remote_paths.get(r.get("path")) == view["image_id"]]
            # A successful Read proves the tool returned the image, not that
            # the model understood it. Keep all requests, including retries.
            statuses = {r["status"] for r in reads}
            view["read_status"] = ("READ_COMPLETED" if "completed" in statuses else
                "READ_STARTED" if "started" in statuses else "READ_FAILED" if "failed" in statuses else
                "NO_READ_RECORDED" if self.state["audit_complete"] else "UNKNOWN")
            view["reads"] = reads

    def consume(self, event):
        row = dict(event, received_elapsed_s=time.monotonic()-self.started)
        self.state["events"].append(row)
        self.append_stream("events.jsonl", json.dumps(row, ensure_ascii=False) + "\n")
        try:
            if event.get("kind") == "session":
                self.state["remote_setup"] = event
                for remote in event.get("images", []):
                    view = self._view(remote["image_id"])
                    self.remote_paths[remote["path"]] = view["image_id"]
                    view.update(remote_rgb_sha256=remote.get("rgb_sha256"),
                                remote_pillow_version=remote.get("pillow_version"),
                                verification="VERIFIED" if view["rgb_sha256"] == remote.get("rgb_sha256") else "HASH_MISMATCH")
            elif event.get("kind") == "read":
                key = event.get("tool_use_id") or event["event_id"]
                read = self.reads.setdefault(key, {"tool_use_id": key})
                path = event.get("arguments", {}).get("file_path")
                if isinstance(path, str) and not posixpath.isabs(path) and self.state.get("remote_setup"):
                    remote_job = posixpath.dirname(self.state["remote_setup"]["images"][0]["path"])
                    path = posixpath.normpath(posixpath.join(remote_job, path))
                read.update(path=path, status=event["status"],
                            error=event.get("error"))
                if event["status"] == "started":
                    read["started_ns"] = event.get("timestamp_ns")
                elif read.get("started_ns") and event.get("timestamp_ns"):
                    read["duration_s"] = (event["timestamp_ns"]-read["started_ns"])/1e9
            elif event.get("tool") == "audit_finished":
                self.state["audit_complete"] = True
            elif event.get("status") == "ok" and event.get("tool") in {
                    "rotate_image", "crop_image", "resize_image"}:
                if event.get('result', {}).get('reused'):
                    self._match_reads()
                    self.flush()
                    return
                args = dict(event["arguments"])
                parent = args["image_id"]
                args["image_id"] = self.ids[parent]
                result = self.tools.call(event["tool"], args)
                remote = event["result"]
                remote_id = remote["image_id"]
                self.ids[remote_id] = result["image_id"]
                self.remote_paths[remote["path"]] = remote_id
                verified = (result["rgb_sha256"] == remote.get("rgb_sha256") and
                            self._view(parent)["verification"] == "VERIFIED")
                self.state["views"].append({**result, "image_id": remote_id,
                    "parent_image_id": parent, "remote_path": remote["path"],
                    "operation": event["tool"], "arguments": event["arguments"],
                    "duration_s": event.get("duration_s"), "read_status": "UNKNOWN",
                    "remote_rgb_sha256": remote.get("rgb_sha256"),
                    "remote_pillow_version": remote.get("pillow_version"),
                    "verification": "VERIFIED" if verified else "UNVERIFIED_REPLAY"})
            elif event.get("tool") == "map_point" and event.get("status") == "ok":
                source = self._view(event["arguments"]["image_id"])
                target = self._view(f"image_{event['result']['original_image_index']}")
                self._point_overlay(source, event["arguments"]["pixel_xy"], "selected_in_view", len(self.state["events"]))
                self._point_overlay(target, event["result"]["pixel_xy"], "mapped_to_original", len(self.state["events"]))
        except Exception as exc:
            row["debug_error"] = f"{type(exc).__name__}: {exc}"
            self.state["errors"].append(row["debug_error"])
        self._match_reads()
        self.flush()

    def finish(self, status, error=None):
        self.state.update(status=status, error=error)
        self.save_timings()
        self._match_reads()
        self.flush()
