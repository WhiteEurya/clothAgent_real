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

from .image_tools_mcp import (ImageTools, IMAGE_TOOLS, VERIFIED_DELIVERIES, image_content_summary,
                             returned_image_metadata, image_matches, verify_image_delivery)
from .claude_stream import urgent_event


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
        self.inspections = {}
        self.delivery_checks = {}
        self.started = time.monotonic()
        self._last_snapshot = float('-inf')
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

    def consume_claude_line(self, line, *, on_event=None):
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
        if event.get('subtype') == 'responses_diagnostic':
            self.append_stream('responses_diagnostics.jsonl', json.dumps(event, ensure_ascii=False) + '\n')
            self.write('responses_last_request.json', event)
        self.state['last_claude_event'] = {k: v for k, v in row.items() if k != 'event'} | {
            'type': event.get('type'), 'subtype': event.get('subtype')}
        self.state['claude_event_count'] = self.claude_event_count
        counts = self.state.setdefault('claude_event_counts', {})
        label = str(event.get('type', 'unknown'))
        if event.get('subtype'):
            label += '.' + str(event['subtype'])
        counts[label] = counts.get(label, 0) + 1
        # Keep every raw event above, but do not turn status/token notifications
        # into thousands of transcript headings or rerun image verification.
        if event.get('type') in {'system', 'stream_event'} and not urgent_event(event):
            if on_event:
                on_event(event)
            self.flush()
            return self.state['last_claude_event']
        header = f"## {self.claude_event_count}. {event.get('type', 'message')} at +{elapsed:.3f}s"
        if gap is not None:
            header += f" (receipt gap {gap:.3f}s)"
        parts = [header]
        inspections_changed = False
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
                    tool = block.get('name', '')
                    if block.get('id') and (tool == 'Read' or tool.removeprefix('mcp__cloth_image__') in IMAGE_TOOLS):
                        inspections_changed = True
                        inspection = self.inspections.setdefault(block['id'], {'tool_use_id': block['id']})
                        inspection.update(tool=tool, arguments=block.get('input', {}))
                    parts.append('Tool input: ' + json.dumps(block, ensure_ascii=False))
                elif kind == 'tool_result':
                    inspections_changed = True
                    identity = block.get('tool_use_id')
                    summary = image_content_summary(block.get('content'), on_image=self._save_returned_image)
                    inspection = self.inspections.setdefault(identity, {'tool_use_id': identity})
                    inspection.update(stream_content=summary, stream_error=bool(block.get('is_error')))
                    metadata = returned_image_metadata(block.get('content'))
                    if metadata:
                        inspection['image_metadata'] = metadata
                    self.append_stream('image_delivery.jsonl', json.dumps({
                        'tool_use_id': identity, 'boundary': 'cli_tool_result',
                        'image_content': summary, 'image_metadata': inspection.get('image_metadata', {}),
                        'is_error': inspection['stream_error']}, ensure_ascii=False) + '\n')
                    # Raw streams retain exact content. Keep the readable transcript
                    # useful instead of rendering megabytes of base64 in Viser.
                    parts.append('Tool result: ' + json.dumps({
                        'tool_use_id': identity, 'image_content': summary,
                        'metadata': inspection.get('image_metadata', {}), 'is_error': inspection['stream_error']}, ensure_ascii=False)
                        if summary['image_count'] or summary['errors'] else
                        'Tool result: ' + json.dumps(block, ensure_ascii=False))
        if event.get('type') == 'result':
            self.write('claude_result.json', event)
            self.state['claude_metrics'] = {key: event[key] for key in (
                'duration_ms', 'duration_api_ms', 'num_turns', 'usage', 'modelUsage',
                'total_cost_usd', 'stop_reason', 'subtype', 'is_error') if key in event}
            parts.append(json.dumps(event, ensure_ascii=False, indent=2))
        if len(parts) == 1:
            parts.append(json.dumps(event, ensure_ascii=False))
        self.append_stream('claude_transcript.md', '\n\n'.join(parts) + '\n\n')
        if inspections_changed:
            self._match_reads()
        if on_event:
            on_event(event)
        self.flush(force=event.get('type') == 'result')
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

    def flush(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_snapshot < 1:
            return
        self.state["elapsed_s"] = now - self.started
        self.write("image_debug.json", self.state)
        self._last_snapshot = now

    def progress(self, stage, event, duration_s, details):
        self.state["progress"].append(dict(stage=stage, event=event, duration_s=duration_s,
                                          elapsed_s=time.monotonic()-self.started, **details))
        if stage in {'claude_stream', 'claude_text'}:
            self.append_stream('claude_transcript.md',
                f'## {stage} at +{time.monotonic()-self.started:.3f}s\n\n' +
                json.dumps(details, ensure_ascii=False) + '\n\n')
        if stage == 'claude_stream':
            self.state['claude_stream'] = details
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

    def _save_returned_image(self, raw, details):
        directory = self.directory / 'returned_images'
        directory.mkdir(exist_ok=True)
        suffix = '.jpg' if details['mime_type'] == 'image/jpeg' else '.png'
        path = directory / (details['encoded_sha256'] + suffix)
        if not path.exists():
            path.write_bytes(raw)
        return {'saved_path': str(path)}

    def _delivery_check(self, summary, view):
        if image_matches(summary, view):
            return {'status': 'VERIFIED', 'method': 'exact_rgb_sha256'}
        images = summary.get('images', []) if isinstance(summary, dict) else []
        raw = None
        key = (view['image_id'], view.get('rgb_sha256'), json.dumps(summary, sort_keys=True))
        if key not in self.delivery_checks:
            if len(images) == 1 and images[0].get('saved_path'):
                path = Path(images[0]['saved_path'])
                if path.resolve().parent == self.directory / 'returned_images' and not path.is_symlink():
                    raw = path.read_bytes()
            self.delivery_checks[key] = verify_image_delivery(summary, view, view['path'], raw)
        return self.delivery_checks[key]

    def _match_reads(self):
        for view in self.state["views"]:
            reads = [r for r in self.reads.values() if self.remote_paths.get(r.get("path")) == view["image_id"]]
            # Legacy lifecycle label, deliberately NOT an image-content gate.
            statuses = {r["status"] for r in reads}
            view["read_status"] = ("READ_COMPLETED" if "completed" in statuses else
                "READ_STARTED" if "started" in statuses else "READ_FAILED" if "failed" in statuses else
                "NO_READ_RECORDED" if self.state["audit_complete"] else "UNKNOWN")
            view["reads"] = reads
            inspections = []
            for record in self.inspections.values():
                tool = record.get('tool', '')
                if not record.get('tool_use_id') or not (tool == 'Read' or tool.removeprefix('mcp__cloth_image__') in IMAGE_TOOLS):
                    continue
                args = record.get('arguments', {})
                metadata = record.get('image_metadata', {})
                path = args.get('file_path')
                if isinstance(path, str) and not posixpath.isabs(path) and self.state.get('remote_setup'):
                    path = posixpath.normpath(posixpath.join(
                        posixpath.dirname(self.state['remote_setup']['images'][0]['path']), path))
                image_id = metadata.get('image_id') or self.remote_paths.get(path)
                if image_id is None and record.get('tool', '').endswith('__view_image'):
                    image_id = args.get('image_id')
                if image_id != view['image_id']:
                    continue
                hook = record.get('hook_content')
                stream = record.get('stream_content')
                check = self._delivery_check(stream, view)
                emitted = check['status']
                # Metadata must describe this exact source, including remote identity.
                metadata_valid = not metadata or (
                    metadata.get('image_id') == view['image_id'] and
                    metadata.get('size') == view['size'] and
                    metadata.get('rgb_sha256') == view['rgb_sha256'] and
                    self.remote_paths.get(metadata.get('path')) == view['image_id'])
                if not metadata_valid or (emitted == 'VERIFIED_TRANSCODE' and
                        tool != 'Read' and not metadata):
                    emitted = 'IDENTITY_MISMATCH'
                    check = {'status': emitted, 'reason': 'missing or mismatched source metadata'}
                returned = ('VERIFIED' if image_matches(hook, view) else
                    emitted if emitted in VERIFIED_DELIVERIES and stream and
                        image_matches(hook, stream['images'][0]) else
                    'UNAVAILABLE' if hook is not None else 'UNKNOWN')
                valid = (emitted in VERIFIED_DELIVERIES and returned != 'UNAVAILABLE' and
                         record.get('hook_status') != 'failed' and not record.get('stream_error'))
                inspections.append({**record, 'returned_image_status': returned,
                    'stream_image_status': emitted, 'delivery_check': check,
                    'image_delivery_status': emitted if valid else
                    emitted if emitted in {'CONTENT_MISMATCH', 'SIZE_MISMATCH', 'IDENTITY_MISMATCH'} else
                    'UNAVAILABLE' if returned == 'UNAVAILABLE' or record.get('stream_error')
                    or record.get('hook_status') == 'failed' else emitted})
            view['image_inspections'] = inspections
            view.pop('delivered_image', None)
            accepted = [record for record in inspections if record['image_delivery_status'] in VERIFIED_DELIVERIES]
            if accepted:
                best = next((record for record in accepted if record['image_delivery_status'] == 'VERIFIED'), accepted[0])
                view['delivered_image'] = {**best['stream_content']['images'][0],
                    'tool_use_id': best['tool_use_id'], 'validation': best['delivery_check']}
            for field in ('returned_image_status', 'stream_image_status', 'image_delivery_status'):
                statuses = {record[field] for record in inspections}
                view[field] = next((s for s in ('VERIFIED', 'VERIFIED_TRANSCODE', 'IDENTITY_MISMATCH',
                    'SIZE_MISMATCH', 'CONTENT_MISMATCH', 'UNAVAILABLE') if s in statuses), 'UNKNOWN')
        self.state['delivery_evidence_note'] = (
            'VERIFIED means exact RGB; VERIFIED_TRANSCODE means a same-size JPEG passed bounded full-pixel '
            're-encoding comparison with its verified source. Actual returned bytes are saved separately. '
            'Neither proves provider receipt or model understanding.')

    def consume(self, event):
        row = dict(event, received_elapsed_s=time.monotonic()-self.started)
        self.state["events"].append(row)
        self.append_stream("events.jsonl", json.dumps(row, ensure_ascii=False) + "\n")
        try:
            if event.get('kind') in {'read', 'tool_lifecycle'}:
                tool = event.get('tool', '')
                if tool == 'Read' or tool.removeprefix('mcp__cloth_image__') in IMAGE_TOOLS:
                    identity = event.get('tool_use_id')
                    record = self.inspections.setdefault(identity, {'tool_use_id': identity})
                    record.update(tool=tool, arguments=event.get('arguments', {}), hook_status=event.get('status'))
                    if 'image_content' in event:
                        record['hook_content'] = event['image_content']
                    if event.get('image_metadata'):
                        record['image_metadata'] = event['image_metadata']
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
            elif event.get('kind') == 'orientation_guard':
                self.append_stream('claude_transcript.md',
                    '## Host orientation audit\n\n' + json.dumps(event, ensure_ascii=False, indent=2) + '\n\n')
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
        self.flush(force=True)
