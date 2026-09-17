"""Job-local RGB inspection tools, deployable standalone with Python + Pillow.

Pixel coordinates denote pixel centers; every derived view stores its exact
affine map back to the originally supplied RGB. No camera or robot imports.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
import json
import math
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

SERVER_NAME = "cloth_image"
MAX_PIXELS = 16_777_216
MAX_SIDE = 8192
MAX_VIEWS = 24
MAX_CALLS = 64
EDIT_TOOLS = frozenset({'rotate_image', 'crop_image', 'resize_image'})
IMAGE_TOOLS = EDIT_TOOLS | {'view_image'}
IDENTITY = [1., 0., 0., 0., 1., 0.]
INSTRUCTIONS = (
    "Use cloth_image tools to inspect the supplied RGB: rotate_image (positive "
    "degrees clockwise), crop_image, resize_image, view_image, image_info, map_point. "
    "Use view_image(image_id) to SEE an original or saved view. Each edit directly returns "
    "the resulting IMAGE alongside its metadata: inspect it without a redundant Read. "
    "Tool text or a saved path alone is not a visual observation. Originals are "
    "image_0, image_1, etc., matching the supplied image manifest. Derived views "
    "are RGB inspection views; an orientation response may select one as the exact "
    "downstream Molmo input. map_point can recover original image_index and pixel_xy. "
    "When the response schema includes image_id, return the exact source image/view ID "
    "and coordinates in THAT view; the host performs the mapping. Never pair a derived "
    "view ID with coordinates already mapped to the original. Otherwise map_point "
    "before returning original-image pixels. "
    "Final executable pixels must belong to the current Camera-A original, never "
    "a static reference or a crop/rotated view. Rxxx IDs keep their original "
    "identity. An image rotation never changes the task's garment-left/right "
    "definition. No mirroring, generated fabric, depth, or robot access is provided."
    " Each tool result includes inspection_history with existing image paths and validated image-return counts. "
    "Use list_images to recover this inventory. Reuse an existing suitable view rather than "
    "recreating it. Identical edits return the same image. If no image is visible, report "
    "IMAGE_UNAVAILABLE; do not repeatedly resize/crop to repair image delivery. Inspect useful views, "
    "then finish the requested decision; do not inspect indefinitely."
)


def _tool(name, description, properties):
    return {"name": name, "description": description, "inputSchema": {
        "type": "object", "additionalProperties": False,
        "properties": {"image_id": {"type": "string"}, **properties},
        "required": ["image_id", *properties],
    }}


TOOLS = [
    {"name": "list_images", "description": "List saved images, operations and image-return records; reuse existing views.",
     "inputSchema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    _tool("image_info", "Get image dimensions and original identity; use view_image for pixels.", {}),
    _tool("view_image", "Return a saved RGB image as image content, plus identity and pixel metadata; no edit.", {}),
    _tool("rotate_image", "Rotate RGB by degrees clockwise, expanding the canvas; directly returns the image.",
          {"degrees_clockwise": {"type": "number", "minimum": -360, "maximum": 360}}),
    _tool("crop_image", "Crop [left, top, right, bottom], right/bottom exclusive; directly returns the image.",
          {"box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}}}),
    _tool("resize_image", "Enlarge/reduce preserving aspect ratio; directly returns the image. Scale > 1 adds no detail.",
          {"scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 8}}),
    _tool("map_point", "Map a derived-view point to its original RGB. Rejects rotation padding/out-of-image points.",
          {"pixel_xy": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}}),
]
TOOL_NAMES = tuple("mcp__" + SERVER_NAME + "__" + tool["name"] for tool in TOOLS)


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()


def image_content_summary(response):
    """Validate Read, MCP and CLI image blocks; return metadata, never base64.

    This verifies content at the observed boundary, not provider receipt or model
    understanding. CLI may resize images; mismatched pixels are not verified.
    """
    images = []
    errors = []

    def inspect(value):
        if isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, dict):
            if value.get('type') == 'image':
                source = value.get('source') if isinstance(value.get('source'), dict) else {}
                file = value.get('file') if isinstance(value.get('file'), dict) else {}
                data = (source.get('data') if source.get('type') == 'base64' else
                        file.get('base64') if isinstance(file, dict) and 'base64' in file else value.get('data'))
                mime = source.get('media_type') or file.get('type') or value.get('mimeType')
                try:
                    if not isinstance(data, str) or not data or len(data) > 90_000_000:
                        raise ValueError('missing/empty/oversized base64 image')
                    raw = base64.b64decode(data, validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        _size(*image.size)
                        actual_mime = Image.MIME.get(image.format)
                        if mime != actual_mime:
                            raise ValueError('image MIME does not match encoded bytes')
                        images.append({'mime_type': mime, 'bytes': len(raw),
                            'size': list(image.size), 'rgb_sha256': pixel_hash(image),
                            'encoded_sha256': hashlib.sha256(raw).hexdigest()})
                except (ValueError, TypeError, OSError, Image.DecompressionBombError) as exc:
                    errors.append(str(exc))
                return
            # MCP hooks wrap content in some CLI releases. Do not interpret JSON
            # strings or paths in text as image content.
            for key in ('content', 'result', 'tool_response'):
                if key in value:
                    inspect(value[key])

    inspect(response)
    return {'status': 'INVALID_IMAGE' if errors else 'VALID_IMAGE' if images else 'NO_IMAGE',
            'image_count': len(images), 'images': images, 'errors': errors}


def returned_image_metadata(response):
    """Extract our tool's identity metadata from its text block, not its pixels."""
    if isinstance(response, list):
        for item in response:
            metadata = returned_image_metadata(item)
            if metadata:
                return metadata
    elif isinstance(response, dict):
        if response.get('type') == 'text':
            try:
                value = json.loads(response.get('text', ''))
                if isinstance(value, dict) and isinstance(value.get('image_id'), str) and isinstance(value.get('path'), str):
                    return {key: value.get(key) for key in ('image_id', 'path', 'size', 'rgb_sha256')}
            except (ValueError, TypeError):
                pass
        for key in ('content', 'result', 'tool_response'):
            if key in response:
                metadata = returned_image_metadata(response[key])
                if metadata:
                    return metadata
    return {}


def image_matches(summary, view):
    return (isinstance(summary, dict) and summary.get('status') == 'VALID_IMAGE'
            and summary.get('image_count') == 1 and len(summary.get('images', [])) == 1
            and summary['images'][0].get('rgb_sha256') == view.get('rgb_sha256')
            and summary['images'][0].get('size') == view.get('size'))


def audit(job, event):
    event = dict(event, event_id=uuid.uuid4().hex, timestamp_ns=time.time_ns())
    # One append keeps independent Read hooks and the MCP process from mixing
    # partial records. Never include Read's response (it may contain image data).
    data = (json.dumps(event, allow_nan=False) + "\n").encode()
    fd = os.open(job / "image_tool_calls.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def read_hook(job):
    payload = json.load(sys.stdin)
    tool = payload.get("tool_name", "")
    if tool != "Read" and not tool.startswith("mcp__cloth_image__"):
        return
    event = payload.get("hook_event_name")
    status = {"PreToolUse": "started", "PostToolUse": "completed",
              "PostToolUseFailure": "failed"}.get(event, "unknown")
    args = payload.get("tool_input") or {}
    response = payload.get('tool_response')
    if isinstance(response, dict) and (response.get('isError') or response.get('is_error')):
        status = 'failed'
    details = {}
    if event == 'PostToolUse' and (tool == 'Read' or tool.removeprefix('mcp__cloth_image__') in IMAGE_TOOLS):
        details['image_content'] = image_content_summary(response)
        details['image_metadata'] = returned_image_metadata(response)
        path = args.get('file_path') if tool == 'Read' else details['image_metadata'].get('path')
        details['image_content']['identity_status'] = 'UNVERIFIED'
        try:
            candidate = job / path if isinstance(path, str) else None
            if candidate is None or candidate.is_symlink() or candidate.resolve().parent != job:
                raise ValueError('no job-local image identity')
            with Image.open(candidate) as image:
                expected = {'size': list(image.size), 'rgb_sha256': pixel_hash(image)}
            details['image_content']['identity_status'] = (
                'VERIFIED' if image_matches(details['image_content'], expected) else 'MISMATCH')
        except (ValueError, OSError) as exc:
            details['image_content']['identity_error'] = str(exc)
    audit(job, {"kind": "read" if tool == "Read" else "tool_lifecycle",
                "tool": tool, "status": status,
                "tool_use_id": payload.get("tool_use_id"),
                "arguments": {"file_path": args.get("file_path")} if tool == "Read" else args,
                "error": str(payload.get("error", "")) if status == "failed" else None, **details})


def orientation_guard(job, payload):
    """Audit a structured orientation result; never contradict missing pixels.

    Kept under the historical hook name for existing CLI configurations. No
    automatic correction is granted: content problems are diagnosed separately
    from visual ambiguity, with no renewed edit/time/turn budget.
    """
    event = payload.get('hook_event_name')
    candidate = None
    if event == 'PreToolUse' and payload.get('tool_name') == 'StructuredOutput':
        candidate = payload.get('tool_input')
    elif event == 'Stop':
        text = payload.get('last_assistant_message', '')
        if isinstance(text, str):
            try:
                candidate = json.loads(text[text.index('{'):text.rindex('}') + 1])
            except (ValueError, json.JSONDecodeError):
                pass
    else:
        return {}

    classification = 'INVALID_RESULT'
    if isinstance(candidate, dict):
        classification = candidate.get('failure_reason') or ('READY' if candidate.get('status') == 'READY' else 'UNCLASSIFIED')
    audit(job, {'kind': 'orientation_guard', 'tool': 'orientation_correction',
                'status': 'not_requested',
                'trigger': event, 'classification': classification, 'candidate': candidate,
                'feedback': {}, 'note': 'Classification is model-reported; host verifies image content separately.'})
    return {}


def forward_audit(job):
    path = job / "image_tool_calls.jsonl"
    path.touch(exist_ok=True)
    with path.open(encoding="utf-8") as stream:
        pending = ""
        while True:
            pending += stream.read()
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                print("__CLOTH_IMAGE_TOOL__ " + line, file=sys.stderr, flush=True)
            time.sleep(.1)


def compose(a, b):
    """Affine a(b(point)), in Pillow's six-coefficient order."""
    return [a[0]*b[0]+a[1]*b[3], a[0]*b[1]+a[1]*b[4], a[0]*b[2]+a[1]*b[5]+a[2],
            a[3]*b[0]+a[4]*b[3], a[3]*b[1]+a[4]*b[4], a[3]*b[2]+a[4]*b[5]+a[5]]


def transform_point(matrix, point):
    x, y = point
    return [matrix[0]*x + matrix[1]*y + matrix[2], matrix[3]*x + matrix[4]*y + matrix[5]]


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def _size(width, height):
    if min(width, height) < 1 or max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        raise ValueError("image exceeds size limit (8192 per side, 16 megapixels)")


class ImageTools:
    def __init__(self, job: Path, image_count: int, edit_limit: int | None = None):
        self.job = job.resolve(strict=True)
        if edit_limit is not None and (type(edit_limit) is not int or not 0 <= edit_limit <= MAX_VIEWS):
            raise ValueError('edit_limit must be an integer in [0, 24]')
        self.edit_limit = edit_limit
        self.views = {}
        self.calls = 0
        self.created = 0
        self.edit_cache = {}
        for i in range(image_count):
            image_id = f"image_{i}"
            path = self.job / f"{image_id}.png"
            if path.is_symlink() or path.resolve().parent != self.job:
                raise ValueError("original image must be inside the job")
            with Image.open(path) as image:
                _size(*image.size)
                self.views[image_id] = dict(image_id=image_id, path=str(path), size=list(image.size),
                    rgb_sha256=pixel_hash(image), pillow_version=Image.__version__,
                    original_image_index=i, original_size=list(image.size), to_original=IDENTITY[:],
                    parent_image_id=None, to_parent=IDENTITY[:])
        # The append-only audit is also the recovery ledger. Do not grant a
        # fresh set of views/calls merely because the MCP process restarted.
        for event in self._events():
            if event.get('tool') in {t['name'] for t in TOOLS}:
                self.calls += 1
            if event.get('tool') in EDIT_TOOLS and event.get('status') == 'ok':
                view = event['result']
                path = Path(view['path'])
                if path.is_symlink() or path.resolve().parent != self.job or not path.is_file():
                    raise ValueError('saved image ledger points outside job or to missing image')
                # Do not persist recursive inspection histories inside views.
                self.views[view['image_id']] = {k: v for k, v in view.items()
                    if k not in {'inspection_history', 'reused', 'edit_budget', 'next_step'}}
                self.edit_cache[self._edit_key(event['tool'], event['arguments'])] = view['image_id']
        self.created = len(self.views) - image_count

    def _events(self):
        path = self.job / 'image_tool_calls.jsonl'
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A concurrent hook may still be appending its line.
        return events

    @staticmethod
    def _edit_key(name, args):
        return json.dumps([name, args], sort_keys=True)

    def inspection_history(self):
        reads = {}
        deliveries = {}
        for event in self._events():
            if event.get('kind') == 'read' and event.get('status') == 'completed':
                path = event.get('arguments', {}).get('file_path')
                if isinstance(path, str):
                    path = str((self.job / path).resolve())
                    if event.get('image_content', {}).get('identity_status') == 'VERIFIED':
                        reads.setdefault(path, set()).add(event.get('tool_use_id') or event.get('event_id'))
            if event.get('kind') == 'tool_lifecycle' and event.get('status') == 'completed':
                metadata = event.get('image_metadata', {})
                if image_matches(event.get('image_content'), metadata):
                    deliveries.setdefault(metadata.get('image_id'), set()).add(event.get('tool_use_id'))
        return [{k: view.get(k) for k in ('image_id', 'path', 'size', 'parent_image_id', 'operation', 'arguments')}
                | {'completed_reads': len(reads.get(view['path'], ())),
                   'validated_image_returns': len(deliveries.get(view['image_id'], ())) + len(reads.get(view['path'], ())),
                   'note': 'Validated tool-return content; not proof of model understanding.'}
                for view in self.views.values()]

    def edit_budget(self, *, consume=False):
        if self.edit_limit is None:
            return None
        # Persist before editing, including failed attempts. MCP restarts and
        # concurrent requests must not grant a fresh allowance in this job.
        with (self.job / 'image_edit_budget.json').open('a+', encoding='utf-8') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            content = stream.read()
            state = json.loads(content) if content else {'limit': self.edit_limit, 'used': 0}
            if state['limit'] != self.edit_limit:
                raise ValueError('image edit budget cannot change within a job')
            exhausted = state['used'] >= state['limit']
            if consume and not exhausted:
                state['used'] += 1
            stream.seek(0)
            stream.truncate()
            json.dump(state, stream)
            stream.flush()
        budget = {**state, 'remaining': state['limit'] - state['used'],
            'next_step': ('Editing is exhausted. Do not request more edits. Use view_image for existing images and '
                          'finish using the existing evidence, or report insufficient evidence under the requested schema; do not restart.'
                          if state['used'] >= state['limit'] else
                          'Finish as soon as the view is suitable; do not use edits just to spend the budget.')}
        if consume and exhausted:
            raise ValueError('image edit budget exhausted: ' + json.dumps(budget))
        return budget

    def call(self, name, args):
        started = time.monotonic()
        self.calls += 1
        event = {"tool": name, "arguments": args}
        try:
            cached = self.edit_cache.get(self._edit_key(name, args)) if name in EDIT_TOOLS else None
            if name in EDIT_TOOLS and cached is None:
                self.edit_budget(consume=True)
            if self.calls > MAX_CALLS:
                raise ValueError("image tool call budget exhausted")
            spec = next((t for t in TOOLS if t["name"] == name), None)
            if spec is None or not isinstance(args, dict) or set(args) != set(spec["inputSchema"]["required"]):
                raise ValueError("unknown tool or invalid argument fields")
            if name == 'list_images':
                result = {}
            else:
                image_id = args["image_id"]
                if not isinstance(image_id, str) or image_id not in self.views:
                    raise ValueError("unknown image_id; use image_N or a returned view ID")
                if cached is not None:
                    result = dict(self.views[cached], reused=True,
                                  next_step='Inspect the attached saved image; no extra Read needed.')
                else:
                    result = self._call(name, args, self.views[image_id])
                    if name in EDIT_TOOLS:
                        self.edit_cache[self._edit_key(name, args)] = result['image_id']
            result = dict(result, inspection_history=self.inspection_history())
            if self.edit_limit is not None:
                result = dict(result, edit_budget=self.edit_budget())
            event.update(status="ok", result=result)
            return result
        except Exception as exc:
            event.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if self.edit_limit is not None:
                event['edit_budget'] = self.edit_budget()
            event["duration_s"] = time.monotonic() - started
            audit(self.job, event)
            inventory = self.job / 'inspection_history.json'
            temporary = inventory.with_suffix('.tmp')
            temporary.write_text(json.dumps({'images': self.inspection_history(),
                'tool_calls': self.calls, 'edit_budget': self.edit_budget()}, indent=2), encoding='utf-8')
            temporary.replace(inventory)

    def _map_point(self, view, point):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("pixel_xy must contain two numbers")
        point = [_number(v) for v in point]
        # Check every ancestor too: a crop may have removed pixels that still
        # lie inside the original image, and rotation can add padding there.
        while True:
            w, h = view["size"]
            if not (-.5 <= point[0] < w-.5 and -.5 <= point[1] < h-.5):
                raise ValueError("point is outside visible source pixels (possibly rotation padding)")
            if view["parent_image_id"] is None:
                return {"original_image_index": view["original_image_index"],
                        "pixel_xy": [max(0, min(w-1, point[0])), max(0, min(h-1, point[1]))],
                        "coordinate_frame": "original_supplied_rgb_pixel_centers"}
            point = transform_point(view["to_parent"], point)
            view = self.views[view["parent_image_id"]]

    def _call(self, name, args, source):
        if name in {"image_info", "view_image"}:
            return dict(source)
        if name == "map_point":
            return self._map_point(source, args["pixel_xy"])
        if self.created >= MAX_VIEWS:
            raise ValueError("derived image budget exhausted")
        path = Path(source["path"])
        if path.is_symlink() or path.resolve().parent != self.job:
            raise ValueError("image escaped job directory")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        w, h = image.size
        if name == "crop_image":
            box = args["box"]
            if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError("box needs four integers")
            left, top, right, bottom = box
            if not (0 <= left < right <= w and 0 <= top < bottom <= h):
                raise ValueError("crop must be inside the source image")
            output = image.crop(tuple(box))
            mapping = [1., 0., float(left), 0., 1., float(top)]
        elif name == "resize_image":
            scale = _number(args["scale"])
            if not 0 < scale <= 8:
                raise ValueError("scale must be positive and at most 8")
            nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
            _size(nw, nh)
            output = image.resize((nw, nh), Image.Resampling.LANCZOS)
            mapping = [w/nw, 0., (w/nw-1)/2, 0., h/nh, (h/nh-1)/2]
        elif name == "rotate_image":
            degrees = _number(args["degrees_clockwise"])
            if not -360 <= degrees <= 360:
                raise ValueError("rotation must be within -360..360 degrees")
            radians = math.radians(degrees)
            c, s = round(math.cos(radians), 15), round(math.sin(radians), 15)
            nw, nh = math.ceil(abs(c)*w + abs(s)*h), math.ceil(abs(s)*w + abs(c)*h)
            _size(nw, nh)
            mapping = [c, s, (w-1)/2-c*(nw-1)/2-s*(nh-1)/2,
                       -s, c, (h-1)/2+s*(nw-1)/2-c*(nh-1)/2]
            # Pillow samples in edge coordinates, whereas metadata uses centers.
            pillow_mapping = mapping[:]
            pillow_mapping[2] += .5 - (mapping[0]+mapping[1])*.5
            pillow_mapping[5] += .5 - (mapping[3]+mapping[4])*.5
            output = image.transform((nw, nh), Image.Transform.AFFINE, pillow_mapping,
                                     Image.Resampling.BICUBIC, fillcolor=(96, 96, 96))
        else:
            raise ValueError("unknown transformation")
        image_id = "view_" + uuid.uuid4().hex[:16]
        destination = self.job / f"{image_id}.png"
        output.save(destination)
        view = dict(image_id=image_id, path=str(destination), size=list(output.size),
            operation=name, arguments=dict(args),
            rgb_sha256=pixel_hash(output), pillow_version=Image.__version__,
            original_image_index=source["original_image_index"], original_size=source["original_size"],
            to_original=compose(source["to_original"], mapping),
            parent_image_id=source["image_id"], to_parent=mapping)
        self.views[image_id] = view
        self.created += 1
        return dict(view, next_step="Inspect the attached image; coordinates are in this view. No extra Read needed.")

    def image_result(self, value):
        """Construct an actual MCP image block and validate its exact saved pixels."""
        path = Path(value['path'])
        if path.is_symlink() or path.resolve().parent != self.job:
            raise ValueError('image escaped job directory')
        block = {'type': 'image', 'mimeType': 'image/png',
                 'data': base64.b64encode(path.read_bytes()).decode('ascii')}
        summary = image_content_summary(block)
        if not image_matches(summary, value):
            raise ValueError('image payload is empty, invalid, or changed since it was saved')
        metadata = dict(value, image_content=summary)
        audit(self.job, {'kind': 'image_delivery', 'tool': 'image_payload', 'status': 'prepared',
                        'image_id': value['image_id'], 'path': str(path), 'image_content': summary})
        return {'content': [{'type': 'text', 'text': json.dumps(metadata)}, block]}


def serve_stdio(tools):
    for line in sys.stdin:
        message = None
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("request must be an object")
            if "id" not in message:
                continue
            method, params = message.get("method"), message.get("params") or {}
            if method in {"initialize", "tools/list"}:
                audit(tools.job, {"kind": "mcp_protocol", "tool": method, "status": "received"})
            if method == "initialize":
                requested = params.get("protocolVersion", "2024-11-05")
                result = {"protocolVersion": requested if requested in {
                    "2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"} else "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
                    "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                try:
                    value = tools.call(params.get("name"), params.get("arguments", {}))
                    result = (tools.image_result(value) if params.get('name') in IMAGE_TOOLS else
                              {"content": [{"type": "text", "text": json.dumps(value)}]})
                except Exception as exc:
                    detail = str(exc)
                    audit(tools.job, {'kind': 'image_delivery', 'tool': params.get('name'),
                                     'status': 'failed', 'error': detail})
                    if tools.edit_limit is not None:
                        detail += '\nedit_budget: ' + json.dumps(tools.edit_budget())
                    detail += '\ninspection_history: ' + json.dumps(tools.inspection_history())
                    result = {"isError": True, "content": [{"type": "text", "text": detail}]}
            else:
                print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                                  "error": {"code": -32601, "message": "method not found"}}), flush=True)
                continue
            response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": message.get("id") if isinstance(message, dict) else None,
                        "error": {"code": -32600, "message": str(exc)}}
        print(json.dumps(response, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--image-count", type=int, required=True)
    parser.add_argument('--edit-limit', type=int, default=None)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--read-hook", action="store_true")
    parser.add_argument("--audit-forward", action="store_true")
    parser.add_argument('--orientation-correction', action='store_true')
    parser.add_argument('--orientation-hook', action='store_true')
    args = parser.parse_args(argv)
    if args.read_hook:
        read_hook(args.job.resolve(strict=True))
        return 0
    if args.audit_forward:
        forward_audit(args.job.resolve(strict=True))
        return 0
    if args.orientation_hook:
        print(json.dumps(orientation_guard(args.job.resolve(strict=True), json.load(sys.stdin))), flush=True)
        return 0
    tools = ImageTools(args.job, args.image_count, edit_limit=args.edit_limit)
    if args.prepare:
        config = {"mcpServers": {SERVER_NAME: {"type": "stdio", "command": sys.executable,
            "args": [str(Path(__file__).resolve()), "--job", str(tools.job),
                     "--image-count", str(args.image_count)] +
                    (['--edit-limit', str(args.edit_limit)] if args.edit_limit is not None else [])}}}
        (tools.job / "image_tools.mcp.json").write_text(json.dumps(config), encoding="utf-8")
        (tools.job / "tool_list.json").write_text(json.dumps({"instructions": INSTRUCTIONS,
            "tools": TOOLS, "images": list(tools.views.values()),
            "edit_budget": tools.edit_budget()}, indent=2), encoding="utf-8")
        hook_command = shlex.join([sys.executable, str(Path(__file__).resolve()),
            "--job", str(tools.job), "--image-count", str(args.image_count), "--read-hook"])
        matcher = 'Read|mcp__cloth_image__.*'
        settings = {"hooks": {event: [{"matcher": matcher, "hooks": [
            {"type": "command", "command": hook_command, "timeout": 10}]}]
            for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure")}}
        if args.orientation_correction:
            guard_command = shlex.join([sys.executable, str(Path(__file__).resolve()),
                '--job', str(tools.job), '--image-count', str(args.image_count), '--orientation-hook'])
            guard = {'type': 'command', 'command': guard_command, 'timeout': 10}
            settings['hooks']['PreToolUse'].append({'matcher': 'StructuredOutput', 'hooks': [guard]})
            settings['hooks']['Stop'] = [{'hooks': [guard]}]
        (tools.job / "image_tools.settings.json").write_text(json.dumps(settings), encoding="utf-8")
        audit(tools.job, {"kind": "session", "tool": "image_tools_ready", "status": "ok",
            "images": list(tools.views.values()), "tools": TOOLS, "settings": settings,
            "edit_budget": tools.edit_budget(),
            "pillow_version": Image.__version__})
    else:
        serve_stdio(tools)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
