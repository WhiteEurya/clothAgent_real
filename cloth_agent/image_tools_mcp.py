"""Job-local RGB inspection tools, deployable standalone with Python + Pillow.

Pixel coordinates denote pixel centers; every derived view stores its exact
affine map back to the originally supplied RGB. No camera or robot imports.
"""
from __future__ import annotations

import argparse
import hashlib
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
IDENTITY = [1., 0., 0., 0., 1., 0.]
INSTRUCTIONS = (
    "Use cloth_image tools to inspect the supplied RGB: rotate_image (positive "
    "degrees clockwise), crop_image, resize_image, image_info, map_point. "
    "Choose operations yourself when useful, then Read the returned path to SEE "
    "the result. Tool text alone is not a visual observation. Originals are "
    "image_0, image_1, etc., matching the supplied image manifest. Derived views "
    "are for inspection only. map_point can recover original image_index and pixel_xy. "
    "When the response schema includes image_id, return the exact source image/view ID "
    "and coordinates in THAT view; the host performs the mapping. Never pair a derived "
    "view ID with coordinates already mapped to the original. Otherwise map_point "
    "before returning original-image pixels. "
    "Final executable pixels must belong to the current Camera-A original, never "
    "a static reference or a crop/rotated view. Rxxx IDs keep their original "
    "identity. An image rotation never changes the task's garment-left/right "
    "definition. No mirroring, generated fabric, depth, or robot access is provided."
)


def _tool(name, description, properties):
    return {"name": name, "description": description, "inputSchema": {
        "type": "object", "additionalProperties": False,
        "properties": {"image_id": {"type": "string"}, **properties},
        "required": ["image_id", *properties],
    }}


TOOLS = [
    _tool("image_info", "Get image dimensions, original identity and Read path.", {}),
    _tool("rotate_image", "Rotate RGB by degrees clockwise, expanding the canvas; Read the returned path.",
          {"degrees_clockwise": {"type": "number", "minimum": -360, "maximum": 360}}),
    _tool("crop_image", "Crop [left, top, right, bottom], right/bottom exclusive; Read the returned path.",
          {"box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}}}),
    _tool("resize_image", "Enlarge/reduce while preserving aspect ratio. Scale > 1 zooms in; adds no detail.",
          {"scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 8}}),
    _tool("map_point", "Map a derived-view point to its original RGB. Rejects rotation padding/out-of-image points.",
          {"pixel_xy": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}}),
]
TOOL_NAMES = tuple("mcp__" + SERVER_NAME + "__" + tool["name"] for tool in TOOLS)


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()


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
    if payload.get("tool_name") != "Read":
        return
    event = payload.get("hook_event_name")
    status = {"PreToolUse": "started", "PostToolUse": "completed",
              "PostToolUseFailure": "failed"}.get(event, "unknown")
    args = payload.get("tool_input") or {}
    audit(job, {"kind": "read", "tool": "Read", "status": status,
                "tool_use_id": payload.get("tool_use_id"),
                "arguments": {"file_path": args.get("file_path")},
                "error": str(payload.get("error", "")) if status == "failed" else None})


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
    def __init__(self, job: Path, image_count: int):
        self.job = job.resolve(strict=True)
        self.views = {}
        self.calls = 0
        self.created = 0
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

    def call(self, name, args):
        started = time.monotonic()
        self.calls += 1
        event = {"tool": name, "arguments": args}
        try:
            if self.calls > MAX_CALLS:
                raise ValueError("image tool call budget exhausted")
            spec = next((t for t in TOOLS if t["name"] == name), None)
            if spec is None or not isinstance(args, dict) or set(args) != set(spec["inputSchema"]["required"]):
                raise ValueError("unknown tool or invalid argument fields")
            image_id = args["image_id"]
            if not isinstance(image_id, str) or image_id not in self.views:
                raise ValueError("unknown image_id; use image_N or a returned view ID")
            result = self._call(name, args, self.views[image_id])
            event.update(status="ok", result=result)
            return result
        except Exception as exc:
            event.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event["duration_s"] = time.monotonic() - started
            audit(self.job, event)

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
        if name == "image_info":
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
            rgb_sha256=pixel_hash(output), pillow_version=Image.__version__,
            original_image_index=source["original_image_index"], original_size=source["original_size"],
            to_original=compose(source["to_original"], mapping),
            parent_image_id=source["image_id"], to_parent=mapping)
        self.views[image_id] = view
        self.created += 1
        return dict(view, next_step="Read this path to inspect the transformed image; map_point before using a pixel.")


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
                    result = {"content": [{"type": "text", "text": json.dumps(value)}]}
                except Exception as exc:
                    result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
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
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--read-hook", action="store_true")
    parser.add_argument("--audit-forward", action="store_true")
    args = parser.parse_args(argv)
    if args.read_hook:
        read_hook(args.job.resolve(strict=True))
        return 0
    if args.audit_forward:
        forward_audit(args.job.resolve(strict=True))
        return 0
    tools = ImageTools(args.job, args.image_count)
    if args.prepare:
        config = {"mcpServers": {SERVER_NAME: {"type": "stdio", "command": sys.executable,
            "args": [str(Path(__file__).resolve()), "--job", str(tools.job),
                     "--image-count", str(args.image_count)]}}}
        (tools.job / "image_tools.mcp.json").write_text(json.dumps(config), encoding="utf-8")
        (tools.job / "tool_list.json").write_text(json.dumps({"instructions": INSTRUCTIONS,
            "tools": TOOLS, "images": list(tools.views.values())}, indent=2), encoding="utf-8")
        hook_command = shlex.join([sys.executable, str(Path(__file__).resolve()),
            "--job", str(tools.job), "--image-count", str(args.image_count), "--read-hook"])
        settings = {"hooks": {event: [{"matcher": "Read", "hooks": [
            {"type": "command", "command": hook_command, "timeout": 10}]}]
            for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure")}}
        (tools.job / "image_tools.settings.json").write_text(json.dumps(settings), encoding="utf-8")
        audit(tools.job, {"kind": "session", "tool": "image_tools_ready", "status": "ok",
            "images": list(tools.views.values()), "tools": TOOLS, "settings": settings,
            "pillow_version": Image.__version__})
    else:
        serve_stdio(tools)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
