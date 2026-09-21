#!/usr/bin/env python3
"""Verify GPT-6 Responses RGB tools without a camera, Molmo, or robot connection."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
from cloth_agent.image_tools_mcp import ImageTools
from cloth_agent.planner_backend import RemoteCodexBackend, parse_claude_json


SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "image_read": {"type": "boolean"}, "observation": {"type": "string", "minLength": 1},
    "original_image_index": {"type": "integer", "minimum": 0},
    "mapped_pixel_xy": {"type": "array", "minItems": 2, "maxItems": 2,
                        "items": {"type": "number"}}},
    "required": ["image_read", "observation", "original_image_index", "mapped_pixel_xy"]}


class CompanyLocalBackend(RemoteCodexBackend):
    """Exercise the production company shell locally; only transport is bypassed."""

    def _upload(self, image):
        return image.resolve().as_uri()


def synthetic_image(path):
    image = Image.new("RGB", (512, 384), "white")
    draw = ImageDraw.Draw(image)
    for box, color, label in (
        ((0, 0, 255, 191), "red", "TOP LEFT: RED"),
        ((256, 0, 511, 191), "green", "TOP RIGHT: GREEN"),
        ((0, 192, 255, 383), "blue", "BOTTOM LEFT: BLUE"),
        ((256, 192, 511, 383), "orange", "BOTTOM RIGHT: ORANGE"),
    ):
        draw.rectangle(box, fill=color)
        draw.text((box[0]+20, box[1]+20), label, fill="white", stroke_width=1, stroke_fill="black")
    draw.ellipse((230, 166, 282, 218), fill="white", outline="black", width=3)
    image.save(path)


def replay(image, events, output):
    """Recreate inspection views locally from the audit, without SCP of PNGs."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(image, output / "image_0.png")
    tools = ImageTools(output, 1)
    ids = {"image_0": "image_0"}
    views = []
    for event in events:
        if event.get("status") != "ok" or event.get("tool") not in {
                "rotate_image", "crop_image", "resize_image", "map_point", "image_info"}:
            continue
        args = dict(event["arguments"])
        args["image_id"] = ids[args["image_id"]]
        result = tools.call(event["tool"], args)
        if event["tool"] in {"rotate_image", "crop_image", "resize_image"}:
            remote_id = event["result"]["image_id"]
            ids[remote_id] = result["image_id"]
            views.append({"tool": event["tool"], "remote_image_id": remote_id,
                          "local_replayed_path": result["path"]})
    return views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="RGB PNG; omitted: generate a synthetic color chart")
    parser.add_argument("--host", default="company-planner")
    parser.add_argument("--timeout-s", type=int, default=300)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="test tools locally, without a model or network")
    mode.add_argument("--local-responses", "--local-codex", dest="local_codex", action="store_true",
                      help="real Responses API from this computer; no HTTPS relay or SSH (old alias retained)")
    parser.add_argument("--output-dir", type=Path, help="new output directory")
    args = parser.parse_args(argv)
    output = args.output_dir or Path("results/image_tools_smoke") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    helper_directory = None
    try:
        output.mkdir(parents=True, exist_ok=False)
        if args.image is None:
            image = (output / "synthetic_rgb.png").resolve()
            synthetic_image(image)
        else:
            image = args.image.resolve(strict=True)
        with Image.open(image) as original:
            if original.format != "PNG" or min(original.size) < 4:
                raise ValueError("use a PNG at least 4 by 4 pixels")
            rw, rh = original.height, original.width  # clockwise 90
        box = [rw//4, rh//4, rw-rw//4, rh-rh//4]
        if args.offline:
            job = output / "local_tools"
            job.mkdir()
            shutil.copyfile(image, job / "image_0.png")
            tools = ImageTools(job, 1)
            rotated = tools.call("rotate_image", {"image_id": "image_0", "degrees_clockwise": 90})
            crop = tools.call("crop_image", {"image_id": rotated["image_id"], "box": box})
            zoom = tools.call("resize_image", {"image_id": crop["image_id"], "scale": 2})
            mapped = tools.call("map_point", {"image_id": zoom["image_id"], "pixel_xy": [1, 1]})
            events = [json.loads(line) for line in (job / "image_tool_calls.jsonl").read_text().splitlines()]
            payload = {"mode": "offline", "mapped_point": mapped, "codex_tested": False}
        else:
            prompt = (
                "This is a tool integration smoke test, not a robot plan. Use view_image on image_0 first. "
                "You MUST call rotate_image on image_0 with degrees_clockwise=90, then inspect its attached image. "
                f"Call crop_image on that rotated view with box={box}, then resize_image on the crop "
                "with scale=2, and inspect the attached enlarged result. No extra Read needed. Call map_point on this final resized "
                "view at pixel_xy=[1,1]. Return JSON with image_read=true only if you actually saw "
                "the images, observation describing what you see, and original_image_index and "
                "mapped_pixel_xy copied from the map_point result. If tools fail, do not claim success."
            )
            if args.local_codex:
                helper_directory = tempfile.TemporaryDirectory(prefix="cloth_company_smoke_")
                launcher = Path(helper_directory.name) / "local_company_shell"
                launcher.write_text('#!/bin/sh\nfor cloth_arg do :; done\nexec /bin/sh -c "$cloth_arg"\n', encoding="utf-8")
                launcher.chmod(0o700)
                backend = CompanyLocalBackend(ssh_host="local-company", ssh_binary=str(launcher), timeout_s=args.timeout_s)
            else:
                backend = RemoteCodexBackend(ssh_host=args.host, timeout_s=args.timeout_s)
            result = backend.invoke(prompt=prompt, image_paths=[image], schema=SCHEMA,
                debug_dir=output / "claude_image_tools" / "smoke",
                system_prompt="Inspect RGB with view_image and images returned by editing tools. Return the required JSON.")
            events = list(result.image_tool_events)
            (output / "image_tool_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
            payload = parse_claude_json(result.stdout)
            success = [e for e in events if e.get("status") == "ok"]
            required = {"rotate_image", "crop_image", "resize_image", "map_point"}
            if not required <= {e.get("tool") for e in success}:
                raise ValueError("Codex did not actually call all required image tools; inspect the audit")
            inspected_views = [e["result"]["image_id"] for e in success
                               if e.get("tool") in {"rotate_image", "resize_image"}]
            delivered = {view['image_id'] for view in result.image_sources
                         if view.get('image_delivery_status') in {'VERIFIED', 'VERIFIED_TRANSCODE'}}
            if not {'image_0', *inspected_views} <= delivered:
                raise ValueError("API submission audit lacks verified original/rotated/enlarged images; inspect image_delivery.jsonl")
            if payload.get("image_read") is not True or not isinstance(payload.get("observation"), str) or not payload["observation"].strip():
                raise ValueError("Codex did not confirm visual inspection")
            mappings = [e["result"] for e in success if e["tool"] == "map_point"]
            if not any(payload.get("original_image_index") == m["original_image_index"] and
                       payload.get("mapped_pixel_xy") == m["pixel_xy"] for m in mappings):
                raise ValueError("returned pixel does not match an actual map_point result")
            payload["locally_replayed_views"] = replay(image, events, output / "replayed_views")
            payload["timings"] = result.timings
            payload["transport"] = "local_company_shell_no_relay_or_ssh" if args.local_codex else "https_and_ssh"
        (output / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        label = "OFFLINE IMAGE TOOLS" if args.offline else "LOCAL RESPONSES IMAGE TOOLS" if args.local_codex else "REMOTE RESPONSES IMAGE TOOLS"
        print(f"{label} PASSED: {output.resolve()}")
        return 0
    except Exception as exc:
        if output.is_dir() and 'backend' in locals():
            (output / "image_tool_events.json").write_text(json.dumps(backend.last_image_tool_events, indent=2), encoding="utf-8")
        print(f"IMAGE TOOLS TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        debug = output / "claude_image_tools" / "smoke"
        if debug.is_dir():
            print(f"Full CLI stdout: {debug.resolve() / 'stdout.log'}", file=sys.stderr)
            print(f"Full CLI stderr: {debug.resolve() / 'stderr.log'}", file=sys.stderr)
        return 1
    finally:
        if helper_directory is not None:
            helper_directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
