#!/usr/bin/env python3
"""Quick smoke test for the remote Codex planner backend.

Live usage (from the Alienware machine)::

    python scripts/remote_planner_test.py path/to/camera_A.png

The test only sends an RGB PNG and asks Codex for a small JSON observation. It
never imports or starts the camera, grounding, or robot controller. ``--mock``
checks the local command and parser wiring without network access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloth_agent.planner_backend import RemoteCodexBackend, parse_claude_json  # noqa: E402


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "image_read": {"type": "boolean"},
        "observation": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["ok", "image_read", "observation", "confidence"],
}


def _check_payload(stdout: str) -> dict:
    # Keep this smoke test usable on a minimal checkout without importing the
    # perception stack (and its NumPy/OpenCV dependencies).
    payload = parse_claude_json(stdout)
    missing = [key for key in SCHEMA["required"] if key not in payload]
    if missing:
        raise RuntimeError(f"Codex JSON is missing fields: {', '.join(missing)}")
    if payload["ok"] is not True or payload["image_read"] is not True:
        raise RuntimeError(f"Codex did not confirm image reading: {payload}")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise RuntimeError(f"invalid confidence: {confidence!r}")
    if set(payload) != set(SCHEMA["required"]) or not isinstance(payload["observation"], str) or not payload["observation"].strip():
        raise RuntimeError("invalid smoke observation schema")
    return payload


def run(image: Path, *, mock: bool, host: str, timeout_s: int) -> dict:
    if image.suffix.lower() != ".png":
        raise ValueError("the smoke test accepts a PNG image")
    if not image.is_file():
        raise FileNotFoundError(image)
    prompt = (
        "Inspect image_0 with the cloth_image view_image tool. Return exactly one JSON object "
        "with fields ok (true), image_read (true), observation (brief visual description), "
        "and confidence (number from 0 to 1). Do not execute commands or control a robot."
    )
    backend = RemoteCodexBackend(ssh_host=host, timeout_s=timeout_s)
    if mock:
        fake_upload = type("Result", (), {"returncode": 0, "stdout": '{"id":"smoke"}', "stderr": ""})()
        fake_ssh = type(
            "Result", (), {
                "returncode": 0,
                "stdout": json.dumps({"result": json.dumps({
                    "ok": True, "image_read": True,
                    "observation": "mock image read", "confidence": 1.0,
                })}),
                "stderr": "",
            }
        )()
        with patch("cloth_agent.planner_backend.subprocess.run", side_effect=[fake_upload, fake_ssh, fake_ssh]):
            result = backend.invoke(
                prompt=prompt, image_paths=[image], schema=SCHEMA,
                system_prompt="Return only the requested JSON.",
            )
    else:
        result = backend.invoke(
            prompt=prompt, image_paths=[image], schema=SCHEMA,
            system_prompt="You are a read-only image analyst. Return only JSON.",
        )
    return _check_payload(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="existing RGB PNG to send")
    parser.add_argument("--host", default="company-planner", help="SSH config host")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--mock", action="store_true", help="test wiring without network")
    args = parser.parse_args()
    try:
        print("[1/4] validating local RGB PNG", flush=True)
        print("[2/4] uploading and relaying image", flush=True)
        payload = run(args.image.resolve(), mock=args.mock, host=args.host, timeout_s=args.timeout_s)
        print("[3/4] Codex returned valid JSON", flush=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("[4/4] REMOTE PLANNER SMOKE TEST PASSED", flush=True)
        return 0
    except Exception as exc:
        print(f"REMOTE PLANNER SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
