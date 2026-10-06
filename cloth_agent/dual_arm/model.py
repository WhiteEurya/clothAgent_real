"""Claude plans two RGB grasp pixels and independently evaluates paired holds."""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
from jsonschema import validate
from PIL import Image, ImageDraw

from ..planner_backend import LocalClaudeBackend, RemoteClaudeBackend, parse_claude_json
from .geometry import DualArmError
from .planning import read_targets

GRASP = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pixel_xy": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "integer", "minimum": 0},
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["pixel_xy", "reason"],
}
PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": 2},
        "mode": {"enum": ["pin_pull", "center_pair"]},
        "pin_arm": {"enum": [None, "left", "right"]},
        "center": {"anyOf": [GRASP, {"type": "null"}]},
        "observation_id": {"type": "string"},
        "grasps": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"left": GRASP, "right": GRASP},
            "required": ["left", "right"],
        },
        "lift_mm": {"type": "number"},
        "spread_mm": {"type": "number"},
        "approach_mm": {"type": "number"},
    },
    "required": [
        "schema_version",
        "mode",
        "pin_arm",
        "center",
        "observation_id",
        "grasps",
        "lift_mm",
        "spread_mm",
        "approach_mm",
    ],
}
ABSTAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"const": "abstain"},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["decision", "reason"],
}
CENTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"observation_id": {"type": "string"}, "center": GRASP},
    "required": ["observation_id", "center"],
}
CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "left_holding": {"type": "boolean"},
        "right_holding": {"type": "boolean"},
        "slip": {"type": "boolean"},
        "overstretched": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [
        "left_holding",
        "right_holding",
        "slip",
        "overstretched",
        "confidence",
        "reason",
    ],
}
PIN_CHECK_SCHEMA = copy.deepcopy(CHECK_SCHEMA)
PIN_CHECK_SCHEMA["properties"].update(
    pin_contact={"type": "boolean"},
    pin_slip={"type": "boolean"},
)
PIN_CHECK_SCHEMA["required"] += ["pin_contact", "pin_slip"]


def reject_abstention(result):
    if result.get("decision") == "abstain":
        raise DualArmError(
            f"Claude declined this observation: {result.get('reason', '')}"
        )


class VisionPlanner:
    def __init__(self, backend="remote", *, host="company-planner", timeout_s=60):
        self.backend, self.host, self.timeout_s = backend, host, timeout_s

    def invoke(self, prompt, images, schema, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        system = "Use only the supplied RGB evidence. Return exactly the required JSON. Do not command robots. If evidence is uncertain, report uncertainty."
        request = {
            "prompt": prompt,
            "images": [str(p) for p in images],
            "schema": schema,
            "system": system,
        }
        (directory / "request.json").write_text(json.dumps(request, indent=2))
        started = time.monotonic()
        try:
            if self.backend == "remote":
                client = RemoteClaudeBackend(
                    ssh_host=self.host, timeout_s=self.timeout_s, image_tools=False
                )
                result = client.invoke(
                    prompt=prompt,
                    image_paths=images,
                    schema=schema,
                    system_prompt=system,
                    overall_timeout_s=self.timeout_s,
                )
            elif self.backend == "local":
                binary = shutil.which("claude")
                if not binary:
                    raise DualArmError("claude executable is unavailable")
                # Stage only the requested evidence in this call's workspace.
                for i, path in enumerate(images):
                    shutil.copyfile(path, directory / f"image_{i}.png")
                command = [
                    binary,
                    "-p",
                    prompt
                    + "\nRead "
                    + ", ".join(f"image_{i}.png" for i in range(len(images))),
                    "--output-format",
                    "json",
                    "--json-schema",
                    json.dumps(schema),
                    "--tools",
                    "Read",
                    "--allowedTools",
                    "Read",
                    "--permission-mode",
                    "dontAsk",
                    "--no-session-persistence",
                    "--system-prompt",
                    system,
                ]
                result = LocalClaudeBackend(timeout_s=self.timeout_s).invoke(
                    prompt=prompt, command=command, cwd=directory
                )
            else:
                raise DualArmError("unknown vision backend")
            (directory / "stdout.txt").write_text(result.stdout)
            (directory / "stderr.txt").write_text(result.stderr)
            payload = parse_claude_json(result.stdout)
            validate(payload, schema)
            (directory / "response.json").write_text(json.dumps(payload, indent=2))
            return payload
        finally:
            (directory / "timing.json").write_text(
                json.dumps({"duration_s": time.monotonic() - started})
            )

    def plan(self, observation, config, directory, *, mode="center_pair"):
        if mode not in {"pin_pull", "center_pair"}:
            raise DualArmError("unknown guidance mode")
        directory = Path(directory)
        with Image.open(observation.image) as image:
            size = image.size
        l = config.limits
        schema = copy.deepcopy(PLAN_SCHEMA)
        schema["properties"]["mode"] = {"const": mode}
        images = [observation.image]
        center = None
        if mode == "center_pair":
            first = self.invoke(
                "First select ONE center point of the visible cloth region to unfold. "
                "Do not select grasp points yet. It must be on this garment, never the table or a robot. "
                f"ORIGINAL image pixels, size={size}, observation_id={observation.meta['observation_id']}. "
                "Return decision=abstain and a reason if no reliable center is visible.",
                images,
                {"anyOf": [CENTER_SCHEMA, ABSTAIN_SCHEMA]},
                directory / "center",
            )
            reject_abstention(first)
            validate(first, CENTER_SCHEMA)
            if first["observation_id"] != observation.meta["observation_id"]:
                raise DualArmError("center belongs to a different observation")
            center = first["center"]
            center_xyz = observation.sample(
                center["pixel_xy"], l["max_surface_spread_mm"]
            )
            directory.mkdir(parents=True, exist_ok=True)
            with Image.open(observation.image) as source:
                rgb = np.asarray(source.convert("RGB")).copy()
            neighborhood = (
                np.linalg.norm(observation.xyz - center_xyz, axis=2)
                <= l["center_radius_mm"]
            )
            # Show the metric neighborhood without resizing or changing pixels.
            rgb[~neighborhood] = (rgb[~neighborhood] * 0.25).astype(np.uint8)
            annotated = Image.fromarray(rgb)
            draw = ImageDraw.Draw(annotated)
            x, y = center["pixel_xy"]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="red", width=2)
            draw.text((x + 8, y), "CENTER", fill="red")
            marked = directory / "fixed_center.png"
            annotated.save(marked)
            images = [observation.image, marked]
            schema["properties"]["center"] = {"const": center}
            schema["properties"]["pin_arm"] = {"type": "null"}
            instruction = (
                f"The center is FIXED: {json.dumps(center)}. Do not change it. "
                f"Choose two grasp points on opposite sides of this center, each within {l['center_radius_mm']} mm in measured 3D. "
                "The host checks this metric neighborhood, separation, reachability and swept collision envelopes. "
                "Both arms will grasp, trial-lift, lift and spread simultaneously. "
                "Image 1 marks the same center on an unresized copy of image 0; "
                "the DARKENED area is outside the measured allowed neighborhood. Pick both points in the bright area. "
            )
        else:
            schema["properties"]["center"] = {"type": "null"}
            schema["properties"]["pin_arm"] = {"enum": ["left", "right"]}
            instruction = (
                "Choose which physical arm will PIN the cloth against its support with closed fingers. "
                "Its pixel is an interior supported pin point, not a grasp. The other arm selects a visible graspable edge. "
                "The pin stays stationary while the other gripper grasps, trial-lifts and pulls horizontally AWAY from the pin. "
                "Do not cross arms or pull toward the pin. spread_mm is the moving arm's total outward travel. "
            )
        prompt = (
            instruction + "Plan two contact points on ONE garment, "
            "one for each physical gripper. Coordinates are pixels in the ORIGINAL supplied image, top-left origin; "
            "do not rotate or resize coordinates. Never select table/background or occluded, ungraspable points. "
            "Return decision=abstain and a reason if a safe pair is not visible; do not invent one. "
            f"Image size {size}; observation_id={observation.meta['observation_id']}. "
            f"Choose lift_mm in [10,{l['max_lift_mm']}], spread_mm (total increase in separation) in [0,{l['max_spread_mm']}], "
            f"approach_mm in [10,{l['max_approach_mm']}]. "
            "Left/right name physical arms, not sleeves. Physical layout description: "
            + config.raw["arm_layout_description"]
        )
        result = self.invoke(
            prompt, images, {"anyOf": [schema, ABSTAIN_SCHEMA]}, directory / "points"
        )
        reject_abstention(result)
        if result["center"] != center:
            raise DualArmError("second stage changed the fixed center")
        validate(result, schema)
        return read_targets(result, config)

    def check(self, original, current, phase, config, directory):
        name = phase.name if hasattr(phase, "name") else phase
        pin = getattr(phase, "pin_arm", None)
        context = (
            (
                f"Physical arm {pin} is pinning, not grasping. Confirm visible cloth contact and no sliding under the pin "
                "in pin_contact/pin_slip. Only the other arm must hold cloth after its grasp. "
                "Visual contact cannot certify force or pressure. "
            )
            if pin
            else ""
        )
        prompt = (
            context
            + "Image 0 is the BEFORE-GRASP reference only. Image 1 is the CURRENT stationary view. "
            f"Current phase: {name}. Determine separately whether each physical gripper visibly holds the garment. "
            "Check empty grasp, slip and excessive stretch. Closed fingers alone do not prove a grasp. "
            "If occluded or ambiguous, return low confidence and do not infer success from the plan. "
            "Physical arm layout: " + config.raw["arm_layout_description"]
        )
        return self.invoke(
            prompt,
            [original.image, current.image],
            PIN_CHECK_SCHEMA if pin else CHECK_SCHEMA,
            directory,
        )


def require_hold(result, phase=None):
    pin = getattr(phase, "pin_arm", None)
    validate(result, PIN_CHECK_SCHEMA if pin else CHECK_SCHEMA)
    if pin:
        moving = "right" if pin == "left" else "left"
        held = (
            getattr(phase, "checkpoint", None) == "pin_only"
            or result[f"{moving}_holding"]
        )
        held = held and result["pin_contact"] and not result["pin_slip"]
    else:
        held = result["left_holding"] and result["right_holding"]
    if (
        not held
        or result["slip"]
        or result["overstretched"]
        or result["confidence"] < 0.8
    ):
        raise DualArmError(f"paired hold not confirmed: {result['reason']}")
