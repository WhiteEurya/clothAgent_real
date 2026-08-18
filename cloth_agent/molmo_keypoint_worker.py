"""GPU worker for one-point Molmo keypoints with model-token confidence.

The model is loaded once.  Each named keypoint is queried independently on
each image so that every record has exactly one semantic target and one
confidence score.  Confidence is the geometric mean of Molmo's probabilities
for the three generated dynamic point-location tokens.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def geometric_mean_probability(values: Sequence[float]) -> float:
    """Return a stable geometric mean for probabilities in ``[0, 1]``."""

    probabilities = [float(value) for value in values]
    if not probabilities:
        return 0.0
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if any(value == 0.0 for value in probabilities):
        return 0.0
    return float(
        math.exp(sum(math.log(value) for value in probabilities) / len(probabilities))
    )


def point_location_probabilities(
    dynamic_token_probabilities: Sequence[float], returned_point_count: int
) -> list[float]:
    """Select the three location tokens for one point, excluding stop tokens.

    MolmoPoint emits three dynamic tokens (patch, subpatch, 3x3 location) per
    point and then a dynamic ``no more points`` token.  The latter terminates
    pointing and must not be counted as coordinate confidence.
    """

    values = [float(value) for value in dynamic_token_probabilities]
    if returned_point_count != 1 or len(values) < 3:
        return []
    return values[:3]


def _load_specs(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("--specs must contain a non-empty JSON list")
    specs: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("every keypoint spec must be an object")
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or name in names or not description:
            raise RuntimeError("keypoint names/descriptions must be non-empty and unique")
        names.add(name)
        specs.append({"name": name, "description": description})
    return specs


def _prompt(description: str) -> str:
    return (
        f"Point to the garment's {description}. Return exactly one point only when "
        "that keypoint is clearly identifiable on visible garment fabric. If it is "
        "occluded, ambiguous, outside the image, or not confidently identifiable, "
        "return no point. Do not point to the table, robot, gripper, or another object."
    )


def _selected_token_probabilities(
    generated_ids: Any,
    generation_scores: Sequence[Any],
) -> list[float]:
    """Return selected-token probabilities for one greedy generation."""

    import torch

    ids = generated_ids[0].tolist()
    probabilities: list[float] = []
    for token_id, scores in zip(ids, generation_scores):
        probability = torch.softmax(scores[0].float(), dim=-1)[int(token_id)]
        probabilities.append(float(probability.detach().cpu().item()))
    return probabilities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.image) != len(args.label) or not args.image:
        raise SystemExit("--image and --label counts must match and be non-empty")
    if any(not path.is_file() for path in args.image):
        raise SystemExit("every --image must exist")
    if not args.specs.is_file():
        raise SystemExit("--specs must be an existing JSON file")
    if len(set(args.label)) != len(args.label):
        raise SystemExit("--label values must be unique")
    specs = _load_specs(args.specs)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("MolmoPoint requires a CUDA GPU; CUDA is unavailable")
    if args.dtype == "bf16" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif args.dtype in {"bf16", "fp16"}:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=model_dtype,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = sorted(
        {
            str(device)
            for device in device_map.values()
            if str(device) in {"cpu", "disk", "meta"}
        }
    )
    if offloaded:
        raise RuntimeError(f"MolmoPoint model/vision tower was offloaded to {offloaded}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=args.local_files_only,
    )
    amp_dtype = (
        torch.bfloat16
        if args.dtype == "bf16" and torch.cuda.is_bf16_supported()
        else torch.float16
        if args.dtype != "fp32"
        else torch.float32
    )
    # Dynamic point tokens start after the normal and additional text vocab.
    text_config = model.config.text_config
    point_token_start = int(text_config.vocab_size + text_config.additional_vocab_size)

    views: list[dict[str, Any]] = []
    for image_path, label in zip(args.image, args.label):
        image = Image.open(image_path).convert("RGB")
        records: list[dict[str, Any]] = []
        for spec in specs:
            prompt = _prompt(spec["description"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                return_pointing_metadata=True,
                images_kwargs={"max_crops": args.max_crops},
            )
            metadata = inputs.pop("metadata")
            inputs = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            autocast_context = (
                torch.autocast("cuda", dtype=amp_dtype)
                if amp_dtype != torch.float32
                else nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                generation = model.generate(
                    **inputs,
                    logits_processor=model.build_logit_processor_from_inputs(inputs),
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            generated_ids = generation.sequences[:, inputs["input_ids"].size(1) :]
            generated_text = processor.post_process_image_text_to_text(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            raw = model.extract_image_points(
                generated_text,
                metadata["token_pooling"],
                metadata["subpatch_mapping"],
                metadata["image_sizes"],
            )
            raw = raw.tolist() if hasattr(raw, "tolist") else raw
            selected_probabilities = _selected_token_probabilities(
                generated_ids, generation.scores
            )
            token_ids = generated_ids[0].tolist()
            dynamic_point_probabilities = [
                probability
                for token_id, probability in zip(token_ids, selected_probabilities)
                if int(token_id) >= point_token_start
            ]
            points: list[list[float]] = []
            if isinstance(raw, list):
                for point in raw:
                    if not isinstance(point, list) or len(point) < 2:
                        continue
                    x_px, y_px = float(point[-2]), float(point[-1])
                    if (
                        math.isfinite(x_px)
                        and math.isfinite(y_px)
                        and 0 <= x_px < image.width
                        and 0 <= y_px < image.height
                    ):
                        points.append([x_px, y_px])
            point_probabilities = point_location_probabilities(
                dynamic_point_probabilities, len(points)
            )
            termination_probability = (
                float(dynamic_point_probabilities[3])
                if len(points) == 1 and len(dynamic_point_probabilities) > 3
                else None
            )
            if len(points) == 1 and len(point_probabilities) == 3:
                status = "point_returned"
                pixel_xy: list[float] | None = points[0]
                confidence = geometric_mean_probability(point_probabilities)
            elif len(points) == 0:
                status = "not_found"
                pixel_xy = None
                confidence = 0.0
            else:
                status = "ambiguous"
                pixel_xy = None
                confidence = 0.0
            records.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "prompt": prompt,
                    "status": status,
                    "pixel_xy": pixel_xy,
                    "confidence": confidence,
                    "confidence_definition": (
                        "geometric_mean_probability_of_the_three_generated_molmo_point_tokens"
                    ),
                    "point_token_probabilities": point_probabilities,
                    "termination_point_token_probability": termination_probability,
                    "raw_valid_points": points,
                    "raw_point_count": len(points),
                    "generated_text": generated_text,
                }
            )
            del generation, inputs, metadata
            torch.cuda.empty_cache()
        views.append(
            {
                "label": str(label).upper(),
                "image": str(image_path),
                "image_size": [image.width, image.height],
                "records": records,
            }
        )

    payload = {
        "schema_version": 1,
        "model": args.model,
        "query_mode": "one_independent_point_per_keypoint_with_token_confidence",
        "confidence_definition": (
            "geometric_mean_probability_of_the_three_generated_molmo_point_tokens"
        ),
        "confidence_is_calibrated_probability": False,
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
