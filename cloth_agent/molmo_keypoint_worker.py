"""GPU worker for one-point Molmo keypoints with model-token confidence.

The model is loaded once. Collar localization is deliberately independent from
the weak garment-axis hypothesis: a sewn neck-label topology guide is queried
first, then the collar is queried with that local relation. Remaining keypoints
may still be inferred concurrently.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


AXIS_SPECS = (
    {
        "name": "axis_top",
        "description": "the center of the garment's collar or neck opening; the top point of its own collar-to-hem centerline",
    },
    {
        "name": "axis_bottom",
        "description": "the center of the garment's lowest visible bottom hem; the bottom point of its own collar-to-hem centerline",
    },
)

# Axis endpoints are weak layout hypotheses on folded garments. In particular,
# neither collar nor neckline may alias axis_top: doing so turns one wrong
# centerline endpoint into a high-confidence false collar observation.
AXIS_RECORD_ALIASES: dict[str, str] = {}
NECK_LABEL_NAMES = frozenset({"neck_label", "neck_tag"})
COLLAR_NAMES = frozenset({"collar", "neckline"})


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
        if name in {spec["name"] for spec in AXIS_SPECS}:
            raise RuntimeError("keypoint specs may not redefine axis_top or axis_bottom")
        names.add(name)
        specs.append({"name": name, "description": description})
    return specs


def _prompt(
    description: str,
    context: str = "",
    *,
    allow_clothing_label: bool = False,
) -> str:
    context_prefix = f"{context.strip()}\n" if context.strip() else ""
    label_rule = (
        "This is the dedicated neck-label topology-guide query, so point directly "
        "to the small sewn neck/size label when it is identifiable. Do not point to "
        "a printed chest graphic, loose packaging, table marking, or unrelated tag."
        if allow_clothing_label
        else "Do not point to a clothing label or tag."
    )
    return (
        f"{context_prefix}Point to this keypoint on the garment: {description}. "
        "Return exactly one point only when "
        "that keypoint is clearly identifiable on visible garment fabric. If it is "
        "occluded, ambiguous, outside the image, or not confidently identifiable, "
        f"return no point. {label_rule} Do not point to the table, robot, gripper, "
        "printed graphic, image border, or another object. Follow the requested semantic "
        "region literally; do not silently replace a broad region request with an edge or boundary."
    )


def _partition_specs_for_axis_reuse(
    specs: Sequence[dict[str, Any]],
    axis_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split independent queries from semantic aliases of axis records."""

    axis_by_name = {
        str(record.get("name", "")): record
        for record in axis_records
        if isinstance(record, dict)
    }
    independent: list[dict[str, Any]] = []
    reused: dict[str, dict[str, Any]] = {}
    for spec in specs:
        semantic_name = str(spec["name"])
        source_axis_name = AXIS_RECORD_ALIASES.get(semantic_name)
        if source_axis_name is None:
            independent.append(spec)
            continue
        source = axis_by_name.get(source_axis_name)
        if source is None:
            raise RuntimeError(
                f"semantic keypoint {semantic_name!r} requires missing axis record "
                f"{source_axis_name!r}"
            )
        record = dict(source)
        record.update(
            {
                "name": semantic_name,
                "description": str(spec["description"]),
                "query_mode": "axis_alias_reuse_no_second_model_call",
                "reused_axis_record": True,
                "source_axis_name": source_axis_name,
                "source_axis_description": source.get("description"),
                "source_axis_prompt": source.get("prompt"),
            }
        )
        reused[semantic_name] = record
    return independent, reused


def _selected_token_probabilities(
    generated_ids: Any,
    generation_scores: Sequence[Any],
) -> list[float]:
    """Return selected-token probabilities for the first batch row."""

    return _selected_token_probabilities_for_row(
        generated_ids,
        generation_scores,
        row_index=0,
    )


def _selected_token_probabilities_for_row(
    generated_ids: Any,
    generation_scores: Sequence[Any],
    *,
    row_index: int,
) -> list[float]:
    """Return selected-token probabilities for one greedy batch row."""

    import torch

    ids = generated_ids[row_index].tolist()
    probabilities: list[float] = []
    for token_id, scores in zip(ids, generation_scores):
        probability = torch.softmax(scores[row_index].float(), dim=-1)[int(token_id)]
        probabilities.append(float(probability.detach().cpu().item()))
    return probabilities


def _extract_batch_points(
    raw: Any,
    *,
    batch_index: int,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """Keep only points decoded for this batch row's one image."""

    points: list[list[float]] = []
    if not isinstance(raw, list):
        return points
    for point in raw:
        if not isinstance(point, list) or len(point) < 2:
            continue
        # MolmoPoint's batched decoder returns
        # ``[example_id, image_index, x, y]``. A single-image compatibility
        # path may return only ``[x, y]``.
        if len(point) >= 4:
            try:
                if int(point[1]) != int(batch_index):
                    continue
            except (TypeError, ValueError):
                continue
        try:
            x_px, y_px = float(point[-2]), float(point[-1])
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(x_px)
            and math.isfinite(y_px)
            and 0 <= x_px < image_width
            and 0 <= y_px < image_height
        ):
            points.append([x_px, y_px])
    return points


def _infer_keypoint_record(
    spec: dict[str, Any],
    *,
    image: Any,
    processor: Any,
    model: Any,
    torch: Any,
    amp_dtype: Any,
    point_token_start: int,
    max_crops: int,
    max_new_tokens: int,
    use_stream: bool,
    context: str = "",
) -> dict[str, Any]:
    """Infer one independent prompt, optionally on its own CUDA stream."""

    semantic_name = str(spec["name"])
    prompt = _prompt(
        spec["description"],
        context,
        allow_clothing_label=semantic_name in NECK_LABEL_NAMES,
    )
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
        images_kwargs={"max_crops": max_crops},
    )
    metadata = inputs.pop("metadata")
    stream = torch.cuda.Stream() if use_stream else None
    stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_context:
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
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
    if stream is not None:
        stream.synchronize()

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
    points = _extract_batch_points(
        raw,
        batch_index=0,
        image_width=image.width,
        image_height=image.height,
    )
    selected_probabilities = _selected_token_probabilities(
        generated_ids,
        generation.scores,
    )
    token_ids = generated_ids[0].tolist()
    dynamic_point_probabilities = [
        probability
        for token_id, probability in zip(token_ids, selected_probabilities)
        if int(token_id) >= point_token_start
    ]
    point_probabilities = point_location_probabilities(
        dynamic_point_probabilities,
        len(points),
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
    return {
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


def _point_context(record: dict[str, Any] | None) -> str | None:
    """Return a compact coordinate string for one successful point record."""

    if not isinstance(record, dict) or record.get("status") != "point_returned":
        return None
    pixel = record.get("pixel_xy")
    if not isinstance(pixel, list) or len(pixel) != 2:
        return None
    return f"({float(pixel[0]):.1f},{float(pixel[1]):.1f})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--gpu-max-memory-gib",
        type=float,
        default=0.0,
        help="when positive, cap CUDA model placement and permit remaining layers on CPU",
    )
    parser.add_argument("--allow-cpu-offload", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--direct-keypoints",
        action="store_true",
        help=(
            "query only the requested keypoints without the generic garment-axis "
            "or neck-label/collar prequeries; useful for image-relative targets"
        ),
    )
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=2,
        help=(
            "number of independent anchor prompts to infer together; prompts "
            "remain separate questions"
        ),
    )
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
    if args.query_batch_size < 1:
        raise SystemExit("--query-batch-size must be positive")
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
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": model_dtype,
        "device_map": "auto",
        "local_files_only": args.local_files_only,
    }
    if args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load-in-8bit requires bitsandbytes in the Molmo environment"
            ) from exc
        # This warning is emitted once per quantized matmul in bitsandbytes
        # 0.50.x, producing tens of thousands of duplicate lines per point.
        logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
        # bitsandbytes 0.50.x currently calls ``.view(-1)`` on the CUDA
        # ``argwhere`` result used by its LLM.int8 outlier path.  With the
        # MolmoPoint point-predictor activation shape and our PyTorch build,
        # that result is non-contiguous and inference aborts before returning
        # even the first point.  A zero threshold disables only that optional
        # outlier split while retaining 8-bit model loading, which is required
        # for MolmoPoint-8B to fit beside the desktop workload on a 4090.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
            # MolmoPoint's custom coordinate head applies Linear layers to a
            # four-dimensional [batch, crop, patch, feature] tensor.  The
            # bitsandbytes Linear8bitLt CUDA kernel accepts only 2-D/3-D
            # activations, so this small task-specific head must stay in the
            # requested floating-point dtype.  The vision tower and connector
            # also stay floating point because point accuracy is the purpose
            # of this worker; the much larger text backbone remains quantized.
            llm_int8_skip_modules=["vit", "connector", "point_predictor"],
        )
    if args.gpu_max_memory_gib > 0:
        load_kwargs["max_memory"] = {
            0: f"{float(args.gpu_max_memory_gib):.2f}GiB",
            "cpu": "64GiB",
        }
        load_kwargs["offload_folder"] = str(args.output.parent / "model_offload")
        load_kwargs["offload_state_dict"] = True
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = sorted(
        {
            str(device)
            for device in device_map.values()
            if str(device) in {"cpu", "disk", "meta"}
        }
    )
    if offloaded and not args.allow_cpu_offload:
        raise RuntimeError(f"MolmoPoint model/vision tower was offloaded to {offloaded}")
    if offloaded:
        print(
            f"MolmoPoint CPU offload enabled for devices={offloaded}; "
            f"gpu_max_memory_gib={args.gpu_max_memory_gib:.2f}",
            file=sys.stderr,
            flush=True,
        )
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
    parallel_fallback = False
    for image_path, label in zip(args.image, args.label):
        image = Image.open(image_path).convert("RGB")
        base_infer_kwargs = {
            "image": image,
            "processor": processor,
            "model": model,
            "torch": torch,
            "amp_dtype": amp_dtype,
            "point_token_start": point_token_start,
            "max_crops": args.max_crops,
            "max_new_tokens": args.max_new_tokens,
        }
        if args.direct_keypoints:
            # Image-relative targets become less reliable when the generic axis
            # preamble redefines left/right as garment-as-worn. Query the
            # requested target directly and avoid two unnecessary model calls.
            records = [
                _infer_keypoint_record(
                    spec,
                    context=(
                        "Use the provided image coordinates and spatial wording in the "
                        "requested keypoint description exactly."
                    ),
                    use_stream=False,
                    **base_infer_kwargs,
                )
                for spec in specs
            ]
            for record in records:
                record.update(
                    {
                        "query_mode": "direct_requested_keypoint_query",
                        "reused_axis_record": False,
                        "source_axis_name": None,
                    }
                )
            torch.cuda.empty_cache()
            views.append(
                {
                    "label": str(label).upper(),
                    "image": str(image_path),
                    "image_size": [image.width, image.height],
                    "axis_reference": None,
                    "axis_records": [],
                    "records": records,
                    "axis_model_query_count": 0,
                    "semantic_model_query_count": len(specs),
                    "reused_axis_record_count": 0,
                    "neck_label_topology_guide": None,
                }
            )
            continue
        # Stage 1 is intentionally sequential. The current garment's own
        # collar-to-hem axis must be established before any left/right query.
        axis_records = [
            _infer_keypoint_record(
                spec,
                image=image,
                processor=processor,
                model=model,
                torch=torch,
                amp_dtype=amp_dtype,
                point_token_start=point_token_start,
                max_crops=args.max_crops,
                max_new_tokens=args.max_new_tokens,
                use_stream=False,
            )
            for spec in AXIS_SPECS
        ]
        axis_top = axis_records[0].get("pixel_xy")
        axis_bottom = axis_records[1].get("pixel_xy")
        axis_reference: dict[str, Any] = {
            "top_name": "axis_top",
            "bottom_name": "axis_bottom",
            "top_pixel_xy": axis_top,
            "bottom_pixel_xy": axis_bottom,
            "top_status": axis_records[0].get("status"),
            "bottom_status": axis_records[1].get("status"),
        }
        if (
            isinstance(axis_top, list)
            and len(axis_top) == 2
            and isinstance(axis_bottom, list)
            and len(axis_bottom) == 2
        ):
            dx = float(axis_bottom[0]) - float(axis_top[0])
            dy = float(axis_bottom[1]) - float(axis_top[1])
            if math.hypot(dx, dy) >= 1.0:
                axis_context = (
                    "Current garment centerline: collar-top=("
                    f"{float(axis_top[0]):.1f},{float(axis_top[1]):.1f}), bottom=("
                    f"{float(axis_bottom[0]):.1f},{float(axis_bottom[1]):.1f}), "
                    f"direction=({dx:.1f},{dy:.1f}). Use this garment centerline, "
                    "not image axes. Left/right means the garment's left/right as worn. "
                    f"The bottom hem is near y={float(axis_bottom[1]):.1f}; do not use "
                    "a sleeve edge or printed panel for hem anchors."
                )
            else:
                axis_context = (
                    "The axis_top and axis_bottom queries collapsed to the same pixel, "
                    "so the current garment centerline is unavailable. Do not infer a "
                    "centerline or left/right direction from this degenerate axis; use "
                    "independent semantic evidence and current appearance instead."
                )
        else:
            axis_context = (
                "Infer the current garment's own collar-to-hem centerline first. "
                "Use that centerline, not image axes; left/right means the garment's "
                "left/right as worn. If the centerline is not reliable, return no point."
            )
        axis_reference["axis_context"] = axis_context
        # Stage 2: locate the sewn neck/size label as a topology guide. Unlike
        # action anchors, this dedicated query is allowed to point to the label.
        neck_label_specs = [
            spec for spec in specs if str(spec["name"]) in NECK_LABEL_NAMES
        ]
        neck_label_records = [
            _infer_keypoint_record(
                spec,
                context=(
                    "Locate the shirt's sewn neck/size label as a topology guide. "
                    "On the approved flat reference it sits immediately inside and below "
                    "the neckline. This query may point to the label itself. The garment "
                    "may be folded or rotated, so do not assume image-top means collar."
                ),
                use_stream=False,
                **base_infer_kwargs,
            )
            for spec in neck_label_specs
        ]
        for record in neck_label_records:
            record.update(
                {
                    "query_mode": "independent_neck_label_topology_query",
                    "reused_axis_record": False,
                    "source_axis_name": None,
                    "topology_guide": True,
                }
            )
        neck_label_record = next(
            (
                record
                for record in neck_label_records
                if record.get("status") == "point_returned"
            ),
            None,
        )
        neck_label_point = _point_context(neck_label_record)

        # Stage 3: ask a fresh collar question. The label-to-neckline relation
        # is stronger than axis_top on folded garments, and axis_top is explicitly
        # demoted to a weak hypothesis that the model may contradict.
        collar_specs = [spec for spec in specs if str(spec["name"]) in COLLAR_NAMES]
        if neck_label_point is not None:
            collar_context = (
                f"A separate query located the sewn neck/size label at {neck_label_point}. "
                "In the approved flat reference, that label is immediately inside/below "
                "the neckline and the collar band surrounds its upper/outer side. Use the "
                "label only to localize topology: point to adjacent collar-band or neck-"
                "opening fabric, never to the label itself. Do not substitute a sleeve, "
                "shoulder, chest fold, or broad interior panel. The earlier axis_top is only "
                "a weak folded-garment hypothesis; ignore it when it conflicts with the label."
            )
            collar_query_mode = "neck_label_guided_independent_collar_query"
        else:
            collar_context = (
                "The sewn neck/size label was not reliably located. Make an independent "
                "collar decision from visible garment topology. The earlier axis_top is a "
                "weak hypothesis only and must not be copied. Do not substitute a sleeve, "
                "shoulder, chest fold, or broad interior panel; return no point if uncertain."
            )
            collar_query_mode = "independent_collar_query_without_neck_label"
        collar_records = [
            _infer_keypoint_record(
                spec,
                context=collar_context,
                use_stream=False,
                **base_infer_kwargs,
            )
            for spec in collar_specs
        ]
        for record in collar_records:
            record.update(
                {
                    "query_mode": collar_query_mode,
                    "reused_axis_record": False,
                    "source_axis_name": None,
                    "topology_guide_name": (
                        str(neck_label_record["name"])
                        if neck_label_record is not None
                        else None
                    ),
                    "topology_guide_pixel_xy": (
                        neck_label_record.get("pixel_xy")
                        if neck_label_record is not None
                        else None
                    ),
                }
            )
        collar_record = next(
            (
                record
                for record in collar_records
                if record.get("status") == "point_returned"
            ),
            None,
        )
        collar_point = _point_context(collar_record)

        remaining_specs = [
            spec
            for spec in specs
            if str(spec["name"]) not in NECK_LABEL_NAMES | COLLAR_NAMES
        ]
        remaining_context = axis_context
        if collar_point is not None:
            remaining_context += (
                f" A separate label-guided collar query proposed {collar_point}; use it as "
                "a semantic clue for shoulder side identity, not as an unquestionable fact."
            )
        inferred_records: list[dict[str, Any]] = []
        try:
            if args.query_batch_size == 1:
                inferred_records = [
                    _infer_keypoint_record(
                        spec,
                        context=remaining_context,
                        use_stream=False,
                        **base_infer_kwargs,
                    )
                    for spec in remaining_specs
                ]
            else:
                # MolmoPoint's released remote model currently rejects true
                # batch_size>1 in its point embedding path. Use separate CUDA
                # streams instead: every prompt remains batch_size=1 and keeps
                # its independent point-token constraints, while independent
                # anchors can overlap on the GPU.
                with ThreadPoolExecutor(max_workers=args.query_batch_size) as pool:
                    futures = [
                        pool.submit(
                            _infer_keypoint_record,
                            spec,
                            context=remaining_context,
                            use_stream=True,
                            **base_infer_kwargs,
                        )
                        for spec in remaining_specs
                    ]
                    inferred_records = [future.result() for future in futures]
        except Exception as exc:
            if args.query_batch_size == 1:
                raise
            print(
                f"parallel Molmo queries failed ({type(exc).__name__}: {exc}); "
                "retrying sequentially",
                file=sys.stderr,
            )
            parallel_fallback = True
            torch.cuda.synchronize()
            inferred_records = [
                _infer_keypoint_record(
                    spec,
                    context=remaining_context,
                    use_stream=False,
                    **base_infer_kwargs,
                )
                for spec in remaining_specs
            ]
        for record in inferred_records:
            record.update(
                {
                    "query_mode": "independent_keypoint_query",
                    "reused_axis_record": False,
                    "source_axis_name": None,
                }
            )
        inferred_by_name = {
            str(record["name"]): record
            for record in neck_label_records + collar_records + inferred_records
        }
        records = [inferred_by_name[str(spec["name"])] for spec in specs]
        torch.cuda.empty_cache()
        views.append(
            {
                "label": str(label).upper(),
                "image": str(image_path),
                "image_size": [image.width, image.height],
                "axis_reference": axis_reference,
                "axis_records": axis_records,
                "records": records,
                "axis_model_query_count": len(axis_records),
                "semantic_model_query_count": len(specs),
                "reused_axis_record_count": 0,
                "neck_label_topology_guide": (
                    neck_label_record if neck_label_record is not None else None
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "model": args.model,
        "query_mode": (
            "direct_requested_keypoints"
            if args.direct_keypoints
            else "axis_then_neck_label_guided_independent_collar"
        ),
        "axis_record_aliases": dict(AXIS_RECORD_ALIASES),
        "axis_specs": list(AXIS_SPECS),
        "query_batch_size": args.query_batch_size,
        "parallel_fallback": parallel_fallback,
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
