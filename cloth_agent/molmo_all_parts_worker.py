"""GPU worker for zero-shot Molmo garment-part points with UNKNOWN allowed.

Each garment part remains an independent question. Independent questions are
batched in small groups so one model invocation can serve several anchors
without turning them into a compound prompt.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SPECS = (
    (
        "garment_center",
        "center of the shirt fabric body halfway between the collar and the bottom hem; do not point to the printed graphic, label, or table",
        (255, 40, 40),
    ),
    ("neckline", "collar, neckline, or neck opening", (255, 170, 0)),
    (
        "left_shoulder",
        "the garment's left shoulder as worn: the sleeve-to-torso seam on the left side of its own centerline, not the sleeve tip",
        (50, 180, 255),
    ),
    (
        "right_shoulder",
        "the garment's right shoulder as worn: the sleeve-to-torso seam on the right side of its own centerline, not the sleeve tip",
        (80, 220, 80),
    ),
    (
        "left_sleeve_tip",
        "the outermost endpoint on the garment silhouette of the left sleeve as worn, on the left side of its own centerline",
        (180, 80, 255),
    ),
    (
        "right_sleeve_tip",
        "the outermost endpoint on the garment silhouette of the right sleeve as worn, on the right side of its own centerline",
        (255, 80, 190),
    ),
    (
        "left_bottom_hem",
        "the left endpoint of the shirt's actual bottom outer hem boundary, on the garment silhouette and not on the printed graphic",
        (80, 220, 220),
    ),
    (
        "right_bottom_hem",
        "the right endpoint of the shirt's actual bottom outer hem boundary, on the garment silhouette and not on the printed graphic",
        (220, 220, 60),
    ),
    (
        "lower_left_half_center",
        "a point on plain shirt fabric in the lower half, between the garment centerline and the left outer edge; avoid the printed graphic",
        (120, 255, 120),
    ),
    (
        "lower_right_half_center",
        "a point on plain shirt fabric in the lower half, between the garment centerline and the right outer edge; avoid the printed graphic",
        (120, 180, 255),
    ),
)

AXIS_SPECS = (
    (
        "axis_top",
        "the center of the collar/neck opening; this is the top reference point of the garment's own longitudinal centerline",
        (255, 255, 255),
    ),
    (
        "axis_bottom",
        "the center of the garment's bottom hem; this is the bottom reference point of the garment's own longitudinal centerline",
        (255, 255, 255),
    ),
)


def _prompt(description: str, context: str = "") -> str:
    context_prefix = f"{context.strip()}\n" if context.strip() else ""
    return (
        f"{context_prefix}Point to this garment's {description}. If this part cannot be identified "
        "with confidence in the current image, answer UNKNOWN. The point must be on visible shirt fabric or its outer silhouette; never point to the printed graphic, clothing label, table, robot, or image border. For an edge landmark, point exactly on the garment boundary. Return at most one point."
    )


def _extract_batch_pixels(
    raw: object,
    *,
    batch_index: int,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """Keep only points decoded for one batch row's single image."""

    pixels: list[list[float]] = []
    if not isinstance(raw, list):
        return pixels
    for point in raw:
        if not isinstance(point, list) or len(point) < 2:
            continue
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
            pixels.append([x_px, y_px])
    return pixels


def _infer_all_parts_record(
    spec: tuple[str, str, tuple[int, int, int]],
    *,
    image: Any,
    processor: Any,
    model: Any,
    torch: Any,
    amp_dtype: Any,
    max_crops: int,
    max_new_tokens: int,
    context: str = "",
    use_stream: bool,
) -> dict[str, object]:
    """Infer one independent garment-part prompt on an optional CUDA stream."""

    name, description, color = spec
    query = _prompt(description, context)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
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
            output = model.generate(
                **inputs,
                logits_processor=model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=max_new_tokens,
            )
    if stream is not None:
        stream.synchronize()

    generated_tokens = output[:, inputs["input_ids"].size(1) :]
    text = processor.post_process_image_text_to_text(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    raw = model.extract_image_points(
        text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )
    raw = raw.tolist() if hasattr(raw, "tolist") else raw
    pixels = _extract_batch_pixels(
        raw,
        batch_index=0,
        image_width=image.width,
        image_height=image.height,
    )
    selected = pixels[0] if len(pixels) == 1 else None
    return {
        "name": name,
        "description": description,
        "color": list(color),
        "prompt": query,
        "generated_text": text,
        "raw_points": pixels,
        "raw_point_count": len(pixels),
        "status": "point_returned" if selected is not None else "unknown",
        "selected_pixel_xy": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=2,
        help="number of independent garment-part prompts inferred together",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.image) != len(args.label) or not args.image:
        raise SystemExit("--image and --label counts must match")
    if any(not path.is_file() for path in args.image):
        raise SystemExit("every --image must be an existing file")
    if args.query_batch_size < 1:
        raise SystemExit("--query-batch-size must be positive")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("MolmoPoint requires CUDA")
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
        raise RuntimeError(f"MolmoPoint model was offloaded to {offloaded}")
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

    views: list[dict[str, object]] = []
    parallel_fallback = False
    for image_path, image_label in zip(args.image, args.label):
        image = Image.open(image_path).convert("RGB")
        records: list[dict[str, object]] = []
        # Stage 1 is deliberately sequential: establish the garment's own
        # longitudinal axis before asking any left/right question.
        axis_records = [
            _infer_all_parts_record(
                spec,
                image=image,
                processor=processor,
                model=model,
                torch=torch,
                amp_dtype=amp_dtype,
                max_crops=args.max_crops,
                max_new_tokens=args.max_new_tokens,
                context="",
                use_stream=False,
            )
            for spec in AXIS_SPECS
        ]
        axis_top = axis_records[0].get("selected_pixel_xy")
        axis_bottom = axis_records[1].get("selected_pixel_xy")
        axis_reference: dict[str, object] = {
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
            axis_context = (
                "Garment centerline reference: collar-top=("
                f"{float(axis_top[0]):.1f},{float(axis_top[1]):.1f}), bottom=("
                f"{float(axis_bottom[0]):.1f},{float(axis_bottom[1]):.1f}), "
                f"direction=({dx:.1f},{dy:.1f}). "
                "Use this garment centerline, not image axes. Left/right means "
                "the garment's left/right as worn. The bottom outer hem is the "
                f"lowest shirt boundary near y={float(axis_bottom[1]):.1f}; for "
                "bottom-hem anchors do not choose a sleeve edge or the printed "
                "panel. For the garment center and lower-half anchors, choose "
                "plain shirt fabric below or beside the printed panel."
            )
            axis_reference["axis_context"] = axis_context
        else:
            axis_context = (
                "Infer the garment's own collar-to-hem centerline first. Use that "
                "centerline, not image axes; left/right means the garment's "
                "left/right as worn."
            )
            axis_reference["axis_context"] = axis_context

        main_specs = tuple(spec for spec in SPECS if spec[0] != "neckline")
        infer_kwargs = {
            "image": image,
            "processor": processor,
            "model": model,
            "torch": torch,
            "amp_dtype": amp_dtype,
            "max_crops": args.max_crops,
            "max_new_tokens": args.max_new_tokens,
            "context": axis_context,
        }
        try:
            if args.query_batch_size == 1:
                records = [
                    _infer_all_parts_record(spec, use_stream=False, **infer_kwargs)
                    for spec in main_specs
                ]
            else:
                # The released remote model has a batch-size>1 bug in its
                # point-embedding path. Separate CUDA streams preserve one
                # prompt per batch_size=1 call while overlapping independent
                # anchor inference when the GPU has headroom.
                with ThreadPoolExecutor(max_workers=args.query_batch_size) as pool:
                    futures = [
                        pool.submit(
                            _infer_all_parts_record,
                            spec,
                            use_stream=True,
                            **infer_kwargs,
                        )
                        for spec in main_specs
                    ]
                    records = [future.result() for future in futures]
        except Exception as exc:
            if args.query_batch_size == 1:
                raise
            print(
                f"parallel Molmo all-parts queries failed ({type(exc).__name__}: {exc}); "
                "retrying sequentially",
                file=sys.stderr,
            )
            parallel_fallback = True
            torch.cuda.synchronize()
            records = [
                _infer_all_parts_record(spec, use_stream=False, **infer_kwargs)
                for spec in main_specs
            ]
        # Reuse the axis-top query as the public neckline record, then restore
        # the canonical ten-part order. The axis-bottom point is retained as a
        # reference line, not counted as a garment-part anchor.
        neckline_spec = next(spec for spec in SPECS if spec[0] == "neckline")
        neckline_record = dict(axis_records[0])
        neckline_record.update(
            {
                "name": neckline_spec[0],
                "description": neckline_spec[1],
                "color": list(neckline_spec[2]),
            }
        )
        record_by_name = {str(record["name"]): record for record in records}
        record_by_name["neckline"] = neckline_record
        records = [record_by_name[spec[0]] for spec in SPECS]
        torch.cuda.empty_cache()
        views.append(
            {
                "label": image_label,
                "image": str(image_path),
                "image_size": list(image.size),
                "axis_reference": axis_reference,
                "records": records,
            }
        )

    payload = {
        "model": args.model,
        "query_mode": "axis_first_zero_shot_all_parts_point_or_unknown_batched",
        "query_batch_size": args.query_batch_size,
        "parallel_fallback": parallel_fallback,
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
