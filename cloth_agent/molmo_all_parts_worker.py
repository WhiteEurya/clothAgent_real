"""GPU worker for zero-shot Molmo garment-part points with UNKNOWN allowed."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SPECS = (
    ("garment_center", "geometric center of the whole garment", (255, 40, 40)),
    ("neckline", "collar, neckline, or neck opening", (255, 170, 0)),
    ("left_shoulder", "image-left shoulder or upper-left shoulder seam", (50, 180, 255)),
    ("right_shoulder", "image-right shoulder or upper-right shoulder seam", (80, 220, 80)),
    ("left_sleeve_tip", "outermost image-left sleeve tip or upper edge", (180, 80, 255)),
    ("right_sleeve_tip", "outermost image-right sleeve tip or upper edge", (255, 80, 190)),
    ("left_bottom_hem", "image-left end of the bottom hem", (80, 220, 220)),
    ("right_bottom_hem", "image-right end of the bottom hem", (220, 220, 60)),
    (
        "lower_left_half_center",
        "center of the lower image-left half of the garment",
        (120, 255, 120),
    ),
    (
        "lower_right_half_center",
        "center of the lower image-right half of the garment",
        (120, 180, 255),
    ),
)


def _prompt(description: str) -> str:
    return (
        f"Point to this garment's {description}. If this part cannot be identified "
        "with confidence in the current image, answer UNKNOWN. Return at most one point."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.image) != len(args.label) or not args.image:
        raise SystemExit("--image and --label counts must match")
    if any(not path.is_file() for path in args.image):
        raise SystemExit("every --image must be an existing file")

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
    for image_path, image_label in zip(args.image, args.label):
        image = Image.open(image_path).convert("RGB")
        records: list[dict[str, object]] = []
        for name, description, color in SPECS:
            query = _prompt(description)
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
                output = model.generate(
                    **inputs,
                    logits_processor=model.build_logit_processor_from_inputs(inputs),
                    max_new_tokens=args.max_new_tokens,
                )
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
            pixels: list[list[float]] = []
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
                        pixels.append([x_px, y_px])
            selected = pixels[0] if len(pixels) == 1 else None
            records.append(
                {
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
            )
            del output, inputs, metadata
            torch.cuda.empty_cache()
        views.append(
            {
                "label": image_label,
                "image": str(image_path),
                "image_size": list(image.size),
                "records": records,
            }
        )

    payload = {
        "model": args.model,
        "query_mode": "zero_shot_all_parts_point_or_unknown",
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
