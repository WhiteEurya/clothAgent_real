"""MolmoPoint worker for independent one-point-per-landmark queries.

The model is loaded once, then queried separately for each prompt. This is
more reliable than asking MolmoPoint to emit a large ordered point group and
avoids loading eight copies of the model in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if len(args.image) not in {1, 2} or any(not path.is_file() for path in args.image):
        raise SystemExit("one or two existing --image files are required")
    if not args.prompt:
        raise SystemExit("at least one --prompt is required")

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
    images = [Image.open(path).convert("RGB") for path in args.image]
    amp_dtype = (
        torch.bfloat16
        if args.dtype == "bf16" and torch.cuda.is_bf16_supported()
        else torch.float16
        if args.dtype != "fp32"
        else torch.float32
    )
    points: list[dict[str, object]] = []
    generated_text: list[dict[str, object]] = []
    for image_index, image in enumerate(images):
        for prompt_index, prompt in enumerate(args.prompt):
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
            if not isinstance(raw, list) or len(raw) != 1:
                raise RuntimeError(
                    f"landmark query {prompt_index + 1} on image {image_index} "
                    f"returned {len(raw) if isinstance(raw, list) else 'invalid'} points; expected exactly 1"
                )
            point = raw[0]
            if not isinstance(point, list) or len(point) < 2:
                raise RuntimeError(f"invalid Molmo point for query {prompt_index + 1}: {point!r}")
            points.append(
                {
                    "image_index": image_index,
                    "prompt_index": prompt_index,
                    "pixel_xy": [float(point[-2]), float(point[-1])],
                }
            )
            generated_text.append(
                {
                    "image_index": image_index,
                    "prompt_index": prompt_index,
                    "text": text,
                }
            )
            del output, inputs, metadata
            torch.cuda.empty_cache()
    payload = {
        "model": args.model,
        "image_sizes": [list(image.size) for image in images],
        "landmark_points": points,
        "generated_text": generated_text,
        "prompt_count": len(args.prompt),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
