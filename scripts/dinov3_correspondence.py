#!/usr/bin/env python3
"""Frozen DINOv3 dense correspondence for flat/folded garment images.

This is deliberately independent from the robot and Claude code.  It loads a
reference image, a JSON list of manually clicked reference points, and one or
more current images.  For every reference point it extracts the corresponding
reference patch token, computes cosine similarity against every current patch
token, and saves the top-1 match plus a similarity heatmap.

Example::

    python scripts/dinov3_correspondence.py \
      --reference flat.png \
      --points flat_points.json \
      --current single_fold=fold1.png \
      --current fold2=fold2.png \
      --model facebook/dinov3-vitb16-pretrain-lvd1689m \
      --output results/dinov3/fold_set_01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("PyTorch is required; run this script in the DINOv3 environment") from exc
    return torch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "current"


def _parse_current(raw_values: list[str]) -> list[tuple[str, Path]]:
    if not raw_values:
        raise ValueError("at least one --current NAME=IMAGE argument is required")
    parsed: list[tuple[str, Path]] = []
    used: set[str] = set()
    for raw in raw_values:
        if "=" in raw:
            name, path_text = raw.split("=", 1)
            name = name.strip()
        else:
            path_text = raw
            name = Path(path_text).stem
        if not name:
            raise ValueError(f"invalid --current value: {raw!r}")
        if name in used:
            raise ValueError(f"duplicate current image name: {name}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        used.add(name)
        parsed.append((name, path))
    return parsed


@dataclass(frozen=True)
class Point:
    point_id: int
    label: str
    x: float
    y: float


def _load_points(path: Path, reference: Image.Image) -> list[Point]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_points = payload.get("points") if isinstance(payload, dict) else payload
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"{path} does not contain a non-empty points list")
    width, height = reference.size
    points: list[Point] = []
    for index, raw in enumerate(raw_points, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"point {index} is not an object")
        x, y = float(raw["x"]), float(raw["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"point {index} is not finite")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"point {index} ({x}, {y}) lies outside reference image {width}x{height}")
        points.append(Point(int(raw.get("id", index)), str(raw.get("label", f"p{index:02d}")), x, y))
    return points


class FrozenDINOv3:
    """Small Hugging Face adapter that exposes normalized patch tokens only."""

    def __init__(
        self,
        model_id: str,
        *,
        input_size: int,
        device: str,
        local_files_only: bool,
    ) -> None:
        torch = _require_torch()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - depends on user env
            raise RuntimeError("transformers is required to load DINOv3") from exc
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.torch = torch
        self.input_size = int(input_size)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
        self.processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModel.from_pretrained(model_id, **kwargs)
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)
        config = getattr(self.model, "config", None)
        patch_size = getattr(config, "patch_size", None)
        if patch_size is None and getattr(config, "vision_config", None) is not None:
            patch_size = getattr(config.vision_config, "patch_size", None)
        if patch_size is None:
            patch_size = 16
        self.patch_size = int(patch_size)
        if self.input_size % self.patch_size:
            raise ValueError(f"input_size {self.input_size} must be divisible by patch_size {self.patch_size}")
        mean = getattr(self.processor, "image_mean", None) or [0.485, 0.456, 0.406]
        std = getattr(self.processor, "image_std", None) or [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.grid_h = self.input_size // self.patch_size
        self.grid_w = self.input_size // self.patch_size

    def _tensor(self, image: Image.Image):
        torch = self.torch
        resized = image.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor - self.mean) / self.std

    def _patch_tokens(self, output):
        torch = self.torch
        candidates = []
        for name in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state", "hidden_states"):
            value = getattr(output, name, None)
            if name == "hidden_states" and value is not None:
                if isinstance(value, (tuple, list)):
                    value = value[-1] if value else None
            if value is not None:
                candidates.append(value)
        for value in candidates:
            if value.ndim == 4:
                # Some vision backbones return [B, D, H, W].
                if value.shape[-2:] == (self.grid_h, self.grid_w):
                    return value.flatten(2).transpose(1, 2)
                if value.shape[1:3] == (self.grid_h, self.grid_w):
                    return value.flatten(1, 2)
            if value.ndim != 3:
                continue
            expected = self.grid_h * self.grid_w
            if value.shape[1] < expected:
                continue
            # DINOv2/DINOv3 variants may prepend CLS and register tokens.  The
            # patch tokens are the final regular grid tokens in all supported
            # Hugging Face outputs, so discard any prefix conservatively.
            tokens = value[:, -expected:, :]
            return tokens
        raise RuntimeError("DINOv3 output did not contain a patch-token tensor")

    def encode(self, image: Image.Image):
        torch = self.torch
        with torch.inference_mode():
            output = self.model(pixel_values=self._tensor(image))
            tokens = self._patch_tokens(output).float()
            tokens = torch.nn.functional.normalize(tokens, dim=-1)
        return tokens[0].reshape(self.grid_h, self.grid_w, -1)

    def point_to_patch(self, x: float, y: float, image_size: tuple[int, int]) -> tuple[int, int]:
        width, height = image_size
        x_input = 0.0 if width <= 1 else x * (self.input_size - 1) / (width - 1)
        y_input = 0.0 if height <= 1 else y * (self.input_size - 1) / (height - 1)
        px = int(round((x_input - self.patch_size / 2.0) / self.patch_size))
        py = int(round((y_input - self.patch_size / 2.0) / self.patch_size))
        return (
            min(self.grid_w - 1, max(0, px)),
            min(self.grid_h - 1, max(0, py)),
        )

    def patch_to_point(self, px: int, py: int, image_size: tuple[int, int]) -> tuple[float, float]:
        width, height = image_size
        x_input = (px + 0.5) * self.patch_size
        y_input = (py + 0.5) * self.patch_size
        x = 0.0 if self.input_size <= 1 else x_input * (width - 1) / (self.input_size - 1)
        y = 0.0 if self.input_size <= 1 else y_input * (height - 1) / (self.input_size - 1)
        return float(min(width - 1, max(0.0, x))), float(min(height - 1, max(0.0, y)))


def _color(index: int, total: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[int(round(179 * index / max(1, total))), 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_points(image: np.ndarray, points: list[tuple[float, float, str]], colors: list[tuple[int, int, int]]) -> np.ndarray:
    canvas = image.copy()
    for (x, y, label), color in zip(points, colors):
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _resize_to_height(image: np.ndarray, target_height: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = target_height / max(1, height)
    resized = cv2.resize(image, (max(1, int(round(width * scale))), target_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _side_by_side(
    reference: np.ndarray,
    current: np.ndarray,
    ref_points: list[tuple[float, float, str]],
    cur_points: list[tuple[float, float, str]],
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    target_height = max(reference.shape[0], current.shape[0], 720)
    left, left_scale = _resize_to_height(reference, target_height)
    right, right_scale = _resize_to_height(current, target_height)
    canvas = np.zeros((target_height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] :] = right
    for (rx, ry, label), (cx, cy, _), color in zip(ref_points, cur_points, colors):
        p0 = (int(round(rx * left_scale)), int(round(ry * left_scale)))
        p1 = (left.shape[1] + int(round(cx * right_scale)), int(round(cy * right_scale)))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    canvas[:, left.shape[1] - 2 : left.shape[1] + 2] = (255, 255, 255)
    return canvas


def _heatmap_overlay(image: np.ndarray, similarity: np.ndarray, point: tuple[float, float, str], color: tuple[int, int, int]) -> np.ndarray:
    values = similarity.astype(np.float32)
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    normalized = np.clip((values - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    heat = cv2.applyColorMap(np.uint8(np.round(normalized * 255.0)), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlay = cv2.addWeighted(image, 0.43, heat, 0.57, 0.0)
    x, y, label = point
    center = (int(round(x)), int(round(y)))
    cv2.drawMarker(overlay, center, color, cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def _save_heatmap_grid(images: list[np.ndarray], labels: list[str], output: Path) -> None:
    if not images:
        return
    thumb_w, thumb_h = 360, 270
    columns = min(5, len(images))
    rows = int(math.ceil(len(images) / columns))
    grid = np.zeros((rows * thumb_h, columns * thumb_w, 3), dtype=np.uint8)
    for index, (image, label) in enumerate(zip(images, labels)):
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, 28), (0, 0, 0), -1)
        cv2.putText(thumb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        row, column = divmod(index, columns)
        grid[row * thumb_h : (row + 1) * thumb_h, column * thumb_w : (column + 1) * thumb_w] = thumb
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)


def run(
    reference_path: Path,
    points_path: Path,
    currents: list[tuple[str, Path]],
    output_dir: Path,
    *,
    model_id: str,
    input_size: int,
    device: str,
    local_files_only: bool,
    save_individual_heatmaps: bool,
) -> dict:
    reference = Image.open(reference_path).convert("RGB")
    points = _load_points(points_path, reference)
    extractor = FrozenDINOv3(
        model_id,
        input_size=input_size,
        device=device,
        local_files_only=local_files_only,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_bgr = cv2.cvtColor(np.asarray(reference), cv2.COLOR_RGB2BGR)
    colors = [_color(index, len(points)) for index in range(len(points))]
    ref_overlay = _draw_points(
        reference_bgr,
        [(point.x, point.y, str(point.point_id)) for point in points],
        colors,
    )
    cv2.imwrite(str(output_dir / "reference_points.png"), ref_overlay)
    reference_features = extractor.encode(reference)
    ref_patch_indices = [extractor.point_to_patch(point.x, point.y, reference.size) for point in points]
    ref_vectors = extractor.torch.stack(
        [reference_features[py, px] for px, py in ref_patch_indices],
        dim=0,
    )

    run_payload = {
        "version": 1,
        "created_at": _now(),
        "reference": str(reference_path),
        "points": str(points_path),
        "model": model_id,
        "input_size": input_size,
        "patch_size": extractor.patch_size,
        "feature_grid": [extractor.grid_h, extractor.grid_w],
        "device": str(extractor.device),
        "frozen": True,
        "currents": {},
    }

    for current_name, current_path in currents:
        current = Image.open(current_path).convert("RGB")
        current_bgr = cv2.cvtColor(np.asarray(current), cv2.COLOR_RGB2BGR)
        current_features = extractor.encode(current)
        current_flat = current_features.reshape(-1, current_features.shape[-1])
        similarity = ref_vectors @ current_flat.T
        top_values, top_indices = similarity.max(dim=1)
        current_matches: list[dict] = []
        current_points: list[tuple[float, float, str]] = []
        heatmaps: list[np.ndarray] = []
        heatmap_labels: list[str] = []
        raw_similarity = similarity.reshape(len(points), extractor.grid_h, extractor.grid_w).detach().cpu().numpy().astype(np.float32)
        np.save(output_dir / f"current_{_slug(current_name)}_similarity.npy", raw_similarity)
        for index, point in enumerate(points):
            flat_index = int(top_indices[index].item())
            py, px = divmod(flat_index, extractor.grid_w)
            match_x, match_y = extractor.patch_to_point(px, py, current.size)
            score = float(top_values[index].item())
            current_matches.append(
                {
                    "point_id": point.point_id,
                    "label": point.label,
                    "reference_xy": [point.x, point.y],
                    "reference_patch_xy": [ref_patch_indices[index][0], ref_patch_indices[index][1]],
                    "current_xy": [match_x, match_y],
                    "current_patch_xy": [px, py],
                    "cosine_similarity": score,
                }
            )
            current_points.append((match_x, match_y, str(point.point_id)))
            heatmap = _heatmap_overlay(
                current_bgr,
                raw_similarity[index],
                (match_x, match_y, str(point.point_id)),
                colors[index],
            )
            heatmaps.append(heatmap)
            heatmap_labels.append(f"{point.point_id} {point.label}  sim={score:.3f}")
            if save_individual_heatmaps:
                cv2.imwrite(
                    str(output_dir / f"current_{_slug(current_name)}_point_{point.point_id:02d}_heatmap.png"),
                    heatmap,
                )

        current_overlay = _draw_points(current_bgr, current_points, colors)
        side_by_side = _side_by_side(
            ref_overlay,
            current_overlay,
            [(point.x, point.y, str(point.point_id)) for point in points],
            current_points,
            colors,
        )
        slug = _slug(current_name)
        cv2.imwrite(str(output_dir / f"current_{slug}_matches.png"), current_overlay)
        cv2.imwrite(str(output_dir / f"current_{slug}_correspondence_side_by_side.png"), side_by_side)
        _save_heatmap_grid(heatmaps, heatmap_labels, output_dir / f"current_{slug}_similarity_heatmaps.png")
        json_path = output_dir / f"current_{slug}_matches.json"
        json_path.write_text(
            json.dumps(
                {
                    "reference": str(reference_path),
                    "current": str(current_path),
                    "model": model_id,
                    "feature_grid": [extractor.grid_h, extractor.grid_w],
                    "matches": current_matches,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (output_dir / f"current_{slug}_matches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(current_matches[0]))
            writer.writeheader()
            writer.writerows(current_matches)
        run_payload["currents"][current_name] = {
            "image": str(current_path),
            "matches_json": json_path.name,
            "matches_csv": f"current_{slug}_matches.csv",
            "similarity_npy": f"current_{slug}_similarity.npy",
            "matches_overlay": f"current_{slug}_matches.png",
            "side_by_side": f"current_{slug}_correspondence_side_by_side.png",
            "similarity_heatmaps": f"current_{slug}_similarity_heatmaps.png",
        }
    (output_dir / "run.json").write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved DINOv3 correspondence results to {output_dir}")
    return run_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--current", required=True, action="append", metavar="NAME=IMAGE")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--input-size", type=int, default=448, help="square model input; must be divisible by patch size")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-individual-heatmaps", action="store_true")
    args = parser.parse_args()
    reference = args.reference.expanduser().resolve()
    points = args.points.expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if not points.is_file():
        raise FileNotFoundError(points)
    run(
        reference,
        points,
        _parse_current(args.current),
        args.output.expanduser().resolve(),
        model_id=args.model,
        input_size=args.input_size,
        device=args.device,
        local_files_only=args.local_files_only,
        save_individual_heatmaps=not args.no_individual_heatmaps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
