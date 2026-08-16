#!/usr/bin/env python3
"""Human review of frozen DINOv3 top-1 matches.

For each reference point the script shows the reference point, the proposed
current match, and its similarity heatmap.  Press ``y`` when the match lands
on the same physical/material region, ``n`` when it does not, ``u`` for
uncertain, and ``q`` to save and stop.  This intentionally does not attempt to
infer correctness from image heuristics: the requested metric is visual.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _fit_height(image: np.ndarray, height: int) -> tuple[np.ndarray, float]:
    scale = height / max(1, image.shape[0])
    return cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), height), interpolation=cv2.INTER_AREA), scale


def _show_match(reference: np.ndarray, current: np.ndarray, match: dict, heatmap_path: Path, index: int, total: int) -> None:
    ref = reference.copy()
    cur = current.copy()
    rx, ry = match["reference_xy"]
    cx, cy = match["current_xy"]
    cv2.drawMarker(ref, (int(round(rx)), int(round(ry))), (0, 255, 255), cv2.MARKER_CROSS, 32, 3, cv2.LINE_AA)
    cv2.putText(ref, str(match["point_id"]), (int(rx) + 10, int(ry) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(cur, (int(round(cx)), int(round(cy))), (0, 255, 255), cv2.MARKER_CROSS, 32, 3, cv2.LINE_AA)
    cv2.putText(cur, str(match["point_id"]), (int(cx) + 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    target_height = max(ref.shape[0], cur.shape[0], 620)
    ref, ref_scale = _fit_height(ref, target_height)
    cur, cur_scale = _fit_height(cur, target_height)
    side = np.zeros((target_height, ref.shape[1] + cur.shape[1], 3), dtype=np.uint8)
    side[:, : ref.shape[1]] = ref
    side[:, ref.shape[1] :] = cur
    cv2.line(
        side,
        (int(round(rx * ref_scale)), int(round(ry * ref_scale))),
        (ref.shape[1] + int(round(cx * cur_scale)), int(round(cy * cur_scale))),
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    if heatmap_path.is_file():
        heatmap = _load_image(heatmap_path)
        heatmap, _ = _fit_height(heatmap, target_height)
        panel = np.concatenate([side, heatmap], axis=1)
    else:
        panel = side
    title = (
        f"{index}/{total}  {match.get('label', '')}  sim={float(match['cosine_similarity']):.4f}  "
        "y=same material, n=wrong, u=uncertain, q=save"
    )
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("DINOv3 manual review", panel)


def review(result_dir: Path, current_name: str, output_path: Path) -> dict:
    run_payload = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    current_info = run_payload.get("currents", {}).get(current_name)
    if current_info is None:
        raise KeyError(f"current name {current_name!r} not found; choose from {sorted(run_payload.get('currents', {}))}")
    matches_path = result_dir / current_info["matches_json"]
    payload = json.loads(matches_path.read_text(encoding="utf-8"))
    reference = _load_image(Path(run_payload["reference"]))
    current = _load_image(Path(current_info["image"]))
    matches = payload["matches"]
    judgments: list[dict] = []
    for index, match in enumerate(matches, start=1):
        heatmap = result_dir / f"current_{current_name.replace(' ', '_')}_point_{int(match['point_id']):02d}_heatmap.png"
        if not heatmap.is_file():
            # Correspondence sanitizes names; fall back to the current result
            # entry's filename prefix when the name contains punctuation.
            import re

            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", current_name).strip("._") or "current"
            heatmap = result_dir / f"current_{slug}_point_{int(match['point_id']):02d}_heatmap.png"
        _show_match(reference, current, match, heatmap, index, len(matches))
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("y"), ord("Y"), ord("n"), ord("N"), ord("u"), ord("U"), ord("q"), ord("Q")):
                break
        if key in (ord("q"), ord("Q")):
            break
        judgment = {ord("y"): "same_material", ord("Y"): "same_material", ord("n"): "wrong", ord("N"): "wrong", ord("u"): "uncertain", ord("U"): "uncertain"}[key]
        judgments.append({"point_id": match["point_id"], "label": match.get("label", ""), "judgment": judgment})
    cv2.destroyAllWindows()
    same = sum(item["judgment"] == "same_material" for item in judgments)
    wrong = sum(item["judgment"] == "wrong" for item in judgments)
    uncertain = sum(item["judgment"] == "uncertain" for item in judgments)
    result = {
        "created_at": _now(),
        "current": current_name,
        "matches": len(matches),
        "reviewed": len(judgments),
        "same_material": same,
        "wrong": wrong,
        "uncertain": uncertain,
        "judgments": judgments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("current", "matches", "reviewed", "same_material", "wrong", "uncertain")}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--current", required=True, help="NAME passed to dinov3_correspondence.py")
    parser.add_argument("--output", type=Path, help="review JSON; defaults to result-dir/review_NAME.json")
    args = parser.parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    output = args.output or (result_dir / f"review_{args.current.replace(' ', '_')}.json")
    review(result_dir, args.current, output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
