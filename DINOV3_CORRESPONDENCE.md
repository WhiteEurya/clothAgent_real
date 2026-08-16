# Standalone frozen DINOv3 garment correspondence experiment

This experiment is intentionally outside the robot/Claude pipeline. It uses
one flat reference image, manually clicked reference pixels, and one or more
images of the same garment after a single fold. The model is frozen; the only
learned operation is a cosine-similarity lookup over patch tokens.

## 1. Annotate 30 reference points

Run this in an environment with OpenCV GUI support:

```bash
python scripts/dinov3_annotate_points.py \
  --image /path/to/flat_reference.png \
  --output /path/to/flat_reference_points.json \
  --num-points 30
```

The default click order has 5 points on each sleeve, 10 on the torso, 5 on
the hem, and 5 around the collar. The window shows the label expected for the
next click. Use `u` to undo, then press Enter when all 30 points are present.
You can supply your own comma-separated labels with `--labels`.

## 2. Run frozen DINOv3 correspondence

The default model is the Hugging Face DINOv3 ViT-B/16 checkpoint. If the
checkpoint is gated or stored locally, pass the local directory instead of the
model ID and use `--local-files-only`.

```bash
python scripts/dinov3_correspondence.py \
  --reference /path/to/flat_reference.png \
  --points /path/to/flat_reference_points.json \
  --current single_fold_01=/path/to/single_fold_01.png \
  --current single_fold_02=/path/to/single_fold_02.png \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --input-size 448 \
  --output /path/to/dinov3_results/single_fold_set_01
```

For each current image the output contains:

- `current_<name>_matches.png`: every top-1 current match, numbered with the
  source point ID;
- `current_<name>_correspondence_side_by_side.png`: reference/current images
  with corresponding numbered points joined by lines;
- `current_<name>_similarity_heatmaps.png`: a contact sheet of all point-wise
  cosine-similarity maps;
- `current_<name>_point_XX_heatmap.png`: one heatmap per reference point;
- `current_<name>_matches.json` and `.csv`: source coordinate, top-1 target
  coordinate, patch coordinates, and cosine similarity;
- `current_<name>_similarity.npy`: raw `[num_points, grid_y, grid_x]` maps.

The images are resized to the same square input for both states, and the
reference coordinates are mapped to the nearest DINO patch center. No mask,
depth, robot pose, semantic prompt, or Claude call is used.

## 3. Record the visual count

To count how many top-1 matches really land on the same material region, review
each point manually:

```bash
python scripts/dinov3_review.py \
  --result-dir /path/to/dinov3_results/single_fold_set_01 \
  --current single_fold_01
```

Press `y` for same material region, `n` for wrong region, `u` for uncertain,
or `q` to save early. The resulting JSON reports `same_material`, `wrong`, and
`uncertain`; there is no automatic semantic correctness claim.

## Model/runtime note

The script uses `transformers.AutoImageProcessor` and `AutoModel`, calls
`eval()` and `requires_grad_(False)`, and runs under `torch.inference_mode()`.
It does not fall back to DINOv2 or a supervised model: if the requested DINOv3
checkpoint cannot be loaded, it fails with the loading error so the experiment
does not silently change models.
