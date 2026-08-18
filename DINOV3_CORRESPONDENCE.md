# Standalone frozen DINOv3 garment correspondence experiment

This experiment is intentionally outside the robot/Claude pipeline. It uses
one flat reference image, manually clicked reference pixels, and one or more
images of the same garment after a single fold. The model is frozen; the only
learned operation is a cosine-similarity lookup over patch tokens.

## 1. Annotate 30 reference points

Run this in the calibrated camera environment with OpenCV GUI support. When
`--image` is omitted, every launch captures a fresh Camera A RGB-D observation
using the existing perception configuration before opening the point window:

```bash
python scripts/dinov3_annotate_points.py \
  --output /path/to/flat_reference_points.json \
  --num-points 30
```

The timestamped RGB, depth, calibrated base-XYZ map, valid-XYZ mask, and camera
metadata are saved beside the points JSON. To relabel an existing image without
touching a camera, retain the offline mode explicitly:

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
  --input-width 448 \
  --input-height 336 \
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

For the calibrated 640x480 cameras, images use an aspect-preserving 448x336
input. DINOv2-small therefore produces a 24x32 token grid instead of a
geometry-distorting 32x32 square grid. Reference coordinates are mapped to the
nearest DINO patch center. No robot pose, semantic prompt, or Claude call is
used.

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

The official Facebook DINOv3 Hugging Face repositories are gated. Request
access and authenticate with Hugging Face before omitting `--local-files-only`;
the scripts do not embed or print an access token.

## Canonical area graph prototype

`cloth_agent/canonical_area_graph.py` extends the existing annotated points
without changing robot perception or policy. A flat-reference garment mask is
partitioned by constrained/geodesic Voronoi paths that cannot leave the cloth.
Each seed therefore owns an explicit material-region mask plus region features,
while matching V1 uses a seed-centered local token set clipped to that region.

Build a reference graph and feature bank:

```bash
python scripts/canonical_area_graph.py build \
  --reference /path/to/flat_reference.png \
  --points /path/to/flat_reference_points.json \
  --reference-mask /path/to/flat_reference_garment_mask.png \
  --neighbors /path/to/canonical_neighbors.json \
  --sample-radius 1 \
  --output results/canonical_graph/reference_01 \
  --local-files-only
```

The optional neighbors JSON is an explicit `area_id -> [neighbor_area_id]`
mapping. If it is omitted, the builder uses a symmetric reference-image k-NN
graph and records `adjacency_source=symmetric_knn_heuristic`; this is useful for
debugging but is not a material-connectivity ground truth.

For an offline flat reference on a visually distinct table, a reproducible
largest-component mask can be prepared with:

```bash
python scripts/canonical_area_graph.py prepare-reference-mask \
  --image /path/to/flat_reference.png \
  --output /path/to/flat_reference_garment_mask.png
```

The graph archive stores per-side region-label grids, each area's region patch
coordinates, the seed-local matching samples, and all sampled region features.
`canonical_regions.png` visualizes the constrained partition.

For example, point IDs 1–3 become `A01`–`A03`, and an intrinsic chain is:

```json
{
  "A01": ["A02"],
  "A02": ["A03"],
  "A03": []
}
```

The saved graph is `canonical_graph.json`; its compressed feature bank is
`canonical_feature_bank.npz`. To bypass DINOv3 with any other dense extractor,
pass an `(H, W, D)` NumPy array via `--reference-features` at build time and
`--current-features NAME=features.npy` at match time.

Run soft matching and topology refinement:

```bash
python scripts/canonical_area_graph.py match \
  --graph results/canonical_graph/reference_01/canonical_graph.json \
  --reference /path/to/flat_reference.png \
  --current folded=/path/to/folded.png \
  --mask folded=/path/to/folded_garment_mask.png \
  --surface-xyz folded=/path/to/camera_A_base_xyz_mm.npy \
  --ground-truth folded=/path/to/folded_node_ground_truth.json \
  --output results/canonical_graph/reference_01/matches \
  --local-files-only
```

When a saved dense A/B perception result has a fused garment mask but no
per-camera mask PNG, project that existing segmentation back into one RGB-D
view before matching:

```bash
python scripts/canonical_area_graph.py prepare-mask \
  --result /path/to/perception/result.json \
  --camera-label A \
  --output results/canonical_graph/camera_A_garment_mask.png
```

This reuses the current perception implementation's fused-cloud projection,
observed-depth consistency check, and height envelope. It does not recapture a
camera, refit the table, change fusion, or introduce a new garment segmenter.
The command also writes projection diagnostics and a sparse-mask preview next
to the requested PNG.

Ground truth is optional and is only used for evaluation. Run matching once,
read the stable current node IDs (`Xyyy_xxx`) from the visibility JSON, annotate
their canonical identities, and rerun with a file such as:

```json
{
  "X000_000": "A01",
  "X000_001": "A02",
  "X000_002": "A03"
}
```

To create that sparse evaluation file without typing node IDs, use the
side-by-side human annotation tool:

```bash
python scripts/canonical_area_annotate.py \
  --graph results/canonical_graph/reference_01/canonical_graph.json \
  --visibility results/canonical_graph/reference_01/matches/folded_visibility.json \
  --reference /path/to/flat_reference.png \
  --current /path/to/folded.png \
  --output results/canonical_graph/reference_01/folded_ground_truth.json
```

For deformed garments, add `--reference-first`: click a known canonical area
first, then click its actual current-image location. The current click snaps
internally to the nearest valid feature patch, so the image is not covered by
a patch grid. Right-click a current assignment to remove it; `u` undoes, `s`
saves, and `q` saves and closes. Existing compatible output is resumed unless
`--overwrite` is used.
Only deliberately selected visible nodes are labeled—the tool is an evaluation
aid and does not ask Claude to perform dense garment annotation. The produced
JSON can be passed directly back through `--ground-truth folded=...`.

Multiple folded observations can be evaluated in one invocation by repeating
the named `--current`, `--mask`, `--surface-xyz`, and `--ground-truth` options.

The implementation maintains two intentionally different graphs:

```text
G_M    fixed canonical material graph from the reference
G_O(t) current visual/depth observation adjacency
```

An edge in `G_O(t)` means only that two current patches appear adjacent or
continuous under the image, mask, feature, and optional calibrated-depth
tests. It never implies an intrinsic material edge in `G_M`.

Every current visible patch retains feature-only and refined top-k ranks. The
refinement uses positive evidence only:

```text
refined_score(area, node)
  = bidirectional Chamfer cosine(reference local set, current local set)
  + topology_lambda * positive compatible-anchor support
```

More exactly, for current node `x` and canonical candidate `a`:

```text
R(a) = reference seed-centered 3x3 token set, clipped to canonical region a
C(x) = current mask-valid 3x3 token set around current node x
feature(x,a) = 0.5 * [mean_r max_c cosine(r,c) + mean_c max_r cosine(c,r)]
compatible(a) = {a} union canonical_neighbors(a)
anchor(y) = high-confidence feature-only canonical identity, or NONE
support(x, a) = confidence-weighted fraction of adjacent anchors in compatible(a)
plausibility(x, a) = clip((feature(x,a)-minimum_similarity)/(1-minimum_similarity), 0, 1)
refined(x, a) = feature(x, a) + lambda * plausibility(x,a) * support(x,a)
```

An ambiguous neighbor contributes no support. A canonically incompatible
neighbor contributes no support and no penalty. Local candidate plausibility
gates the bonus so one confidently wrong anchor cannot promote a visually
implausible candidate from zero evidence. The refinement remains a single,
interpretable pass rather than a GNN.

The raw refined score is intentionally retained for debugging and can exceed
1.0 because it includes the additive topology term. The separately reported
`confidence` is bounded to `[0, 1]` as
`clip(refined_score / (1 + effective_topology_lambda), 0, 1)`, where the
effective lambda is zero for a node with no known current-surface neighbors.
The `--confident-threshold` gate is applied to this bounded value. If several candidates
fall inside the ambiguity margin, all of those canonical areas receive
`AMBIGUOUS` evidence; the runner does not incorrectly call the lower-ranked
hypotheses unobserved.

Current observation edges are conservative: mask-valid four-connected patches need
both local feature continuity and, when supplied, a calibrated 3-D surface gap
below the configured threshold. Rejected pixel-neighbor pairs are saved as
`current_unknown_edges`.

Accepted observation edges are compared with the final canonical beliefs and
reported separately as:

- `MATERIAL_CONSISTENT`: same canonical area or a direct `G_M` edge;
- `UNEXPECTED_ADJACENCY`: confident identities are not canonical neighbors,
  recorded as a possible fold/contact/occlusion boundary without changing any
  identity score;
- `AMBIGUOUS`: either endpoint lacks sufficient canonical identity evidence.

Outputs include:

- per-node feature-only/refined top-k candidates and ambiguity;
- a source-image crop for every ambiguous node, plus its bounding box,
  canonical candidate neighbors, and already matched surrounding areas;
- canonical `VISIBLE`, `AMBIGUOUS`, and `UNOBSERVED` states;
- visible-hidden/ambiguous canonical frontiers;
- feature-only versus feature+topology Top-1, Recall@3/5, ambiguous rate, and
  wrong-high-confidence rate when ground truth is supplied;
- counts of topology corrections and topology-induced errors;
- `G_O(t)` versus `G_M` relation records and counts for material-consistent,
  unexpected, and ambiguous observation adjacencies;
- a visualization with green visible matches, yellow ambiguity, red frontier
  targets/edges, and gray unobserved areas.

`ENGINEERING_ISSUE:` The current reference representation is a list of sparse
manually annotated points, not a verified intrinsic material-area partition.
The prototype therefore makes the smallest extension—each point owns a local
patch-token neighborhood—and supports explicit adjacency JSON. A future true
surface partition should replace only the area construction input; the feature
bank, soft matcher, graph refinement, evaluation, and visualization APIs do
not depend on DINOv3 or on point annotations.

`ENGINEERING_ISSUE:` Current observation adjacency is not recovered material
connectivity. Mask support, local feature continuity, and calibrated XYZ gaps
reject many unsafe pixel edges, but two cloth layers touching in 3-D can still
look continuous. This is now an intended observable: canonically incompatible
confident endpoints are emitted as `UNEXPECTED_ADJACENCY`, not silently
converted into a material edge and not used as negative matching evidence.
