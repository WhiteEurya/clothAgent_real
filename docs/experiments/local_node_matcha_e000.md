# MATCHA e000 backend

Status: implemented aggregation, run-held-out evaluation, cache, and GPU timing
runner. **Real model inference is unverified and blocked by missing MATCHA fusion
weights.** No synthetic/random-weight result is reported as pretrained MATCHA.

## Fixed protocol

Use the original samples.csv (SHA256 checked), labels and all13 baseline folds.
Both fixed methods use the official full MATCHA semantic descriptor at512x512,
default diffusion ensemble8, fixed seed0, original workspace crop:

- `global_mean`: mean dense feature map, then L2 normalize.
- `spatial_8x8`: mean in each of64 fixed cells, normalize each cell, average
  corresponding-cell cosine. This retains layout, not correspondence count.

Class score is average cosine against all S0/S1 training representatives.
The existing RoMa module's **generic bank-evaluation function only** is reused;
no RoMa model is imported/invoked. Its `calibrate`/`metrics` are the original
DINO rejection protocol. Outer test runs are absent from banks/calibration;
inner calibration queries exclude their own entire run too. Both fixed methods
are reported; do not tune/select using outer test performance.

## Storage and environment

All models, source downloads, isolated dependencies, feature maps and caches:
`/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/`.

- `matcha_source`: official `nv-dvl/matcha` shallow clone.
- `matcha_deps`, `roma_deps`: isolated packages with existing molmo torch.
- `matcha_sd21`: original SD2.1 mirror (`sd2-community/stable-diffusion-2-1`);
  download complete; all three safetensors files verified against upstream LFS SHA256.
  Checksums/revision are in `matcha_sd21/verified_weights.json`.
- `matcha_torch/hub/checkpoints`: symlink to the already downloaded DINOv2-L.
- `matcha_pretrained.pth`: **missing**.

Model initialization uses official `Matcha` directly (same core as
`MatchaFeature.describe`), avoiding unused sparse keypoint detectors. Routes
SD2.1 to local files and replaces optional xformers attention with PyTorch SDPA.
Requires pretrained learned fusion keys; fails if those keys are missing. These
compatibility changes still require a real forward test once weights exist.

## Run after weights are available

```bash
/home/sja/miniconda3/envs/molmo/bin/python \
  -m cloth_agent.experiments.local_node_matcha_e000
```

Optional `--weights /mnt/newssd/.../matcha_pretrained.pth`, `--assets`, `--input`,
`--output`. Defaults preserve the requested report location:
`reports/local_node_classifier_e000/matcha/`.

Without complete weights returns code2, `status.json` and an explicit blocked
report. It does not produce placeholder predictions or invented latency.

After success: predictions.csv, feature_latency.csv, query_latency.csv,
fold_audit.json, report.json/report.md. Dense feature caches are on newssd and
keyed by image/model/config hashes. Feature latency is synchronized GPU forward
with an already prepared image. Query timing separately measures fresh RGB
preprocessing + H2D + full forward + aggregation + cached representative bank
comparison + gate. Model load, bank precomputation and disk image decoding are
excluded. Includes measured allocated peak VRAM. No training/GPT/robot API.

## Weight search (2026-10-13)

- Current official Drive ID `17p0-ne4B9H60cC_AF6G284SetuImDUDO`: HTTP404.
- Original README before commit4277633: Drive ID
  `1-ItlDnwXlIDS7rRNr9IN6Mig5QvvSEHZ`: HTTP401 / not publicly downloadable.
- [Official issue2](https://github.com/nv-dvl/matcha/issues/2) and
  [issue3](https://github.com/nv-dvl/matcha/issues/3) report unavailable weights;
  no replies at inspection time.
- Official forks retain the same links; checked trees contain sparse detector
  weights, not MATCHA fusion checkpoint.
- HF `raymondt/matcha` has `matcha.zip` but is **manually gated**; not downloaded,
  contents and applicability unverified. Its dataset repo has no weights.
- Other similarly named HF models checked are text/TTS/diffusion/robot datasets,
  not this CVPR2025 dense matcher. They are not substitutes.

Search evidence JSON is stored beside model assets. No authorization request
was submitted to a model owner and no access controls were bypassed.

## Tests

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest -q \
  tests/test_local_node_classifier_e000.py \
  tests/test_local_node_rejection_e000.py \
  tests/test_local_node_roma_e000.py \
  tests/test_local_node_matcha_e000.py
```
