# Frozen RoMa e000 benchmark

The independent entry point is `cloth_agent.experiments.local_node_roma_e000`.
It reads the existing `samples.csv`, manifest, and DINO rejection audit. It checks the
sample-list hash and each source image hash. It never extracts new labels, modifies
robot code, calls a planner, or trains network weights.

## Environment

This evaluated configuration uses Python 3.11, torch 2.11.0+cu128, torchvision
from the existing `molmo` environment, and official `romatch==0.1.2`.
The additional packages were isolated in `/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/roma_deps`: timm 1.0.30,
kornia 0.8.3, kornia-rs 0.2.0, loguru 0.7.3, opencv-python 5.0.0.93,
h5py 3.16.0 and matplotlib 3.11.2. No fused-local-corr extension is used.

Official pretrained weights (not trained on this dataset):

- https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth
- https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

The report config records SHA256 hashes. The downloaded dependencies/weights must be
restored if removed. We use the full RoMa Outdoor model, not TinyRoMa or RoMa v2.

## Run

```bash
PYTHONPATH=/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/roma_deps:$PWD \
/home/sja/miniconda3/envs/molmo/bin/python \
  -m cloth_agent.experiments.local_node_roma_e000 \
  --weights /mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/roma_weights/roma_outdoor.pth \
  --backbone-weights /mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/roma_weights/dinov2_vitl14_pretrain.pth
```

Default output: `reports/local_node_classifier_e000/roma/`.
A GPU-visible environment is required by default. `--device cpu` is supported;
CPU and GPU measurements have separate config/cache identities. CPU VRAM is N/A.

## Frozen scoring and split

Use original fixed crop, with RoMa's default 560 coarse/864 upsample input
resolution. Each symmetric dense pair produces mean confidence over all workspace
pixels, with invalid matches contributing zero. The class score is the mean of
pair scores against **all** eligible S0 or S1 training representatives. No representative
is selected by test performance. Similarity columns are scores, not cosine values.

All cross-run query/known-bank pairs can be cached independently of labels or
thresholds. The outer test run is excluded from the bank and all calibration.
For inner calibration, also exclude the calibration query's whole run.
Call the existing `calibrate(..., 'similarity_and_margin')` and `metrics()` without
modification. No score/threshold tuning using outer test results is permitted.

High-confidence ratio (>=0.5), valid ratio, valid-confidence mean/median and
16x16 occupied-cell workspace coverage in both images are diagnostic columns.
The ROI includes table background: it is not a garment segmentation mask.

## Audit files

- `config.json`: fixed choices, dataset/model hashes.
- `pair_cache/`: image-content/config-keyed pair measurements; enables resume.
- `pair_scores.csv`: dense matching scores, per-pair latency/peak allocated VRAM.
- `fold_audit.json`: exact outer train/test and inner representative IDs/thresholds.
- `predictions.csv`: SOURCE/TARGET/UNKNOWN in `output`; S0/S1 in `prediction` for
  compatibility with existing metrics; includes both bank scores and margin.
- `report.json`, `report.md`: unchanged metrics compared with DINO CLS+margin.
- `hard_negative_visualization.png`: existing S2 false accept plus highest-scoring
  training S0/S1 examples; representative selection is for display only.

Dense confidence is a correspondence measure, not calibrated evidence of cloth
state equivalence. A failure to reject S2 is a valid negative result. This module
is not connected to Graph Replay and does not authorize robot execution.
