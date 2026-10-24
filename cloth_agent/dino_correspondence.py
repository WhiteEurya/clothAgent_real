"""Frozen DINOv2 point correspondence. No camera, robot, planner or network calls.

Public coordinates are continuous image-edge coordinates: pixel (i,j) has centre
(i+.5,j+.5). Labels are consumed only by evaluate(), never by match().
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE, PATCH, GRID, DIM = 448, 14, 32, 384
DEFAULT_MODEL = Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small')
PREPROCESS = dict(size=SIZE, patch=PATCH, resize='PIL bicubic; round; preserve aspect',
                  padding='centred, zero in normalized space',
                  mean=[.485, .456, .406], std=[.229, .224, .225],
                  coordinates='continuous image edges; pixel index + 0.5',
                  version=1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def image_metadata(size):
    w, h = size
    if w < 1 or h < 1:
        raise ValueError('Image dimensions must be positive')
    s = SIZE / max(w, h)
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    left, top = (SIZE - rw) // 2, (SIZE - rh) // 2
    return dict(original_wh=[w, h], resized_wh=[rw, rh],
                scale_xy=[rw / w, rh / h], offset_xy=[left, top],
                padding_ltrb=[left, top, SIZE-rw-left, SIZE-rh-top])


def original_to_input(point, metadata):
    return np.asarray(point, dtype=float) * metadata['scale_xy'] + metadata['offset_xy']


def input_to_original(point, metadata):
    return (np.asarray(point, dtype=float) - metadata['offset_xy']) / metadata['scale_xy']


def preprocess(image):
    image = image.convert('RGB')
    meta = image_metadata(image.size)
    resized = np.asarray(image.resize(tuple(meta['resized_wh']), Image.Resampling.BICUBIC),
                         dtype=np.float32) / 255
    resized = (resized - np.asarray(PREPROCESS['mean'], np.float32)) / np.asarray(PREPROCESS['std'], np.float32)
    tensor = np.zeros((SIZE, SIZE, 3), np.float32)
    x, y = meta['offset_xy']
    w, h = meta['resized_wh']
    tensor[y:y+h, x:x+w] = resized
    return tensor.transpose(2, 0, 1).copy(), meta


def valid_patches(meta):
    """Keep centres inside content; excludes pure padding and off-image outputs.

    Partially padded patches whose centres are inside remain valid. In a <14px
    thin image no centre may exist: the caller must report NO_RESULT.
    """
    yy, xx = np.mgrid[:GRID, :GRID]
    x, y = (xx + .5) * PATCH, (yy + .5) * PATCH
    left, top = meta['offset_xy']
    w, h = meta['resized_wh']
    return ((x >= left) & (x < left+w) & (y >= top) & (y < top+h)).ravel()


def point_patch(point, meta):
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p >= meta['original_wh']):
        raise ValueError('Source point must be inside original A')
    col, row = np.floor(original_to_input(p, meta) / PATCH).astype(int)
    return int(row * GRID + col)


def patch_point(index, meta):
    row, col = divmod(index, GRID)
    return input_to_original([(col+.5)*PATCH, (row+.5)*PATCH], meta).tolist()


def separated_peaks(scores, radius=2., count=3):
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('NMS radius must be nonnegative and finite')
    values = np.asarray(scores, dtype=np.float32).copy()
    yy, xx = np.mgrid[:GRID, :GRID]
    out = []
    for _ in range(count):
        index = int(values.argmax())
        if not np.isfinite(values[index]):
            break
        out.append(index)
        row, col = divmod(index, GRID)
        values[((yy-row)**2 + (xx-col)**2 <= radius**2).ravel()] = -np.inf
    return out


def match(features_a, features_b, meta_a, meta_b, point, radius=2.):
    """Exhaustive single-patch cosine search, independent of evaluation labels."""
    started = time.perf_counter()
    for features in (features_a, features_b):
        if features.shape != (GRID*GRID, DIM) or not np.isfinite(features).all():
            raise ValueError('Expected finite 1024 x 384 patch features')
    source = point_patch(point, meta_a)
    valid_a, valid_b = valid_patches(meta_a), valid_patches(meta_b)
    result = dict(source_xy=list(map(float, point)), source_index=source,
                  source_grid_rc=list(divmod(source, GRID)), candidates=[],
                  top1_score=None, independent_peak_gap=None, candidate_gaps=[],
                  cycle_error_patches=None, back_xy=None, back_index=None,
                  patch_wh_original_a=(PATCH/np.asarray(meta_a['scale_xy'])).tolist(),
                  patch_wh_original_b=(PATCH/np.asarray(meta_b['scale_xy'])).tolist(),
                  nms_radius_patches=radius, nms_metric='Euclidean; suppress distance <= radius',
                  search_mask='patch centre inside content only; no foreground or label mask',
                  reliability='UNCALIBRATED_DIAGNOSTICS_ONLY')
    scores = np.full(GRID*GRID, -np.inf, np.float32)
    if not valid_a[source] or not valid_b.any():
        result['status'] = 'NO_VALID_SOURCE_PATCH' if not valid_a[source] else 'NO_VALID_TARGET_PATCH'
    else:
        # einsum avoids large BLAS thread-pool startup costs on each UI click.
        scores = np.clip(np.einsum('ij,j->i', features_b, features_a[source]), -1, 1)
        scores[~valid_b] = -np.inf
        indices = separated_peaks(scores, radius)
        for rank, index in enumerate(indices, 1):
            result['candidates'].append(dict(rank=rank, index=index,
                grid_rc=list(divmod(index, GRID)), xy=patch_point(index, meta_b), score=float(scores[index])))
        result['top1_score'] = float(scores[indices[0]])
        result['candidate_gaps'] = [float(scores[indices[0]]-scores[i]) for i in indices[1:]]
        if len(indices) > 1:
            result['independent_peak_gap'] = result['candidate_gaps'][0]
        reverse = np.einsum('ij,j->i', features_a, features_b[indices[0]])
        reverse[~valid_a] = -np.inf
        back = int(reverse.argmax())
        result.update(status='OK', back_index=back, back_xy=patch_point(back, meta_a),
            cycle_error_patches=float(np.linalg.norm(np.subtract(divmod(back, GRID), divmod(source, GRID)))))
    result['matching_s'] = time.perf_counter() - started
    return result, scores.reshape(GRID, GRID)


def polygon_distance(point, polygon):
    """Euclidean original-pixel distance to a closed admissible region (0 inside)."""
    p = np.asarray(point, float)
    poly = np.asarray(polygon, float)
    a, b = poly, np.roll(poly, -1, axis=0)
    d = b-a
    t = np.clip(np.einsum('ij,ij->i', p-a, d) / np.maximum(np.sum(d*d, axis=1), 1e-20), 0, 1)
    distance = float(np.min(np.linalg.norm(p-(a+t[:, None]*d), axis=1)))
    if distance < 1e-7:
        return 0.
    inside = False
    for (x1, y1), (x2, y2) in zip(a, b):
        if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2-x1)*(p[1]-y1)/(y2-y1)+x1:
            inside = not inside
    return 0. if inside else distance


def validate_label(label, size):
    visibility = label.get('visibility', 'unlabeled')
    if visibility not in ('visible', 'invisible', 'uncertain', 'unlabeled'):
        raise ValueError('Unknown visibility label')
    polygons = label.get('regions', [])
    target = label.get('target_xy')
    if visibility == 'visible' and not polygons and target is None:
        raise ValueError('Visible labels need a corresponding region or exact target point')
    if visibility != 'visible' and polygons:
        raise ValueError('Only visible labels can have corresponding regions')
    if target is not None:
        p = np.asarray(target, dtype=float)
        if visibility != 'visible' or p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError('Exact target point requires a visible label and finite XY')
        if np.any(p < 0) or np.any(p >= size):
            raise ValueError('Exact target point is outside B')
    for poly in polygons:
        v = np.asarray(poly, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3 or not np.isfinite(v).all():
            raise ValueError('Region must contain at least three finite XY vertices')
        if np.any(v < 0) or np.any(v > size):
            raise ValueError('Region is outside B')
        area = abs(np.sum(v[:, 0]*np.roll(v[:, 1], -1)-v[:, 1]*np.roll(v[:, 0], -1))) / 2
        if area < 1e-6:
            raise ValueError('Region must have nonzero area')


def evaluate(result, label, size):
    validate_label(label, size)
    out = dict(visibility=label.get('visibility', 'unlabeled'), top1_hit=None,
               top3_hit=None, top1_region_error_px=None, candidate_region_errors_px=[],
               top1_point_error_px=None, candidate_point_errors_px=[],
               eligible_for_blind_metrics=bool(label.get('before_prediction', False)))
    if out['visibility'] == 'visible' and label.get('regions'):
        out.update(top1_hit=False, top3_hit=False)
    if out['visibility'] == 'visible' and result['candidates'] and label.get('regions'):
        distances = [min(polygon_distance(c['xy'], poly) for poly in label['regions'])
                     for c in result['candidates']]
        out.update(top1_hit=distances[0] == 0, top3_hit=min(distances) == 0,
                   top1_region_error_px=distances[0], candidate_region_errors_px=distances)
    if out['visibility'] == 'visible' and result['candidates'] and label.get('target_xy') is not None:
        distances = [float(np.linalg.norm(np.asarray(c['xy'])-label['target_xy']))
                     for c in result['candidates']]
        out.update(top1_point_error_px=distances[0], candidate_point_errors_px=distances)
    return out


def heatmap_overlay(image, scores, meta, opacity=.48):
    """Fixed cosine scale [-1,1]. Upsampling is display-only, not localization."""
    # Upsample values and validity separately to avoid painting padding as data.
    finite = np.isfinite(scores)
    numerator = Image.fromarray(np.where(finite, scores, 0).astype(np.float32)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    weight = Image.fromarray(finite.astype(np.float32)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x, y = meta['offset_xy']; w, h = meta['resized_wh']
    box = (x, y, x+w, y+h)
    num = np.asarray(numerator.crop(box).resize(image.size, Image.Resampling.BILINEAR))
    den = np.asarray(weight.crop(box).resize(image.size, Image.Resampling.BILINEAR))
    values = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    colors = cosine_colors(values)
    alpha = (opacity*(den > 1e-6))[..., None]
    return Image.fromarray(np.uint8(np.clip(np.asarray(image.convert('RGB'))*(1-alpha) + colors*alpha, 0, 255)))


def cosine_colors(values):
    """Blue (-1), pale (0), red (+1), identical for every pair."""
    values = np.clip(np.asarray(values), -1, 1)
    low, mid, high = np.array([35, 70, 190]), np.array([245, 245, 235]), np.array([215, 35, 35])
    return np.where((values < 0)[..., None], mid + (-values)[..., None]*(low-mid),
                    mid + values[..., None]*(high-mid))


def draw_result(image, result, label=None, source=False):
    im = image.copy().convert('RGB'); draw = ImageDraw.Draw(im)
    r = max(6, min(im.size)/75)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(18, min(im.size)//28))
    except OSError:
        font = ImageFont.load_default()
    if source:
        points = [('A', result['source_xy'], '#ffce45')]
        if result.get('back_xy'):
            points.append(('R', result['back_xy'], '#54ffda'))
    else:
        points = [(str(c['rank']), c['xy'], ['#ffce45', '#62ffce', '#dc9aff'][c['rank']-1]) for c in result['candidates']]
        if (label or {}).get('target_xy') is not None:
            points.append(('GT', label['target_xy'], '#55ff55'))
        for poly in (label or {}).get('regions', []):
            draw.line([tuple(p) for p in poly]+[tuple(poly[0])], fill='#55ff55', width=3)
    for name, (x, y), color in points:
        draw.ellipse((x-r, y-r, x+r, y+r), outline='black', width=5)
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=2)
        draw.text((x+r+3, y-r), name, font=font, fill=color, stroke_width=2, stroke_fill='black')
    return im


class FrozenDino:
    """Local HF DINOv2-S/14: final layernorm output excluding CLS, L2 normalized.

    HF last_hidden_state[:,1:] is the patch-only counterpart of official
    forward_features()['x_norm_patchtokens']. No default centre-crop processor.
    """
    def __init__(self, path=DEFAULT_MODEL, device='auto', threads=4):
        started = time.perf_counter()
        import torch
        import transformers
        from transformers import AutoModel
        self.torch = torch
        torch.set_num_threads(max(1, threads))
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        path = Path(path).resolve()
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False).to(self.device)
        cfg = self.model.config
        if (cfg.model_type, cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) != ('dinov2', 14, 384, 12, 6):
            raise ValueError('Requires pretrained DINOv2 ViT-S/14, 384-dimensional patch tokens')
        self.contract = dict(model='DINOv2 ViT-S/14', weights={p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()},
                             preprocess=PREPROCESS, tokens='final layernorm patch tokens, CLS removed, per-vector L2',
                             torch=torch.__version__, transformers=transformers.__version__, device=self.device, dtype='float32')
        self.sync()
        self.load_s = time.perf_counter()-started
        self.calls = 0

    def sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def encode(self, images):
        t = time.perf_counter()
        prepared = [preprocess(im) for im in images]
        batch = self.torch.from_numpy(np.stack([x[0] for x in prepared]))
        prep_s = time.perf_counter()-t
        self.sync(); t = time.perf_counter()
        with self.torch.inference_mode():
            tokens = self.model(pixel_values=batch.to(self.device)).last_hidden_state[:, 1:].float()
            tokens = self.torch.nn.functional.normalize(tokens, p=2, dim=-1)
            tokens = tokens.cpu().numpy()
        self.sync()
        if tokens.shape != (len(images), GRID*GRID, DIM):
            raise ValueError(f'Unexpected patch token shape {tokens.shape}')
        self.calls += 1
        return tokens, dict(preprocess_s=prep_s, forward_and_transfer_s=time.perf_counter()-t,
                            newly_encoded_images=len(images), forward_calls=1)


class FeatureStore:
    def __init__(self, encoder, directory):
        self.encoder = encoder
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.memory = {}

    def get_pair(self, paths):
        started = time.perf_counter()
        items, missing, hits = [], {}, []
        for path in paths:
            path = Path(path).resolve()
            with Image.open(path) as im:
                image = im.convert('RGB')  # No EXIF transpose; graph coordinates refer to raw RGB.
            pixels = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            identity = dict(rgb_pixels_sha256=pixels, encoder=self.encoder.contract)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            item = dict(path=str(path), sha256=sha(path), rgb_pixels_sha256=pixels,
                        image=image, meta=image_metadata(image.size), cache_key=key)
            items.append(item)
            if key in self.memory:
                hits.append('memory')
            elif (self.directory/(key+'.npz')).exists():
                with np.load(self.directory/(key+'.npz'), allow_pickle=False) as data:
                    features = data['features']
                    if str(data['identity'].item()) != json.dumps(identity, sort_keys=True):
                        raise ValueError('Feature cache identity mismatch')
                self._validate(features)
                self.memory[key] = features
                hits.append('disk')
            else:
                missing.setdefault(key, (image, identity))
                hits.append('new')
        timing = dict(preprocess_s=0., forward_and_transfer_s=0., newly_encoded_images=0, forward_calls=0)
        if missing:
            vectors, timing = self.encoder.encode([v[0] for v in missing.values()])
            for (key, (_, identity)), features in zip(missing.items(), vectors):
                self._validate(features)
                target = self.directory/(key+'.npz')
                with target.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f, features=features, identity=json.dumps(identity, sort_keys=True))
                target.with_suffix('.tmp').replace(target)
                self.memory[key] = features
        for item in items:
            item['features'] = self.memory[item['cache_key']]
        timing.update(pair_load_s=time.perf_counter()-started, cache_sources=hits,
                      model_load_s=self.encoder.load_s, device=self.encoder.device)
        return items, timing

    @staticmethod
    def _validate(features):
        if features.shape != (GRID*GRID, DIM) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError('Invalid feature cache')
        if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
            raise ValueError('Feature vectors must be L2 normalized')
