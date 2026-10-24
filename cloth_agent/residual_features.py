"""Frozen visual/action features. Run the encoder in the existing DINO environment."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from PIL import Image

from . import dino_correspondence as dino

VERSION = 'dino_s14_action_projection32_v1'
DIM = 238
PYTHON = Path('/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python')


def action_features(tokens, size, rows):
    """No labels, candidate IDs, rank or overlaid pixels enter the features."""
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.shape != (1024, 384) or not np.isfinite(tokens).all():
        raise ValueError('invalid DINO tokens')
    meta = dino.image_metadata(size)
    valid = dino.valid_patches(meta).reshape(32, 32)
    projection = np.random.default_rng(20261022).normal(size=(384, 32)) / np.sqrt(32)

    def project(v):
        v = v @ projection
        return v / max(np.linalg.norm(v), 1e-12)

    def local(point):
        row, col = divmod(dino.point_patch(point, meta), 32)
        ys, xs = slice(max(0, row-1), min(32, row+2)), slice(max(0, col-1), min(32, col+2))
        patch = tokens.reshape(32, 32, 384)[ys, xs][valid[ys, xs]]
        if not len(patch):
            raise ValueError('grasp/target has no content patches')
        return project(patch.mean(axis=0))

    scene = project(tokens[valid.ravel()].mean(axis=0))
    result = {}
    for r in rows:
        pa, pb = np.asarray(r['grasp_px'], float), np.asarray(r['target_px'], float)
        delta = pb-pa
        length = np.linalg.norm(delta)
        if length <= 0:
            raise ValueError('zero length action')
        ux, uy = delta/length
        a, b = local(pa), local(pb)
        path = np.mean([local(pa+delta*t) for t in (.25, .5, .75)], axis=0)
        angle = r['grasp_angle_deg']
        radians = 0 if angle is None else np.deg2rad(2*angle)
        ps, pt = r['Ps']/100, r['Pt']/100
        basic = [ps, pt, *(2*pa/size-1), *(2*pb/size-1), *(delta/size),
                 length/np.linalg.norm(size), np.cos(radians) if angle is not None else 0,
                 np.sin(radians) if angle is not None else 0, float(angle is not None), ps*pt, ps-pt]
        x = np.concatenate([basic, a, b-a, path-a, a*ux, a*uy, scene*ux, scene*uy])
        if x.shape != (DIM,) or not np.isfinite(x).all():
            raise ValueError('invalid residual features')
        result[r['id']] = x.tolist()
    return result


def extract(image, rows, folder):
    """One local encoding, no downloads; latency is separate from GPT and training."""
    import subprocess
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    (folder/'candidates.json').write_text(json.dumps(rows))
    started = time.monotonic()
    with (folder/'encoder.log').open('w') as log:
        subprocess.run([str(PYTHON), '-B', '-m', 'cloth_agent.residual_features',
                        '--image', str(Path(image).resolve()), '--folder', str(folder.resolve())],
                       cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log,
                       check=True, timeout=120)
    result = json.loads((folder/'features.json').read_text())
    result['wall_s'] = time.monotonic()-started
    (folder/'features.json').write_text(json.dumps(result, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.folder/'candidates.json').read_text())
    encoder = dino.FrozenDino(device='cpu')
    image = Image.open(args.image).convert('RGB')
    tokens, timing = encoder.encode([image])
    np.save(args.folder/'patch_tokens.npy', tokens[0], allow_pickle=False)
    # Device/library versions are evidence, not the identity of learned inputs.
    identity = dict(version=VERSION, dimension=DIM, weights=encoder.contract['weights'],
                    preprocess=encoder.contract['preprocess'])
    dino.save_json(args.folder/'features.json', dict(identity=identity, image_sha256=dino.sha(args.image),
        features=action_features(tokens[0], np.array(image.size), rows),
        encoder_load_s=encoder.load_s, encoder=encoder.contract, timing=timing))


if __name__ == '__main__':
    main()
