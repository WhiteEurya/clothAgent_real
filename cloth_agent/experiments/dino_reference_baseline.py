"""Frozen DINO reference-action retrieval; no robot connection or action execution.

prepare caches reference CLS, patches and geometry. query encodes a saved current
RGB exactly once. Reference images are each edge's actual demonstration pre-image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from cloth_agent.dino_correspondence import (
    DEFAULT_MODEL, DIM, GRID, FrozenDino, preprocess, save_json, sha, valid_patches,
)


class JointEncoder(FrozenDino):
    def __init__(self, model=DEFAULT_MODEL, device='cpu'):
        super().__init__(model, device=device)
        self.contract['tokens'] = 'final layernorm CLS and patches; per-vector L2'

    def encode_joint(self, image):
        tensor, geometry = preprocess(image)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(pixel_values=self.torch.from_numpy(tensor[None]).to(self.device))
            tokens = self.torch.nn.functional.normalize(outputs.last_hidden_state.float(), dim=-1)
            tokens = tokens[0].cpu().numpy()
        self.sync()
        self.calls += 1
        if tokens.shape != (GRID * GRID + 1, DIM) or not np.isfinite(tokens).all():
            raise ValueError('Invalid DINO output')
        return tokens[0], tokens[1:], geometry, time.perf_counter() - started


def references(graph_path):
    graph_path = Path(graph_path).resolve()
    graph = json.loads(graph_path.read_text())
    trajectory_path = Path(graph['source_trajectory']['path'])
    if sha(trajectory_path) != graph['source_trajectory']['sha256']:
        raise ValueError('Trajectory changed since graph creation')
    trajectory = json.loads(trajectory_path.read_text())
    steps = {s['step_id']: s for s in trajectory['steps']}
    nodes = {n['node_id']: n for n in graph['nodes']}
    refs = []
    for edge in sorted(graph['edges'], key=lambda e: e['edge_id']):
        step = steps[edge['seed_step_id']]
        observation_id = step['before_observation']
        if observation_id not in nodes[edge['source_node']]['representative_observations']:
            raise ValueError(f"Pre-image not in source node: {edge['edge_id']}")
        rgb = trajectory['observations'][observation_id]['rgb']
        if sha(rgb['path']) != rgb['sha256']:
            raise ValueError(f"Reference image changed: {rgb['path']}")
        refs.append(dict(node_id=edge['source_node'], edge_id=edge['edge_id'],
                         step_id=step['step_id'], observation_id=observation_id,
                         image=rgb, action_family=edge['action_family'], edge=edge))
    if not refs:
        raise ValueError('No reference action pre-images')
    return refs, dict(graph=dict(path=str(graph_path), sha256=sha(graph_path)),
                      trajectory=dict(path=str(trajectory_path), sha256=sha(trajectory_path)))


def rank_nodes(refs, features, query):
    """Exact cosine ranking; max over edge pre-images; deterministic tie order."""
    features, query = np.asarray(features), np.asarray(query)
    if features.shape != (len(refs), DIM) or query.shape != (DIM,):
        raise ValueError('Invalid CLS dimensions')
    if not np.isfinite(features).all() or not np.isfinite(query).all():
        raise ValueError('Nonfinite CLS')
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5) or not np.isclose(np.linalg.norm(query), 1, atol=1e-5):
        raise ValueError('CLS must be L2 normalized')
    scores = np.clip(np.einsum('ij,j->i', features, query), -1, 1)
    order = sorted(range(len(refs)), key=lambda i: (-float(scores[i]), refs[i]['node_id'], refs[i]['edge_id']))
    ranked, seen = [], set()
    for i in order:
        if refs[i]['node_id'] in seen:
            continue
        seen.add(refs[i]['node_id'])
        ranked.append(dict(rank=len(ranked)+1, score=float(scores[i]), reference_index=i,
                           **refs[i]))
    return ranked


def prepare(graph, output, model=DEFAULT_MODEL, device='cpu'):
    refs, provenance = references(graph)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'features').mkdir()
    (output / 'images').mkdir()
    encoder = JointEncoder(model, device)
    cached, cls_rows = {}, []
    for i, ref in enumerate(refs):
        digest = ref['image']['sha256']
        if digest not in cached:
            image_path = output / 'images' / f'{digest}.png'
            shutil.copyfile(ref['image']['path'], image_path)
            if sha(image_path) != digest:
                raise ValueError('Image changed while copying')
            with Image.open(image_path) as im:
                cls, patches, geometry, seconds = encoder.encode_joint(im.convert('RGB'))
            feature_path = output / 'features' / f'{digest}.npz'
            np.savez_compressed(feature_path, cls=cls, patches=patches, valid=valid_patches(geometry))
            cached[digest] = dict(cls=cls, geometry=geometry, forward_s=seconds,
                                  features=str(feature_path.relative_to(output)),
                                  features_sha256=sha(feature_path), image=str(image_path.relative_to(output)))
        entry = cached[digest]
        # Preserve the source path separately; cached RGB is authoritative online.
        ref['source_image'] = ref['image']
        ref.update({k: v for k, v in entry.items() if k != 'cls'})
        cls_rows.append(entry['cls'])
        if (i+1) % 20 == 0 or i+1 == len(refs):
            print(f'References: {i+1}/{len(refs)}', flush=True)
    np.save(output / 'cls.npy', np.stack(cls_rows))
    manifest = dict(version=1, **provenance, contract=encoder.contract, references=refs,
                    cls_sha256=sha(output / 'cls.npy'), unique_images=len(cached),
                    offline_forward_calls=encoder.calls,
                    node_count=len({r['node_id'] for r in refs}),
                    ranking='exact cosine; max over edge pre-images per source node; ties node_id then edge_id',
                    scope='all graph action edges; no goal filtering; retrieval only')
    save_json(output / 'manifest.json', manifest)
    return manifest


def query(cache, current, output, model=DEFAULT_MODEL, device='cpu', reset=None):
    query_started = time.perf_counter()
    cache, current, output = Path(cache).resolve(), Path(current).resolve(), Path(output).resolve()
    manifest = json.loads((cache / 'manifest.json').read_text())
    if sha(cache / 'cls.npy') != manifest['cls_sha256']:
        raise ValueError('CLS cache checksum mismatch')
    load_started = time.perf_counter()
    encoder = JointEncoder(model, device)
    model_load_s = time.perf_counter()-load_started
    if encoder.contract != manifest['contract']:
        raise ValueError('Reference/current encoder contracts differ; rebuild cache')
    initialization = None
    if reset is not None:
        reset = Path(reset).resolve()
        initialization = json.loads((reset / 'result.json').read_text())
        if initialization['status'] != 'EXECUTED' or sha(reset / initialization['final_after']) != sha(current):
            raise ValueError('Current image must be the completed reset final observation')
        initialization = dict(directory=str(reset), result=initialization,
                              result_sha256=sha(reset / 'result.json'))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(current, output / 'I_current.png')
    encode_started = time.perf_counter()
    with Image.open(output / 'I_current.png') as im:
        cls, patches, geometry, seconds = encoder.encode_joint(im.convert('RGB'))
    decode_preprocess_forward_s = time.perf_counter()-encode_started
    np.savez_compressed(output / 'current_features.npz', cls=cls, patches=patches, valid=valid_patches(geometry))
    search_started = time.perf_counter()
    reference_cls = np.load(cache / 'cls.npy', allow_pickle=False)
    reference_cls_load_s = time.perf_counter()-search_started
    rank_started = time.perf_counter()
    ranked = rank_nodes(manifest['references'], reference_cls, cls)
    exact_ranking_s = time.perf_counter()-rank_started
    time_to_selection_s = time.perf_counter()-query_started
    top3 = ranked[:3]
    for candidate in top3:
        source = cache / candidate['image']
        if sha(source) != candidate['source_image']['sha256']:
            raise ValueError('Cached reference RGB checksum mismatch')
        if sha(cache / candidate['features']) != candidate['features_sha256']:
            raise ValueError('Cached patches checksum mismatch')
        shutil.copyfile(source, output / f"reference_{candidate['rank']}.png")
    evidence = dict(status='RETRIEVED', cache=str(cache), manifest_sha256=sha(cache / 'manifest.json'),
                    contract=encoder.contract, current=dict(source=str(current), rgb='I_current.png',
                    sha256=sha(output / 'I_current.png'), geometry=geometry, features='current_features.npz',
                    features_sha256=sha(output / 'current_features.npz')),
                    initialization=initialization, current_encoded_images=1, current_forward_calls=encoder.calls,
                    online_reference_forward_calls=0, current_forward_s=seconds,
                    top3=top3, selected=top3[0], ranking=manifest['ranking'], scope=manifest['scope'],
                    score_gap_top1_top2=top3[0]['score']-top3[1]['score'] if len(top3)>1 else None,
                    reliability='UNCALIBRATED; cosine is not a success probability',
                    selected_action_executed=False)
    assert encoder.calls == 1
    save_json(output / 'evidence.json', evidence)
    save_json(output / 'ranking.json', ranked)
    save_json(output / 'selected_action_edge.json', top3[0]['edge'])
    montage = Image.new('RGB', (960, 640), 'white')
    draw = ImageDraw.Draw(montage)
    panels = [('I_current.png', 'CURRENT')] + [(f"reference_{c['rank']}.png",
               f"#{c['rank']} {c['node_id']} / {c['edge_id']} cosine={c['score']:.4f}") for c in top3]
    for i, (filename, label) in enumerate(panels):
        with Image.open(output / filename) as im:
            im = im.convert('RGB'); im.thumbnail((480, 285))
            x, y = (i % 2)*480, (i // 2)*320
            montage.paste(im, (x, y+30)); draw.text((x+10, y+10), label, fill='black')
    montage.save(output / 'top3.jpg')
    evidence['timing'] = dict(model_load_s=model_load_s,
        decode_preprocess_forward_s=decode_preprocess_forward_s, forward_s=seconds,
        reference_cls_load_s=reference_cls_load_s, exact_ranking_s=exact_ranking_s,
        time_to_selection_s=time_to_selection_s,
        query_and_artifacts_s=time.perf_counter()-query_started,
        definition='Measured inside query(), includes model cold-load and file checks; excludes Python startup/imports, offline reference encoding and physical reset.')
    save_json(output / 'evidence.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--graph', type=Path, required=True)
    q = sub.add_parser('query'); q.add_argument('--cache', type=Path, required=True)
    q.add_argument('--current', type=Path, required=True); q.add_argument('--reset', type=Path)
    for child in (p, q):
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--model', type=Path, default=DEFAULT_MODEL)
        child.add_argument('--device', default='cpu')
    args = vars(parser.parse_args()); command = args.pop('command')
    result = (prepare if command == 'prepare' else query)(**args)
    print(json.dumps({k: result[k] for k in ('status', 'unique_images', 'node_count', 'current_forward_calls') if k in result}))


if __name__ == '__main__':
    main()
