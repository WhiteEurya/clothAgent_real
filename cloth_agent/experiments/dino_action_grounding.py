"""Visualize an immutable DINO-selected point-pair action on the saved current RGB.

GPT grounds exactly one frozen reference. No candidate selection, robot dispatch,
or DINO encoding occurs here. Times distinguish original inference, cached search
replay and a fresh grounding request. Human GT is collected in an offline HTML.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from cloth_agent.dino_correspondence import save_json, sha
from .discovery_schema import obj, enum, S, N
from .discovery_model import CodexMultimodal
from .graph_policy import Requests
from .dino_reference_baseline import rank_nodes


POINT = {'anyOf': [{'type': 'null'}, {'type': 'array', 'items': {'type': 'number'},
                                   'minItems': 2, 'maxItems': 2}]}
SCHEMA = obj(status=enum('PROPOSED', 'UNCERTAIN', 'NOT_APPLICABLE'),
             grasp_xy=POINT, target_xy=POINT, confidence=N,
             reference_grasp_description=S, reference_destination_relation=S,
             current_grasp_evidence=S, current_destination_evidence=S,
             transfer_reasoning=S, uncertainty=S)
PROMPT = '''Ground the supplied FIXED reference pick-and-place action onto the current RGB.
DINO already selected the sole reference. You cannot select, rerank or replace it.
Images in order: raw reference; the same reference with Pa (grasp) -> Pb (release
destination) arrow; raw current RGB. The numeric reference points are authoritative.
Return A=grasp_xy and B=target_xy in ORIGINAL CURRENT IMAGE pixel coordinates
(x rightwards, y downwards, integer pixel centres; origin upper-left).
Transfer the reference action's garment-relative meaning, accounting for orientation,
folds and visible material. A must lie on visible graspable fabric. B is the desired
release destination, which can be on the table; it is NOT necessarily a garment
feature to match. Do not merely copy image coordinates or the screen-space vector.
Do not redesign the task or invent a different action. Identify the physical feature
at reference Pa and its corresponding visible feature in the current image; if it
is hidden or ambiguous, explicitly report UNCERTAIN or NOT_APPLICABLE. Use null
points when no defensible proposal exists. Do not invent hidden cloth geometry.
Explain both endpoints and ambiguity in Chinese. Confidence is only a subjective
diagnostic, not measured accuracy. Do not generate robot commands or use any tools.
Return only schema-valid JSON.'''


def validate(value, size):
    points = [value['grasp_xy'], value['target_xy']]
    for point in points:
        if point is not None:
            p = np.asarray(point, float)
            if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > np.asarray(size)-1):
                raise ValueError('Point outside original current image')
    if (points[0] is None) != (points[1] is None):
        raise ValueError('Provide both endpoints or neither')
    if value['status'] == 'PROPOSED' and points[0] is None:
        raise ValueError('PROPOSED requires both endpoints')
    if value['status'] == 'NOT_APPLICABLE' and points[0] is not None:
        raise ValueError('NOT_APPLICABLE must not invent endpoints')
    if points[0] is not None and np.linalg.norm(np.subtract(*points)) < 1:
        raise ValueError('Endpoints coincide')


def action_points(selected):
    command = selected['edge']['seed_action']['exact_command']
    points = {p['name']: p['pixel_xy'] for p in command.get('selected_points', [])}
    if 'start' not in points or 'target' not in points:
        raise ValueError('DINO-selected action lacks recorded Pa/Pb; do not substitute another reference')
    return points['start'], points['target']


def plot_action(path, a, b, output, *, title=None, uncertain=False):
    """Data visualization in exact original pixel coordinates; source RGB untouched."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB')); w, h = im.size
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(rgb); ax.axis('off')
    if a is not None and b is not None:
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color='#ffd43b',
                    lw=3, mutation_scale=28, linestyle='--' if uncertain else '-'))
        for label, point, color in [('A', a, '#ff5252'), ('B', b, '#37dbff')]:
            ax.scatter(*point, s=220, facecolors='none', edgecolors='black', linewidths=6)
            ax.scatter(*point, s=220, facecolors='none', edgecolors=color, linewidths=3)
            ax.text(point[0]+14, point[1]-14, label, color=color, fontsize=20, weight='bold',
                    bbox=dict(facecolor='black', alpha=.8, edgecolor='none', pad=2))
    if title:
        ax.text(.02, .96, title, transform=ax.transAxes, va='top', color='white', fontsize=16,
                bbox=dict(facecolor='black', alpha=.8, edgecolor='none', pad=5))
    fig.savefig(output, dpi=100); plt.close(fig)


def write_review(output, evidence, grounded, timing, reference_points):
    def data_uri(path):
        return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()
    retrieval = Path(evidence['_retrieval_directory'])
    with Image.open(retrieval/'I_current.png') as im: w, h = im.size
    bundle = dict(grasp_xy=grounded['grasp_xy'], target_xy=grounded['target_xy'], width=w, height=h,
                  current_sha256=evidence['current']['sha256'], edge_id=evidence['selected']['edge_id'],
                  grounding_status=grounded['status'])
    retrieval_timing = timing.get('original_retrieval_timing')
    measured_search = retrieval_timing['exact_ranking_s'] if retrieval_timing else timing['cached_exact_search_replay_s']
    measurement_note = (f"本轮 DINO 查询及存图 {retrieval_timing['query_and_artifacts_s']:.2f} 秒，其中模型首次加载 "
                        f"{retrieval_timing['model_load_s']:.2f} 秒；不含 Python 启动、离线缓存和机械臂初始化。"
                        if retrieval_timing else 'DINO 原始记录只计了前向；缓存排序另行回放计时，这些分项不是完整在线总耗时。')
    template = Path(__file__).with_name('dino_action_review.html').read_text()
    fields = dict(REFERENCE=data_uri(output/'reference_action.png'), CURRENT=data_uri(retrieval/'I_current.png'),
                  WIDTH=str(w), HEIGHT=str(h), DATA=json.dumps(bundle).replace('<', '\\u003c'),
                  STATUS=html.escape(grounded['status']),
                  REASON=html.escape(grounded['transfer_reasoning']),
                  UNCERTAINTY=html.escape(grounded['uncertainty']),
                  EDGE=html.escape(evidence['selected']['edge_id']),
                  DINO_MS=f"{timing['dino_original_forward_s']*1000:.1f}",
                  SEARCH_MS=f"{measured_search*1000:.2f}",
                  SEARCH_LABEL='精确排序' if retrieval_timing else '缓存排序回放',
                  MEASUREMENT_NOTE=html.escape(measurement_note),
                  GPT_SECONDS=f"{timing['gpt_grounding_wall_s']:.1f}",
                  TIMING=html.escape(json.dumps(timing, ensure_ascii=False, indent=2)),
                  REFERENCE_POINTS=html.escape(json.dumps(reference_points)))
    for key, value in fields.items(): template = template.replace('@@'+key+'@@', value)
    (output/'review.html').write_text(template)


def run(retrieval, output):
    started = time.perf_counter()
    retrieval, output = Path(retrieval).resolve(), Path(output).resolve()
    evidence = json.loads((retrieval/'evidence.json').read_text())
    selected = evidence['selected']
    cache = Path(evidence['cache'])
    if sha(cache/'manifest.json') != evidence['manifest_sha256']:
        raise ValueError('Reference cache manifest changed')
    manifest = json.loads((cache/'manifest.json').read_text())
    current, reference = retrieval/'I_current.png', retrieval/'reference_1.png'
    if sha(current) != evidence['current']['sha256'] or sha(reference) != selected['source_image']['sha256']:
        raise ValueError('Original current or selected reference RGB changed')
    if sha(cache/'cls.npy') != manifest['cls_sha256'] or sha(retrieval/'current_features.npz') != evidence['current']['features_sha256']:
        raise ValueError('CLS cache changed')
    pa, pb = action_points(selected)
    with Image.open(current) as im: current_size = im.size
    with Image.open(reference) as im: reference_size = im.size
    validate(dict(status='PROPOSED', grasp_xy=pa, target_xy=pb), reference_size)
    # Time exact search on already computed descriptors. Never re-encode current.
    with np.load(retrieval/'current_features.npz', allow_pickle=False) as features:
        cls = features['cls']
    reference_cls = np.load(cache/'cls.npy', allow_pickle=False)
    search_started = time.perf_counter()
    ranked = rank_nodes(manifest['references'], reference_cls, cls)
    search_s = time.perf_counter()-search_started
    if (ranked[0]['node_id'], ranked[0]['edge_id']) != (selected['node_id'], selected['edge_id']):
        raise ValueError('Cached exact search no longer agrees with frozen DINO selection')
    output.mkdir(parents=True, exist_ok=False)
    plot_action(reference, pa, pb, output/'reference_action.png')
    backend = CodexMultimodal(timing_output=output/'model_timing')
    requests = Requests(output/'model', backend, max_calls=2, no_cache=True,
                        schemas={'ground_action': SCHEMA}, prompts={'ground_action': PROMPT},
                        common='Visual grounding of one fixed action. No image search. No robot execution.',
                        protocol='dino-fixed-reference-point-pair-v1')
    payload = dict(reference_size_wh=reference_size, current_size_wh=current_size,
                   reference_grasp_xy=pa, reference_release_xy=pb,
                   coordinate_convention='original image pixel indices; x right, y down',
                   action='pick at Pa, lift/transfer, release at Pb',
                   reference_selection='Already frozen by DINO global cosine top1. No alternatives provided.')
    grounding_started = time.perf_counter()
    try:
        grounded = requests.ask('ground_action', payload,
                    [('reference_raw', reference), ('reference_action', output/'reference_action.png'),
                     ('current_raw', current)], validator=lambda value: validate(value, current_size))
    finally:
        save_json(output/'model_metrics.json', requests.metrics())
    grounding_s = time.perf_counter()-grounding_started
    save_json(output/'grounding.json', grounded)
    plot_action(current, grounded['grasp_xy'], grounded['target_xy'], output/'current_action.png',
                title=f"CURRENT PLAN: {grounded['status']} | " + ('A = grasp, B = release' if grounded['grasp_xy'] is not None else 'NO POINT PAIR'),
                uncertain=grounded['status'] != 'PROPOSED')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    for ax, filename, title in zip(axes, ['reference_action.png', 'current_action.png'],
                    [f"DINO reference {selected['node_id']} / {selected['edge_id']} | Pa -> Pb", 'GPT grounded current action | A -> B']):
        with Image.open(output/filename) as im: ax.imshow(im)
        ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(output/'action_comparison.png'); plt.close(fig)
    timing = dict(dino_original_forward_s=evidence['current_forward_s'],
                  cached_exact_search_replay_s=search_s, gpt_grounding_wall_s=grounding_s,
                  gpt_request_wall_s=requests.metrics()['reasoning_time_s'],
                  gpt_model_calls=requests.model_calls, gpt_cache_hits=requests.cache_hits,
                  model=backend.model, reasoning_effort='high',
                  scope='Original DINO forward and a separately timed cached search replay; not an end-to-end online measurement. GPT wall includes transport, CLI startup and any repair.',
                  original_retrieval_timing=evidence.get('timing'),
                  gpt_event_timing_files=[str(p.relative_to(output)) for p in sorted((output/'model_timing').glob('*.json'))],
                  dino_new_forward_calls=0)
    evidence['_retrieval_directory'] = str(retrieval)
    write_review(output, evidence, grounded, timing, dict(Pa=pa, Pb=pb))
    result = dict(retrieval_evidence_sha256=sha(retrieval/'evidence.json'),
                  current_sha256=sha(current), reference_sha256=sha(reference),
                  selected_node=selected['node_id'], selected_edge=selected['edge_id'],
                  reference_action=dict(Pa=pa, Pb=pb), grounded_action=grounded, timing=timing,
                  grounding_and_artifact_generation_s=time.perf_counter()-started,
                  reference_selector='DINO_ONLY', grounding_inputs='one fixed reference and current RGB',
                  executed=False, accuracy=dict(status='AWAITING_HUMAN_GT' if grounded['grasp_xy'] is not None else 'ABSTAINED_NO_POINT_PAIR',
                    definition='Euclidean original-pixel error separately for A and B; annotation UI can export both. Subjective model confidence is not accuracy.'))
    save_json(output/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.retrieval, args.output)


if __name__ == '__main__': main()
