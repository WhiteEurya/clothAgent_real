"""Top-down primary graph proposal with an independent same-frame raw shadow.

One observation per round. No robot execution interface; raw never substitutes
for primary. Call plan_observation again with a fresh capture for the next round.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from cloth_agent.dual_arm.observation import Observation, sha
from cloth_agent.site_observation import ROOT, DEFAULT_PROFILE, load_site
from cloth_agent.tabletop_view import reconstruct
from .arm6_traj_replay import compile_templates, matching_graph, GARMENT_FRAME
from .discovery_data import save
from .discovery_schema import obj, S, N, IDS, enum
from .graph_policy import Requests, reliable_match
from .graph_policy_matching import signatures, reference_images, validate_match_response
from .graph_policy_schema import MATCH, VECTOR2
from .partitioned_folding import DEFAULT_GRAPH, DEFAULT_POLICY, load_policy, plan_step, render_preview

DEFAULT_VIEW_CONFIG = ROOT / 'config/topdown.zed_red.json'
NULL_PIXEL = {'anyOf': [VECTOR2, {'type': 'null'}]}
GROUND = obj(status=enum('READY', 'BLOCKED'), edge_id=S, confidence=N,
    centerline_px={'anyOf': [{'type': 'array', 'items': VECTOR2, 'minItems': 2, 'maxItems': 2}, {'type': 'null'}]},
    grasp_px=NULL_PIXEL, target_px=NULL_PIXEL, evidence=IDS, unresolved_conditions=IDS)
COMMON = '''Return only JSON matching the supplied schema. Images are attached in image_ids order.
The first image is CURRENT. All pixel coordinates are in that image's native width/height,
integer [x,y], origin top-left. Other images are HISTORICAL graph references, not current.
Give concise visual evidence, not private reasoning. Do not claim physical success.
'''+GARMENT_FRAME.split('All returned grasp_px')[0]+'''
The host may supply a table-plane top-down current image; do NOT undo that transform in
your pixel answers. Do not invent occluded/cropped boundaries or hidden garment landmarks.
'''
PROMPTS = {
    'match': '''Identify the observed cloth state, using node signatures and their references.
Do not select actions. MATCH requires one reliable node with visible positive evidence and
no violated or unresolved decision-critical condition. Otherwise return AMBIGUOUS or NO_MATCH.
Graph references retain their historical camera perspective; compare garment-relative folds
and layer coverage after accounting for rotation, viewpoint and current garment scale.''',
    'ground': '''Choose ONLY an eligible outgoing graph edge and locate its current grasp and
placement anchors. Preserve the demonstrated garment-relative grasp/target relations.
Do not copy historical pixel coordinates, robot arm identity or metric poses.
Also locate TWO distinct visible points along the garment's longitudinal centerline,
as far apart as visibility permits, lying on observed cloth. The host uses measured depth
and physical base positions to choose one arm; you do not choose the arm.
Return READY only if edge correspondence, centerline and both anchors are supported by
visible evidence, with no unresolved critical conditions. Return BLOCKED otherwise.
READY requires non-null centerline_px, grasp_px, target_px and positive evidence.
The host will use one gripper for open/approach/grasp/close/lift/transfer/place/open/retreat;
if that cannot represent the chosen relation, return BLOCKED. No synchronous actions.''',
}


def annotate_view(image_path, decision, output):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    if decision.get('centerline_px'):
        draw.line([tuple(p) for p in decision['centerline_px']], fill='#ffe066', width=4)
    for key, color in [('grasp_px', '#ff5555'), ('target_px', '#55eaff')]:
        if decision.get(key):
            x, y = decision[key]
            draw.ellipse((x-6, y-6, x+6, y+6), fill=color, outline='white')
            draw.text((x+9, y), key, fill=color, stroke_fill='black', stroke_width=1)
    if decision.get('grasp_px') and decision.get('target_px'):
        draw.line([tuple(decision['grasp_px']), tuple(decision['target_px'])], fill='white', width=2)
    image.save(output)


def mapped_request(observation, decision, node, view=None):
    convert = view.source_pixel if view else lambda pixel: pixel
    return dict(observation_id=observation.meta['observation_id'],
        rgb_sha256=observation.meta['sha256']['rgb.png'], current_node=node,
        edge_id=decision['edge_id'], confidence=decision['confidence'], anchor_source='model',
        centerline_px=[convert(p) for p in decision['centerline_px']],
        grasp_px=convert(decision['grasp_px']), target_px=convert(decision['target_px']),
        center_arm=None, decision_view='topdown' if view else 'raw',
        decision_image_sha256=sha(view.image if view else observation.image),
        decision_pixels={k: decision[k] for k in ('centerline_px', 'grasp_px', 'target_px')})


def propose_branch(observation, graph, trajectory, templates, profile, policy,
                   table_normal, image_path, output, backend, *, view=None, max_calls=4, garment_clipped=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests = Requests(output / 'model', backend, no_cache=True, max_calls=max_calls,
        schemas={'match': MATCH, 'ground': GROUND}, prompts=PROMPTS, common=COMMON,
        protocol='topdown-paired-folding-v1')
    result = dict(status='BLOCKED', view='topdown' if view else 'raw', execution_allowed=False,
        role='primary' if view else 'shadow', robot_calls=0, observation_id=observation.meta['observation_id'],
        input_sha256=sha(image_path), match=None, decision=None, plan=None)
    try:
        mg = matching_graph(graph)
        candidates = [n['node_id'] for n in mg['match_nodes']]
        payload = signatures(mg, candidates, with_representatives=True)
        # Remove the legacy sentence saying the host never transforms images.
        payload['task']['side_convention'] = COMMON
        payload['current_image_size_wh'] = list(Image.open(image_path).size)
        payload['current_view'] = ('table-plane RGB warp: elevated folds have parallax; cropped/occluded cloth is not recovered'
                                 if view else 'original rectified camera RGB; cropped/occluded cloth is not recovered')
        payload['capture_garment_clipped'] = garment_clipped
        payload['clipping_instruction'] = 'Clipping may create artificial contour edges. Do not use a clipped boundary as a true garment edge; abstain if this affects state/anchor correspondence.'
        images = reference_images(mg, trajectory, SimpleNamespace(image=image_path), candidates)
        result['match'] = requests.ask('match', payload, images,
            validator=lambda value: validate_match_response(value, mg, policy['minimum_anchor_confidence']))
        match = result['match']
        threshold = policy['minimum_anchor_confidence']
        if not reliable_match(match, mg, threshold):
            result['status'] = 'STATE_UNCERTAIN'
            return result
        node = match['node_id']
        if node in graph.get('terminal_node_ids', []):
            result['status'] = 'GOAL_OBSERVED'
            return result
        outgoing = [e for e in graph['edges'] if e['source_node'] == node and e['edge_id'] in templates]
        if not outgoing:
            result['status'] = 'NO_SUPPORTED_EDGE'
            return result
        edges = [{k: e[k] for k in ('edge_id', 'source_node', 'target_node', 'action_family', 'action_template')}
                 for e in outgoing]
        grounding_payload = dict(current_image_size_wh=payload['current_image_size_wh'],
            current_view=payload['current_view'], matched_node=node,
            capture_garment_clipped=garment_clipped, clipping_instruction=payload['clipping_instruction'],
            eligible_edges=edges, minimum_confidence=threshold)
        def validate_ground(value):
            if value['status'] != 'READY':
                return
            if value['edge_id'] not in {e['edge_id'] for e in outgoing}:
                raise ValueError('Edge must be outgoing from the matched node')
            if (not value['centerline_px'] or not value['grasp_px'] or not value['target_px']
                    or not value['evidence'] or value['unresolved_conditions']):
                raise ValueError('READY needs all four visible anchors, evidence and no unresolved conditions')
            width, height = payload['current_image_size_wh']
            for x, y in [*value['centerline_px'], value['grasp_px'], value['target_px']]:
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError('Anchor outside CURRENT image bounds')
        decision = requests.ask('ground', grounding_payload, images, validator=validate_ground)
        result['decision'] = decision
        save(output / 'decision.json', decision)
        if decision['status'] != 'READY' or decision['confidence'] < threshold:
            result['status'] = 'ANCHORS_UNCERTAIN'
            return result
        annotate_view(image_path, decision, output / 'decision_preview.png')
        request = mapped_request(observation, decision, node, view)
        save(output / 'request.json', request)
        plan = plan_step(graph, templates, profile, observation, request, policy, table_normal)
        plan['model_calls'] = requests.model_calls
        plan['decision_role'] = result['role']
        # A model's confident judgment is not independent visual/physical validation.
        result.update(status=plan['status'], plan=plan)
        save(output / 'plan.json', plan)
        render_preview(observation, plan, output / 'source_preview.png')
    except Exception as exc:
        result.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
    finally:
        result['metrics'] = requests.metrics()
        save(output / 'result.json', result)
    return result


def compare(primary, shadow):
    result = dict(primary_view='topdown', shadow_view='raw', execution_allowed=False, robot_calls=0,
        primary_status=primary['status'], shadow_status=shadow['status'],
        interpretation='Same-frame action proposals only; no physical fold outcome measured. One pair cannot separate view effects from model variability.')
    for label, field, key in [('node', 'match', 'node_id'), ('edge', 'decision', 'edge_id'), ('arm', 'plan', 'selected_arm')]:
        a, b = (primary.get(field) or {}).get(key), (shadow.get(field) or {}).get(key)
        result[label] = dict(topdown=a, raw=b, same=(a == b if a and b else None))
    for key in ('grasp_common_mm', 'target_common_mm'):
        a, b = (primary.get('plan') or {}).get(key), (shadow.get('plan') or {}).get(key)
        result[key] = dict(topdown=a, raw=b, distance_mm=float(np.linalg.norm(np.asarray(a) - b)) if a is not None and b is not None else None)
    result['both_proposals_ready'] = primary['status'] == shadow['status'] == 'PREVIEW_READY'
    return result


def comparison_preview(output, branches, observation):
    """Both proposals on the SAME source RGB, so anchor differences are visible."""
    panels = []
    for name in ('topdown', 'raw'):
        source = output / name / 'proposal' / 'source_preview.png'
        image = Image.open(source if source.is_file() else observation.image).convert('RGB')
        image = image.resize((640, round(image.height * 640 / image.width)))
        panel = Image.new('RGB', (640, image.height + 44), '#182030')
        panel.paste(image, (0, 44))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), f"{name.upper()} {'PRIMARY' if name == 'topdown' else 'SHADOW'}", fill='white')
        draw.text((10, 25), branches[name]['status'], fill='white')
        panels.append(panel)
    combined = Image.new('RGB', (1280, max(p.height for p in panels)), '#182030')
    combined.paste(panels[0], (0, 0))
    combined.paste(panels[1], (640, 0))
    combined.save(output / 'comparison.png')


def plan_observation(observation, output, *, graph_path=DEFAULT_GRAPH, profile_path=DEFAULT_PROFILE,
                     policy_path=DEFAULT_POLICY, view_config_path=DEFAULT_VIEW_CONFIG,
                     backend_factory=None, prepare_only=False, max_calls=4):
    """Integration entry: one snapshot, top-down primary, raw shadow, no dispatch."""
    if max_calls < 2 or max_calls > 8:
        raise ValueError('max_calls per branch must be between 2 and 8')
    # Validate graph/site before model calls; copy capture so both branches are immutable inputs.
    graph, trajectory, templates = compile_templates(graph_path)
    profile, policy = load_site(profile_path), load_policy(policy_path)
    if (observation.meta.get('calibration_id') != profile['calibration_id']
            or observation.meta.get('common_frame') != profile['common_frame']
            or observation.meta.get('camera_serial') != profile['camera']['serial']):
        raise ValueError('Observation does not match the current site')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'observation').mkdir()
    for name in ('rgb.png', 'world_xyz_mm.npy', 'observation.json'):
        shutil.copyfile(observation.directory / name, output / 'observation' / name)
    observation = Observation.load(output / 'observation')
    save(output / 'experiment.json', dict(protocol='topdown-paired-folding-v1', primary='topdown', shadow='raw',
        observation_id=observation.meta['observation_id'], source_sha256=observation.meta['sha256'],
        graph_sha256=sha(graph_path), profile_sha256=sha(profile_path), policy=policy,
        view_config=json.loads(Path(view_config_path).read_text()), model_profile='rbs',
        max_calls_per_branch=max_calls, separate_model_sessions=True, response_sharing=False,
        graph_reference_images='Same unmodified historical perspective RGBs in both branches',
        execution_allowed=False, robot_calls=0))
    try:
        view = reconstruct(observation, output / 'topdown', json.loads(Path(view_config_path).read_text()))
    except Exception as exc:
        save(output / 'primary_plan.json', dict(status='TOPDOWN_PREPARATION_FAILED', error=str(exc),
            execution_allowed=False, robot_calls=0, model_calls=0, operations=[]))
        raise
    if prepare_only:
        result = dict(status='INPUTS_PREPARED', output=str(output), model_calls=0, robot_calls=0, execution_allowed=False)
        save(output / 'preparation.json', result)
        return result
    if backend_factory is None:
        from .discovery_model import CodexMultimodal
        backend_factory = CodexMultimodal
    branches = {}
    for name in ('topdown', 'raw'):
        is_primary = name == 'topdown'
        branches[name] = propose_branch(observation, copy.deepcopy(graph), trajectory, templates, profile, policy,
            view.meta['table_normal_common'], view.image if is_primary else observation.image,
            output / name / 'proposal', backend_factory(), view=view if is_primary else None,
            max_calls=max_calls, garment_clipped=view.meta['garment_clipped'])
    primary = branches['topdown']
    save(output / 'primary_plan.json', primary.get('plan') or dict(status=primary['status'],
        execution_allowed=False, robot_calls=0, operations=[], detail='See topdown/proposal/result.json; raw shadow never substitutes'))
    comparison = compare(primary, branches['raw'])
    comparison.update(observation_id=observation.meta['observation_id'],
        model_calls=sum(b['metrics']['model_calls'] for b in branches.values()),
        garment_clipped=view.meta['garment_clipped'])
    save(output / 'comparison.json', comparison)
    comparison_preview(output, branches, observation)
    lines = ['# 同帧俯视图 / 原图动作提案对照', '',
        '| 项目 | 俯视图（主流程） | 原图（对照） |', '| --- | --- | --- |',
        f"| 状态 | {primary['status']} | {branches['raw']['status']} |"]
    for key in ('node', 'edge', 'arm', 'grasp_common_mm', 'target_common_mm'):
        row = comparison[key]
        lines.append(f"| {key} | {row['topdown']} | {row['raw']} |")
    lines.extend(['', '抓放坐标均为右臂底座公共系毫米；只有有效提案才计算距离。'])
    for key in ('grasp_common_mm', 'target_common_mm'):
        lines.append(f"- {key} 距离：{comparison[key]['distance_mm']} mm")
    lines.extend(['', '仅比较动作提案，没有执行机器人动作。一次对照不能排除模型自身波动。',
                  '主流程只使用俯视图；主流程失败时不会切换为原图提案。'])
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--view-config', type=Path, default=DEFAULT_VIEW_CONFIG)
    parser.add_argument('--prepare-only', action='store_true', help='Render and verify inputs, no model calls')
    parser.add_argument('--max-calls-per-branch', type=int, default=4)
    args = parser.parse_args(argv)
    result = plan_observation(Observation.load(args.observation), args.output,
        graph_path=args.graph, profile_path=args.profile, policy_path=args.policy,
        view_config_path=args.view_config, prepare_only=args.prepare_only, max_calls=args.max_calls_per_branch)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if args.prepare_only or result['primary_status'] in ('PREVIEW_READY', 'GOAL_OBSERVED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
