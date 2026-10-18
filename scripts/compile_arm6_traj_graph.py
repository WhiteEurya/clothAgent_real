#!/usr/bin/env python3
"""Offline Arm6 demonstrations -> visually compiled evidence graph (no dispatch)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments.graph_policy import Requests, evidence_file, frozen, graph_validation
from cloth_agent.experiments.graph_policy_schema import GRAPH
from cloth_agent.experiments.discovery_model import CodexMultimodal


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def import_session(session, *, replay_rejected_iterations=None):
    # Explicit, per-iteration user overrides preserve the original recorder verdict.
    replay_rejected_iterations = replay_rejected_iterations or {}
    folders = {}
    for path in list(session.glob('traj_*/traj.json')) + list(session.glob('pending/iter_*/traj.json')):
        index = int(path.parent.name.rsplit('_', 1)[1])
        if index in folders:
            raise ValueError(f'duplicate iteration {index}')
        folders[index] = path.parent
    observations = {}
    for index, folder in sorted(folders.items()):
        directory = folder/'observation'
        if not (directory/'rgb.png').is_file():
            continue
        metadata = read(directory/'observation.json')
        observations[f'o{index:03d}'] = dict(rgb=evidence_file(directory/'rgb.png'),
            observation=evidence_file(directory/'observation.json'),
            surface=evidence_file(directory/'world_xyz_mm.npy'),
            captured_at=metadata['captured_at'], arm='left', camera='A')
    steps = []
    excluded = []
    for index, folder in sorted(folders.items()):
        manifest = read(folder/'traj.json')
        replay_reason = replay_rejected_iterations.get(index)
        if replay_reason and (manifest.get('status') != 'REJECTED' or manifest.get('operator_accepted') is not False):
            raise ValueError('replay override requires an explicitly rejected iteration')
        if not replay_reason and (manifest.get('status') != 'COMPLETE' or manifest.get('operator_accepted') is not True):
            excluded.append(dict(iteration=index, status=manifest.get('status')))
            continue
        if manifest['arm'] != 'left' or manifest['camera'] != 'A':
            raise ValueError('requires Arm6 / Camera A')
        states = rows(folder/'states.jsonl')
        if len(states) != manifest['samples'] or not states:
            raise ValueError('recorded sample count mismatch')
        events = rows(folder/'events.jsonl')
        manual = [e for e in events if e['kind'] == 'manual_confirmed']
        if len(manual) != 1:
            raise ValueError('requires one successful handoff')
        phases = []
        for phase in ('approach', 'transfer_place', 'return_perception'):
            result = read(folder/phase/'result.json')
            if result['status'] not in ('EXECUTED','ALREADY_AT_PERCEPTION'):
                raise ValueError(f'incomplete phase: {folder}/{phase}')
            phases.append(dict(phase=phase, result=result,
                actual_native_plan=evidence_file(folder/phase/'native_plan.json'),
                motion_source=evidence_file(folder/phase/'motion.py')))
        close_poses = []
        for event in events:
            if event.get('intent',{}).get('gripper') == 'close':
                sample = min(states,key=lambda s:abs(s['t_s']-event['t_s']))
                close_poses.append(dict(button_time_s=event['t_s'], sample_time_s=sample['t_s'],
                    measured_tcp_mm_deg=sample['world_tcp_mm_deg'], gripper_pulse=sample['gripper_pulse']))
        after = f'o{index+1:03d}' if f'o{index+1:03d}' in observations else None
        action = dict(type='ARM6_ASSISTED_DEMONSTRATION', arm='left', axis=6,
            units='world_mm_deg_and_gripper_pulse', selected_points=read(folder/'points.json'),
            automated_phases=phases, manual_input_trace=evidence_file(folder/'events.jsonl'),
            measured_trajectory=evidence_file(folder/'states.jsonl'),
            controller_events=evidence_file(folder/'controller/events.jsonl'),
            handoff=manual[0], sampled_close_poses=close_poses,
            manifest=evidence_file(folder/'traj.json'))
        if replay_reason:
            action['replay_authorization'] = dict(kind='USER_REQUESTED_REJECTED_DEMONSTRATION',
                manifest=action['manifest'].copy(), reason=replay_reason)
        steps.append(dict(step_id=f's{index:03d}',before_observation=f'o{index:03d}',
            after_observation=after,commanded_action=action,
            actual_execution=dict(status='EXECUTED',physical_execution=True,operator_accepted=manifest['operator_accepted']),
            telemetry=dict(samples=len(states), phases=dict(Counter(s['phase'] for s in states))),
            outcome=dict(after_provenance='next iteration initial RGB, not a dedicated post-action capture' if after else 'missing',
                continuity_verified=False, physical_cloth_outcome='UNKNOWN until visual assessment; intervening manual changes cannot be excluded')))
    if not steps:
        raise ValueError('no operator-confirmed Arm6 trajectories')
    return dict(protocol='arm6-demonstration-experience-v1',
        goal='Describe and organize the demonstrated cloth manipulations; final garment goal was not recorded.',
        observations=observations,steps=steps,excluded_iterations=excluded,
        limitations=['Next-iteration images are temporally subsequent evidence, not verified immediate outcomes.',
            'Human confirmation approves retaining the demonstration, not task success or universal action validity.',
            'Measured manual trajectories are evidence, not exact planned grasp primitives. No automatic replay enabled.',
            'No graph node is asserted to be a terminal folded goal.'])


def bind_recorded_actions(graph, trajectory):
    """Action evidence comes from the recorder, never model-transcribed hashes."""
    import copy
    steps = {step['step_id']: step for step in trajectory['steps']}
    changes = []
    for edge in graph['edges']:
        step = steps[edge['seed_step_id']]
        expected = dict(exact_command=step['commanded_action'],
                        source_observation=step['before_observation'])
        if edge['seed_action'] != expected:
            changes.append(edge['edge_id'])
        edge['seed_action'] = copy.deepcopy(expected)
    graph_validation(graph, trajectory)
    return changes


def render(graph, trajectory, output):
    images = output/'images'; images.mkdir()
    for key, observation in trajectory['observations'].items():
        shutil.copy2(observation['rgb']['path'], images/f'{key}.png')
    dot = ['digraph G { rankdir=LR; node [shape=box];']
    for node in graph['nodes']:
        dot.append(f'{json.dumps(node["node_id"])} [label={json.dumps(node["node_id"]+": "+node["task_context"])}];')
    for edge in graph['edges']:
        dot.append(f'{json.dumps(edge["source_node"])} -> {json.dumps(edge["target_node"])} [label={json.dumps(edge["edge_id"]+" / "+edge["seed_step_id"]+" / "+edge["seed_outcome"])}];')
    (output/'graph.dot').write_text('\n'.join(dot+['}']))
    subprocess.run(['dot','-Tsvg',str(output/'graph.dot'),'-o',str(output/'graph.svg')],check=True)
    esc = html.escape
    body = ['<html><meta charset="utf-8"><title>Arm6 trajectory graph</title><style>body{font:16px sans-serif;margin:28px}img{max-width:46%;margin:8px}article{border-top:1px solid #aaa;padding:20px 0}object{width:100%;height:480px}</style>',
        '<h1>Arm6 轨迹经验图</h1><p>离线编译，不执行机器人。后图来自下一 iter；中间人工干预未验证。所有边暂不可自动重放。</p><object data="graph.svg" type="image/svg+xml"></object>']
    steps = {s['step_id']:s for s in trajectory['steps']}
    for node in graph['nodes']:
        body.append(f'<article><h2>{esc(node["node_id"])}：{esc(node["task_context"])}</h2>')
        body.extend(f'<p>{esc(t)}</p>' for t in node['observation_condition']['positive_evidence'])
        body.extend(f'<img src="images/{obs}.png">' for obs in node['representative_observations'])
        body.append('</article>')
    for edge in graph['edges']:
        step = steps[edge['seed_step_id']]
        body.append(f'<article><h2>{esc(edge["edge_id"])}: {esc(edge["source_node"])} → {esc(edge["target_node"])}</h2><p>{esc(edge["action_family"])} / {esc(edge["seed_outcome"])}</p><p>{esc(edge["observed_outcome"])}</p>')
        for key in ('before_observation','after_observation'):
            if step[key]:body.append(f'<img src="images/{step[key]}.png">')
        body.append('<pre>'+esc(json.dumps(edge['action_template'],ensure_ascii=False,indent=2))+'</pre></article>')
    (output/'index.html').write_text('\n'.join(body+['</html>']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    trajectory=import_session(args.session)
    frozen(args.output/'trajectory.json',trajectory)
    requests=Requests(args.output/'compile',CodexMultimodal('gpt-6-astra',900),max_calls=2,
        schemas={'compile':GRAPH},common='You compile offline Arm6 / Camera A cloth experience graphs. Return schema-valid JSON. Use Chinese descriptions. Never execute robot commands.',
        prompts={'compile':'''Compile every supplied physical demonstration into an observation-action graph. Visually compare all images, merge states only when decision-relevant cloth configuration is equivalent. Do not invent a predefined folding sequence or a final goal. Every step must have exactly one edge; Use {} for seed_action.exact_command; the host binds the original recorded action by seed_step_id. Put before_observation into seed_action.source_observation. Source/target nodes must cite respective before/after observations. Consecutive transitions sharing an observation must share the node. Each node needs positive/negative/uncertainty conditions, representative observation IDs, confidence. Each edge needs a relative grasp/target strategy derived from selected points, measured manual close poses and handoff, not invented Arm7 primitives. Outcome evidence cites only supplied observation/step IDs. Controller success is not visual progress. Subsequent images may include unobserved human intervention: explicitly state attribution uncertainty; do not label causal SUCCESS. Missing after image requires an unobserved zero-confidence sink with uncertainty conditions and no outgoing edges. All edges are ONE_SHOT_HYPOTHESIS. Do not claim replayability or general validation. Distinguish cloth rearrangement/no-progress from folding progress. Keep conditions specific to visible cloth regions, with missing semantics marked unknown.'''},protocol='arm6-traj-graph-v1')
    graph=requests.ask('compile',trajectory,[(key,Path(obs['rgb']['path'])) for key,obs in trajectory['observations'].items()],
        validator=lambda value:bind_recorded_actions(value,trajectory))
    for edge in graph['edges']:
        edge.update(replayable=False,replay_block_reason='Arm6 manual trajectory needs a dedicated replay compiler and verified post-action observation.')
    graph.update(protocol='arm6-traj-experience-graph-v1',robot='Arm6',camera='A',
        compiled_at=datetime.now(timezone.utc).isoformat(),source_trajectory=evidence_file(args.output/'trajectory.json'),
        compile_metrics=requests.metrics(),terminal_node_ids=[])
    frozen(args.output/'graph.json',graph)
    render(graph,trajectory,args.output)
    summary=dict(nodes=len(graph['nodes']),edges=len(graph['edges']),
        observations=len(trajectory['observations']),replayable_edges=0,
        outcomes=dict(Counter(e['seed_outcome'] for e in graph['edges'])),
        excluded=trajectory['excluded_iterations'],model_metrics=requests.metrics())
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
