"""Incrementally compile new confirmed Arm6 recordings, preserving reviewed nodes.

New RGBs are partitioned from original images. Cross-node review is a separate
offline stage using the frozen candidate graph and its DINO memory.
"""
import argparse
import copy
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.compile_arm6_archive_graph import (
    import_archive, action_semantics, build_graph, validate_partition,
)
from cloth_agent.experiments.discovery_model import CodexMultimodal
from cloth_agent.experiments.discovery_schema import obj, array, S
from cloth_agent.experiments.graph_policy import (
    Requests, checked_file, evidence_file, frozen, load_frozen, graph_validation,
)
from cloth_agent.experiments.graph_policy_schema import NODE, GRAPH, EDGE


def extend_evidence(old, imported):
    """Identify attempts by manifest path, never by re-enumerated import IDs."""
    trajectory = copy.deepcopy(old)
    old_steps = {s['commanded_action']['manifest']['path']: s for s in old['steps']}
    old_refs = {o['rgb']['path']: (oid, o) for oid, o in old['observations'].items()}
    next_obs = max(int(k[1:]) for k in old['observations']) + 1
    next_step = max(int(s['step_id'][1:]) for s in old['steps']) + 1
    new_steps = []; new_obs = {}
    for step in imported['steps']:
        manifest = step['commanded_action']['manifest']
        if manifest['path'] in old_steps:
            if step['commanded_action'] != old_steps[manifest['path']]['commanded_action']:
                raise ValueError('previously compiled demonstration changed: ' + manifest['path'])
            continue
        item = copy.deepcopy(step); item['step_id'] = f's{next_step:03d}'; next_step += 1
        for key in ('before_observation', 'after_observation'):
            oid = step[key]
            if oid is None: continue
            obs = imported['observations'][oid]; path = obs['rgb']['path']
            if path in old_refs:
                mapped, previous = old_refs[path]
                for field in ('rgb', 'observation', 'surface'):
                    if obs[field] != previous[field]:
                        raise ValueError('previous observation evidence changed: ' + path)
            else:
                mapped = f'o{next_obs:03d}'; next_obs += 1
                new_obs[mapped] = copy.deepcopy(obs); old_refs[path] = (mapped, obs)
            item[key] = mapped
        new_steps.append(item)
    trajectory['steps'].extend(new_steps); trajectory['observations'].update(new_obs)
    trajectory['excluded_iterations'] = imported['excluded_iterations']
    trajectory['sessions'] = imported['sessions']
    return trajectory, new_steps, new_obs


def next_node_number(graph):
    ids = {n['node_id'] for n in graph['nodes']}
    for revision in graph.get('revision_history', []):
        for merge in revision.get('merges', []): ids.update(merge['source_nodes'])
    return max((int(n[1:]) for n in ids if re.fullmatch(r'n\d+', n)), default=-1) + 1


def partition_new(new_obs, source, out):
    path = out/'new_partition.json'
    if path.exists():
        result = load_frozen(path); validate_partition(result, new_obs); return result
    schema = obj(nodes=array(NODE), limitations=array(S))
    req = Requests(out/'state_model', CodexMultimodal(timeout=900), max_calls=2,
        schemas={'partition': schema}, common='Offline RGB cloth-state clustering. Chinese concise descriptions; no robot commands.',
        prompts={'partition': '''Inspect EVERY attached original-resolution RGB. Partition all supplied observation IDs into visually equivalent cloth states usable for replay. Use actual images, not ID sequence, and account for rotation/translation by neckline-to-hem orientation. Do not mirror garment left/right. Ignore small wrinkles and fold-width changes when exposed regions, accessible edges/corners, and broad coverage remain equivalent. Repeated captures before/after a program restart must share a node when they show the same state; never split solely because pictures were captured in different attempts. Do not assume continuity from order: random resets, rejected actions and manual interventions may exist. Preserve distinct states when broad coverage or exposed-versus-hidden garment regions differ. Do not invent hidden layers from wrinkles, and do not merge merely because an action might work in both. Every RGB must appear exactly once, including all members of large groups. State uncertain distinctions explicitly. No fixed number of groups, no goals or actions. Only visual equivalence is being assessed.''',},
        protocol='arm6-incremental-original-rgb-partition-v1')
    images = [(oid, checked_file(obs['rgb'])) for oid,obs in new_obs.items()]
    result = req.ask('partition',dict(observation_ids=list(new_obs)),images,
                     validator=lambda v:validate_partition(v,new_obs))
    frozen(out/'raw_new_partition.json',result)
    start = next_node_number(source)
    result = copy.deepcopy(result)
    for i,node in enumerate(result['nodes']): node['node_id'] = f'n{start+i:03d}'
    frozen(path,result); frozen(out/'state_metrics.json',req.metrics())
    print(f'New original-RGB partition: {len(new_obs)} images -> {len(result["nodes"])} nodes',flush=True)
    return result


def validate_core(graph, trajectory):
    core = {k:copy.deepcopy(graph[k]) for k in GRAPH['properties']}
    core['edges'] = [{k:e[k] for k in EDGE['required']} for e in core['edges']]
    graph_validation(core,trajectory)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('data/demonstrations'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--import-only',action='store_true')
    args=p.parse_args(argv);out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=load_frozen(args.graph);old=load_frozen(checked_file(source['source_trajectory']))
    config=dict(source_graph=evidence_file(args.graph),root=str(args.root.resolve()),robot_calls=0)
    if (out/'config.json').exists():
        if load_frozen(out/'config.json')!=config:raise ValueError('resume source changed')
    else:frozen(out/'config.json',config)
    if (out/'trajectory.json').exists():
        trajectory=load_frozen(out/'trajectory.json');delta=load_frozen(out/'delta.json')
        new_steps=delta['steps'];new_obs=delta['observations']
    else:
        imported=import_archive(args.root)
        trajectory,new_steps,new_obs=extend_evidence(old,imported)
        if not new_steps:raise ValueError('no new confirmed trajectories')
        frozen(out/'trajectory.json',trajectory)
        frozen(out/'delta.json',dict(steps=new_steps,observations=new_obs))
    summary=dict(added_trajectories=len(new_steps),added_images=len(new_obs),
        sessions=dict(Counter(s['session'] for s in new_steps)),
        new_steps=[dict(step_id=s['step_id'],folder=s['source_folder'],after_provenance=s['outcome']['after_provenance']) for s in new_steps],
        excluded=trajectory['excluded_iterations'],robot_calls=0)
    if not (out/'import_summary.json').exists():frozen(out/'import_summary.json',summary)
    print({k:v for k,v in summary.items() if k not in ('excluded','new_steps')},flush=True)
    if args.import_only:return
    delta_trajectory=dict(trajectory,steps=new_steps)
    # Independent state grouping and descriptions of the recorded actions.
    with ThreadPoolExecutor(max_workers=2) as pool:
        state_future=pool.submit(partition_new,new_obs,source,out)
        semantics=action_semantics(delta_trajectory,out,CodexMultimodal(timeout=900))
        new_partition=state_future.result()
    groups=dict(nodes=copy.deepcopy([n for n in source['nodes'] if n['representative_observations']])+new_partition['nodes'],
                limitations=new_partition['limitations'])
    validate_partition(groups,trajectory['observations'])
    # Existing semantics and action evidence are not regenerated.
    for e in source['edges']:
        semantics[e['seed_step_id']]=dict(action_family=e['action_family'],
            grasp_relation=e['action_template']['grasp_relation'],target_relation=e['action_template']['target_relation'],
            observed_outcome=e['observed_outcome'],seed_outcome=e['seed_outcome'],limitations=e['action_template']['other_constraints'])
    bound=build_graph(trajectory,groups,semantics)
    old_edge_ids={e['edge_id'] for e in source['edges']}
    additions=[e for e in bound['edges'] if e['edge_id'] not in old_edge_ids]
    for e in additions:e.update(replayable=False,replay_block_reason='Offline compiled evidence; requires Arm6 replay preflight.')
    graph=copy.deepcopy(source);graph['graph_id']+='_incremental'
    graph['nodes']+=new_partition['nodes']
    known={n['node_id'] for n in graph['nodes']}
    graph['nodes'] += [n for n in bound['nodes'] if n['node_id'] not in known]
    graph['edges']+=additions;graph['source_trajectory']=evidence_file(out/'trajectory.json')
    graph['limitations']+=new_partition['limitations']
    graph.setdefault('revision_history',[]).append(dict(operation='incremental_original_rgb_compile',
        source_graph=evidence_file(args.graph),added_steps=[s['step_id'] for s in new_steps],raw_files_modified=False))
    validate_core(graph,trajectory)
    assert graph['edges'][:len(source['edges'])]==source['edges']
    assert graph['nodes'][:len(source['nodes'])]==source['nodes']
    frozen(out/'graph.json',graph);frozen(out/'new_node_ids.json',[n['node_id'] for n in new_partition['nodes']])
    frozen(out/'summary.json',dict(status='COMPILED_PENDING_CROSS_NODE_REVIEW',
        added_edges=len(additions),added_observed_nodes=len(new_partition['nodes']),
        nodes=len(graph['nodes']),edges=len(graph['edges']),observations=len(trajectory['observations']),robot_calls=0))
    print('Incremental graph compiled; ready for cross-node RGB merge review.',flush=True)


if __name__=='__main__':main()
