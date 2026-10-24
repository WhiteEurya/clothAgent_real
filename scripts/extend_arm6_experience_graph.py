"""Visually extend an Arm6 graph with assisted and free-manual demonstrations.

Evidence compilation only. Free-manual edges are not two-anchor replay templates.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compile_arm6_traj_graph import import_session, read, rows, render
from cloth_agent.experiments.graph_policy import Requests, load_frozen, frozen, evidence_file, graph_validation
from cloth_agent.experiments.graph_policy_schema import GRAPH, EDGE, NODE
from cloth_agent.experiments.discovery_schema import obj, array, S
from cloth_agent.experiments.discovery_model import CodexMultimodal
from cloth_agent.experiments.arm6_traj_replay import GARMENT_FRAME


def import_manual(session):
    folders={int(p.parent.name.rsplit('_',1)[1]):p.parent for p in session.glob('**/traj.json')}
    observations={};steps=[];excluded=[]
    for i,p in sorted(folders.items()):
        obs=p/'observation'
        if (obs/'rgb.png').exists():
            observations[f'o{i:03d}']=dict(rgb=evidence_file(obs/'rgb.png'),
                observation=evidence_file(obs/'observation.json'),surface=evidence_file(obs/'world_xyz_mm.npy'),
                captured_at=read(obs/'observation.json')['captured_at'],arm='left',camera='A')
    for i,p in sorted(folders.items()):
        m=read(p/'traj.json')
        if m.get('status')!='COMPLETE' or m.get('operator_accepted') is not True:
            excluded.append(dict(iteration=i,status=m['status'],manifest=evidence_file(p/'traj.json')));continue
        if m.get('collection_mode') not in ('single_point_manual','manual_only'):
            raise ValueError('expected free manual collection')
        states=rows(p/'states.jsonl');events=rows(p/'events.jsonl')
        if len(states)!=m['samples'] or not states or not np.isfinite([s['world_tcp_mm_deg'] for s in states]).all():
            raise ValueError('invalid manual trajectory feedback')
        ends=[e for e in events if e['kind']=='manual_recording_finished']
        if len(ends)!=1:raise ValueError('manual trajectory not completed')
        keys=[]
        for e in events:
            if e.get('intent',{}).get('gripper') in ('close','open'):
                sample=min(states,key=lambda s:abs(s['t_s']-e['t_s']))
                keys.append(dict(input=e,nearest_feedback=sample,interpretation='sample near button event, NOT a verified grasp/release endpoint'))
        manual=[s for s in states if s['phase']=='manual']
        stride=max(1,len(manual)//12)
        phases=[]
        if (p/'approach/result.json').exists():
            result=read(p/'approach/result.json')
            if result['status']!='EXECUTED':raise ValueError('incomplete approach')
            phases.append(dict(phase='approach',result=result,actual_native_plan=evidence_file(p/'approach/native_plan.json'),motion_source=evidence_file(p/'approach/motion.py')))
        action=dict(type='ARM6_FREE_MANUAL_DEMONSTRATION',arm='left',axis=6,
            units='world_mm_deg_and_gripper_pulse',collection_mode=m['collection_mode'],
            selected_points=read(p/'points.json') if (p/'points.json').exists() else [],
            automated_phases=phases,manual_input_trace=evidence_file(p/'events.jsonl'),
            measured_trajectory=evidence_file(p/'states.jsonl'),controller_events=evidence_file(p/'controller/events.jsonl'),
            manifest=evidence_file(p/'traj.json'),final_state=evidence_file(p/'final_state.json'),
            manual_finish=ends[0],gripper_input_keyframes=keys,
            manual_trace_summary=manual[::stride]+manual[-1:])
        steps.append(dict(step_id=f's{i:03d}',before_observation=f'o{i:03d}',
            after_observation=f'o{i+1:03d}' if f'o{i+1:03d}' in observations else None,
            commanded_action=action,actual_execution=dict(status='EXECUTED',physical_execution=True,operator_accepted=True),
            telemetry=dict(samples=len(states)),outcome=dict(after_provenance='next iteration initial RGB if present; never substitute final TCP for an image',continuity_verified=False)))
    return dict(observations=observations,steps=steps,excluded_iterations=excluded)


def prefix_import(imported,prefix):
    value=copy.deepcopy(imported)
    mapping={k:prefix+k for k in value['observations']}
    value['observations']={mapping[k]:v for k,v in value['observations'].items()}
    for s in value['steps']:
        s['step_id']=prefix+s['step_id'];s['before_observation']=mapping[s['before_observation']]
        s['after_observation']=mapping[s['after_observation']] if s['after_observation'] else None
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--graph',type=Path,required=True);ap.add_argument('--assisted-session',type=Path,required=True)
    ap.add_argument('--manual-session',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--response',type=Path,help='validate a previously saved model delta without another model call')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=bool(args.response))
    graph=load_frozen(args.graph);trajectory=load_frozen(graph['source_trajectory']['path'])
    imports=[prefix_import(import_session(args.assisted_session),'new_assisted_'),
             prefix_import(import_manual(args.manual_session),'new_manual_')]
    new_steps=[];new_obs={}
    for imp in imports:new_steps+=imp['steps'];new_obs.update(imp['observations'])
    trajectory['steps']+=new_steps;trajectory['observations'].update(new_obs)
    trajectory['extension_imports']=[dict(session=str(p.resolve()),excluded=imp['excluded_iterations']) for p,imp in zip([args.assisted_session,args.manual_session],imports)]
    trajectory.setdefault('limitations',[]).extend([
        'Do not bridge sessions across the failed assisted iter0009 or the intervening abort.',
        'Latest manual trajectory lacks an after RGB; unknown sink only.',
        'Free-manual telemetry is evidence, not a validated two-anchor replay primitive.'])
    if (out/'trajectory.json').exists():
        if load_frozen(out/'trajectory.json')!=trajectory:raise ValueError('resume trajectory differs')
    else:frozen(out/'trajectory.json',trajectory)
    schema=obj(new_nodes=array(NODE),observation_assignments=array(obj(observation_id=S,node_id=S)),new_edges=array(EDGE),limitations=array(S))
    old_ids={n['node_id'] for n in graph['nodes']};old_edges={e['edge_id'] for e in graph['edges']}
    step_map={s['step_id']:s for s in new_steps}
    built={}
    def validate(delta):
        g=copy.deepcopy(graph);nodes={n['node_id']:n for n in g['nodes']}
        for n in delta['new_nodes']:
            if n['node_id'] in nodes:raise ValueError('new node ID collides')
            if set(n['representative_observations'])-set(new_obs):raise ValueError('new node references must be new observations')
            nodes[n['node_id']]=copy.deepcopy(n);g['nodes'].append(nodes[n['node_id']])
        # Redundant unchanged legacy assignments carry no new semantic decisions.
        assignments=[]
        for assignment in delta['observation_assignments']:
            if assignment['observation_id'] in new_obs:
                assignments.append(assignment)
            elif (assignment['node_id'] not in old_ids or assignment['observation_id'] not in
                  nodes[assignment['node_id']]['representative_observations']):
                raise ValueError('cannot reassign an existing observation')
        if len(assignments)!=len(new_obs) or {v['observation_id'] for v in assignments}!=set(new_obs):raise ValueError('assign every new image exactly once')
        for v in assignments:
            node=nodes[v['node_id']];oid=v['observation_id']
            if v['node_id'] in graph.get('terminal_node_ids',[]) and any(s['before_observation']==oid for s in new_steps):raise ValueError('cannot attach outgoing demonstrations to terminal goal')
            if oid not in node['representative_observations']:node['representative_observations'].append(oid)
        for n in nodes.values():
            for oid in n['representative_observations']:
                if oid in new_obs and next(a['node_id'] for a in assignments if a['observation_id']==oid)!=n['node_id']:raise ValueError('inconsistent image assignment')
        for edge in delta['new_edges']:
            e=copy.deepcopy(edge)
            if e['edge_id'] in old_edges:raise ValueError('edge ID collision')
            s=step_map[e['seed_step_id']]
            e['seed_action']=dict(exact_command=s['commanded_action'],source_observation=s['before_observation'])
            allowed=set(trajectory['observations'])|{step['step_id'] for step in trajectory['steps']}
            prose=[v for v in e['outcome_evidence'] if v not in allowed]
            if prose:
                # Preserve model prose in the descriptive field; bind evidence IDs
                # from the recorder instead of treating sentences as identifiers.
                e['observed_outcome']+='\n视觉依据：'+'；'.join(prose)
                e['outcome_evidence']=list(dict.fromkeys(
                    [v for v in e['outcome_evidence'] if v in allowed]+
                    [s['step_id'],s['before_observation']]+
                    ([s['after_observation']] if s['after_observation'] else [])))
            e.update(replayable=False,replay_block_reason=('free manual motion requires dedicated replay compiler' if s['commanded_action']['type']=='ARM6_FREE_MANUAL_DEMONSTRATION' else 'new demonstration requires replay template preflight and outgoing edge selection'))
            if not s['after_observation']:e['replay_block_reason']='no after RGB; audit-only incoming edge to unknown sink'
            g['edges'].append(e)
        g['limitations']+=delta['limitations']
        core={k:copy.deepcopy(g[k]) for k in GRAPH['properties']}
        core['edges']=[{k:e[k] for k in EDGE['required']} for e in core['edges']]
        graph_validation(core,trajectory);built['graph']=g
    req=Requests(out/'model',CodexMultimodal(timeout=900),max_calls=2,schemas={'extend':schema},
        common='Offline visual compilation of Arm6 cloth experience graph. Chinese descriptions. Return only schema-valid JSON. Never execute robot actions.\n'+GARMENT_FRAME,
        prompts={'extend':'''Extend the supplied graph with ALL new confirmed demonstrations. Compare ALL attached reference RGBs directly. Preserve existing nodes/edges; assign every new image to an existing visually equivalent node or a new node. Rotation, translation and minor wrinkles alone do not create states, but cloth coverage, layer and fold-boundary differences may. Do not use action compatibility alone to merge images. IDs do not imply sequence. Every new step needs exactly one edge. Use {} for seed_action.exact_command; host binds source telemetry. Do not invent grasp targets for free-manual records: describe visible region plus measured motion with uncertainty; sampled keyframes are not proof of cloth acquisition. Existing goal n007 remains designated; do not invent completion for missing images. Missing after RGB requires a separate unobserved zero-confidence sink with no representatives and no outgoing edges. No cross-session synthetic edges. Failed/aborted iterations are not accepted demonstrations; explicitly retain the gap and never treat later session initial RGB as their measured outcome. All new edges ONE_SHOT_HYPOTHESIS, outcomes UNKNOWN or visible NO_PROGRESS, not causal SUCCESS. Node descriptions should be concise and image-grounded. Keep existing IDs, prefix new nodes x_, edges ex_. Supply uncertainty when folds cannot be resolved. New node representative IDs must agree with observation_assignments.'''})
    payload=dict(existing_nodes=graph['nodes'],existing_edges=[{k:e[k] for k in ['edge_id','source_node','target_node']} for e in graph['edges']],goal_nodes=graph['terminal_node_ids'],new_steps=new_steps,limitations=trajectory['limitations'],import_gaps=trajectory['extension_imports'])
    images=[(k,Path(v['rgb']['path'])) for k,v in trajectory['observations'].items()]
    if args.response:
        import jsonschema
        delta=read(args.response);jsonschema.validate(delta,schema);validate(delta)
    else:req.ask('extend',payload,images,validator=validate)
    metrics=req.metrics()
    if args.response:
        log=out/'model/requests.jsonl'
        attempts=rows(log) if log.exists() else []
        metrics=dict(model_calls=len(attempts),reasoning_time_s=sum(a.get('elapsed_s',0) for a in attempts),
            completed_responses=sum(bool(a.get('raw_response')) for a in attempts),
            reuse_response=str(args.response),normalization='discard redundant unchanged legacy assignments; preserve evidence prose in outcome text and bind recorded evidence IDs')
    g=built['graph'];g['source_trajectory']=evidence_file(out/'trajectory.json');g['graph_id']+='_extended_20261018'
    g['extension_metrics']=metrics;g['revision_history'].append(dict(operation='visual_extension_two_sessions',source_graph=evidence_file(args.graph),added_steps=[s['step_id'] for s in new_steps],raw_files_modified=False))
    frozen(out/'graph.json',g);render(g,trajectory,out)
    summary=dict(status='COMPILED',nodes=len(g['nodes']),edges=len(g['edges']),added_edges=len(new_steps),observations=len(trajectory['observations']),goal_nodes=g['terminal_node_ids'],new_manual_edges=sum(s['commanded_action']['type']=='ARM6_FREE_MANUAL_DEMONSTRATION' for s in new_steps),missing_after_steps=[s['step_id'] for s in new_steps if not s['after_observation']],model_metrics=metrics,robot_calls=0)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
