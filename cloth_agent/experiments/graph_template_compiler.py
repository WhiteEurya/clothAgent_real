"""Offline, per-edge seed command compiler. No model or robot calls."""
import argparse
import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .graph_minimal_grounding import edge_hash, relative_slots, expand_slots, write, register_frame
from .graph_policy import load_frozen, frozen, checked_file, digest, file_hash

PROTOCOL = 'seed-compiled-policy-v1'


def seed_slots(edge, step, obs):
    """Invert measured world XY, not perspective-project elevated TCPs onto fabric."""
    if step['commanded_action'] != edge['seed_action']['exact_command']:
        raise ValueError('exact executed command mismatch')
    ops = step['commanded_action']['operations']
    if (len(ops) < 2 or ops[0]['method'] != 'grasp'
            or any(o['method'] not in ('move_tcp', 'move_transfer') for o in ops[1:])
            or any(set(o['arguments']) != {'right'} for o in ops)):
        raise ValueError('unsupported primitive skeleton: requires one arm7 grasp then transfers')
    lowering = step.get('intermediate_evidence', {}).get('grasp_height_policy.json', {}).get('content', {})
    if lowering.get('mode') != 'declared_grasp_lowering' or len(lowering.get('changes', [])) != 1:
        raise ValueError('missing unambiguous actual grasp lowering record')
    change = lowering['changes'][0]
    if change['arm'] != 'right':raise ValueError('lowering arm mismatch')
    slots = []; errors = []
    cloud = obs.xyz
    valid = np.isfinite(cloud).all(axis=2)
    valid[:2] = False; valid[-2:] = False; valid[:, :2] = False; valid[:, -2:] = False
    for i, op in enumerate(ops):
        pose = np.asarray(op['arguments']['right'], float)
        if pose.shape != (6,) or not np.isfinite(pose).all():raise ValueError('invalid executed pose')
        xy = pose[:2] * 1000
        distance = np.where(valid, np.linalg.norm(cloud[:, :, :2] - xy, axis=2), np.inf)
        if not np.isfinite(distance).any():raise ValueError('no usable surface')
        y, x = np.unravel_index(np.argmin(distance), distance.shape)
        surface = obs.sample([int(x), int(y)], 20.)
        error = float(np.linalg.norm(surface[:2] - xy))
        if error > 2.:raise ValueError(f'operation {i}: measured XY reconstruction exceeds 2 mm ({error:.3f})')
        z = float(pose[2] * 1000 - surface[2])
        if i == 0:
            if abs(pose[2]*1000-change['applied_z_mm']) > .1:raise ValueError('lowering differs from executed command')
            if abs(change['original_z_mm']-surface[2]) > 2.:raise ValueError('grasp surface reconstruction exceeds 2 mm')
            z = 0.
        if not 0 <= z <= 300:raise ValueError(f'operation {i}: unsupported surface-relative height {z}')
        slots.append(dict(operation_index=i,pixel_xy=[int(x),int(y)],z_above_surface_mm=z,rpy_rad=pose[3:].tolist()))
        errors.append(error)
    profile = relative_slots(slots)
    rebuilt = expand_slots(profile,slots[0]['pixel_xy'],slots[-1]['pixel_xy'],0.)
    if any(a['pixel_xy'] != b['pixel_xy'] for a,b in zip(slots,rebuilt)):raise ValueError('normalized geometry roundtrip failed')
    return slots, profile, errors, lowering


def compile_edge(edge, step, trajectory, *, frame_mode="garment"):
    from cloth_agent.dual_arm.observation import Observation
    record = dict(edge_id=edge['edge_id'],action_family=edge['action_family'],
        grasp_semantic_relation=edge['action_template']['grasp_relation'],
        target_semantic_relation=edge['action_template']['target_relation'],
        observed_outcome=edge['observed_outcome'],seed_outcome=edge['seed_outcome'],
        exact_executed_command=step['commanded_action'],
        seed_grasp_px=None,seed_target_px=None,normalized_waypoint_geometry=None,
        surface_relative_height_profile=None,orientation_rule=None,
        primitive_sequence=[dict(index=i,method=o['method']) for i,o in enumerate((step.get('commanded_action') or {}).get('operations',[]))])
    try:
        if step['actual_execution'].get('status') != 'EXECUTED':raise ValueError('execution not completed')
        evidence = trajectory['observations'][step['before_observation']]
        obs = Observation.load(checked_file(evidence['observation']).parent)
        if obs.image.resolve() != checked_file(evidence['rgb']).resolve():raise ValueError('seed RGB/geometry mismatch')
        slots,profile,errors,lowering = seed_slots(edge,step,obs)
        record.update(seed_pixel_source='Host inversion of executed world XY against source RGB-D; <=2 mm reconstruction error, not a claimed original GPT pixel',
            seed_grasp_px=slots[0]['pixel_xy'],seed_target_px=slots[-1]['pixel_xy'],
            normalized_waypoint_geometry=profile,surface_relative_height_profile=[s['z_above_surface_mm'] for s in slots],
            orientation_rule=dict(kind='seed_world_rpy_rotated_by_registered_garment_axis',
                seed_semantic_rule=edge['action_template']['yaw_rule'],max_yaw_delta_deg=30),
            reconstruction_xy_error_mm=errors,actual_grasp_lowering=lowering)
        if frame_mode in {'action_local','action_local_contact'}:
            from .action_local_frame import compile_local
            item=compile_local(edge,step,obs,record,contact_z=frame_mode=='action_local_contact')
            record.update(action_local_profile=item['profile'],action_local_seed_basis=item['seed_basis'],orientation_rule=item['orientation_rule'])
            if not edge['replayable']:raise ValueError('experience only: not selected as replayable successful/progress edge')
            record.update(status='MINIMAL_SUPPORTED',reason=None,minimal_grounding_spec=item['spec'],host_motion_template=item)
            return record
        if frame_mode != 'garment':raise ValueError('unknown frame mode')
        if not edge['replayable']:raise ValueError('experience only: not selected as replayable successful/progress edge')
        visual = edge.get('minimal_visual_evidence')
        if not visual or visual['status'] != 'RESOLVED' or visual['confidence'] < .8:
            raise ValueError('missing reliable seed garment frame / fabric registration ROI')
        frame = visual['garment_frame'];size=list(Image.open(obs.image).size)
        roi=visual['registration_roi']
        if not (0<=roi[0]<roi[2]<=size[0] and 0<=roi[1]<roi[3]<=size[1]):raise ValueError('invalid fabric registration ROI')
        axis=np.asarray(obs.sample(frame['collar_pixel_xy'],20.))[:2]-np.asarray(obs.sample(frame['hem_center_pixel_xy'],20.))[:2]
        if not np.isfinite(axis).all() or np.linalg.norm(axis)<50:raise ValueError('unreliable seed garment axis')
        spec=dict(edge_id=edge['edge_id'],grasp=record['grasp_semantic_relation'],target=record['target_semantic_relation'])
        item=dict(edge_sha256=edge_hash(edge),spec=spec,primitive_sequence=[o['method'] for o in step['commanded_action']['operations']],
            profile=profile,reference_image=str(obs.image),reference_sha256=file_hash(obs.image),
            frame=frame,world_axis_yaw=math.atan2(axis[1],axis[0]),image_size=size,camera_serial=obs.meta['camera_serial'],
            registration_roi=roi,query_registration_roi=[0,0,*size],confidence_min=.8,
            orientation_rule=record['orientation_rule'],source_seed_step=edge['seed_step_id'],
            exact_command_sha256=digest(step['commanded_action']),
            release_rule='unchanged native executor release after recorded primitive skeleton',scope='One-shot seed template; no automatic full fallback')
        # Reject templates whose own reference lacks the support required at runtime.
        register_frame(item,obs)
        record.update(status='MINIMAL_SUPPORTED',reason=None,minimal_grounding_spec=spec,host_motion_template=item)
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
        record.update(status='MINIMAL_UNSUPPORTED',reason=f'{type(exc).__name__}: {exc}')
    return record


def policy_projection(source, trajectory):
    graph=copy.deepcopy(source)
    observed={n['node_id'] for n in source['nodes'] if n['representative_observations']}
    steps={s['step_id']:s for s in trajectory['steps']}
    chosen=set()
    for e in graph['edges']:
        s=steps[e['seed_step_id']]
        eligible=(e['seed_outcome'] in ('SUCCESS','PROGRESS') and e['generalization_status']=='ONE_SHOT_HYPOTHESIS'
            and s['actual_execution'].get('status')=='EXECUTED' and bool(s.get('after_observation'))
            and e['source_node'] in observed and e['target_node'] in observed and e['source_node']!=e['target_node'])
        # Stable first successful seed action per decision state; preserve all alternatives as experience.
        e['replayable']=bool(eligible and e['source_node'] not in chosen)
        if e['replayable']:chosen.add(e['source_node'])
        e.update(policy_outcome=e['seed_outcome'],
                 evidence_status=('MISSING_AFTER' if not s.get('after_observation') else
                     'FAILED_EVIDENCE' if e['seed_outcome'] in ('FAILURE','NO_PROGRESS') else
                     'ONE_SHOT_HYPOTHESIS' if eligible else 'UNVERIFIED_OUTCOME'),
                 preference='PREFERRED' if e['replayable'] else 'EXCLUDED',
                 stop_reason=None if e['replayable'] else 'REQUIRES_FRESH_VALIDATION')
    graph.update(protocol=PROTOCOL,match_nodes=[copy.deepcopy(n) for n in source['nodes'] if n['node_id'] in observed],
                 terminal_node_ids=[])  # no inferred task completion from a trajectory endpoint
    return graph


def compile_policy(graph_path, output, *, frame_mode="action_local_contact"):
    graph_path=Path(graph_path).resolve();output=Path(output).resolve()
    source=load_frozen(graph_path);trajectory=load_frozen(source['seed_trajectory_path'])
    if digest(trajectory)!=source['seed_trajectory_hash']:raise ValueError('seed provenance mismatch')
    from .graph_policy import validate_replay_graph
    validate_replay_graph(source,trajectory)
    if output.exists():
        existing=load_frozen(output/'policy_graph.json')
        if existing['compiler_source_sha256']!=file_hash(graph_path):raise ValueError('existing compiler output differs')
        bundle=load_frozen(output/'host_templates.json')
        if bundle.get('frame_mode','garment')!=frame_mode:raise ValueError('existing output uses a different frame mode; use a new output directory')
        return existing
    output.mkdir(parents=True)
    graph=policy_projection(source,trajectory);steps={s['step_id']:s for s in trajectory['steps']}
    records=[compile_edge(e,steps[e['seed_step_id']],trajectory,frame_mode=frame_mode) for e in graph['edges']]
    bundle=dict(version='minimal-grounding-host-template-v1',frame_mode=frame_mode,source_graph=str(graph_path),
        edges={r['edge_id']:r['host_motion_template'] for r in records if r['status']=='MINIMAL_SUPPORTED'})
    for e,r in zip(graph['edges'],records):
        e['minimal_status']=r['status'];e['minimal_unsupported_reason']=r['reason']
        if r['status']=='MINIMAL_SUPPORTED':
            e['minimal_grounding_spec']=r['minimal_grounding_spec'];e['host_motion_template']=r['host_motion_template']
    graph.update(compiler_source_path=str(graph_path),compiler_source_sha256=file_hash(graph_path),
        minimal_templates_path=str(output/'host_templates.json'))
    frozen(output/'host_templates.json',bundle);frozen(output/'edge_experiences.json',records)
    summary=dict(total_edges=len(records),minimal_compiled_edges=len(bundle['edges']),
        unsupported=[dict(edge_id=r['edge_id'],reason=r['reason']) for r in records if r['status']=='MINIMAL_UNSUPPORTED'])
    write(output/'summary.json',summary)
    (output/'report.md').write_text('# Seed Host template compilation\n\n'+
        f"Edges: {len(records)}; minimal supported: {len(bundle['edges'])}.\n\n"+
        '\n'.join(f"- {r['edge_id']} ({r['action_family']}): {r['status']}; {r['reason'] or 'compiled from own executed seed command'}. Grasp: {r['grasp_semantic_relation']}; target: {r['target_semantic_relation']}" for r in records)+
        '\n\nUse graph_policy_replay --graph policy_graph.json --grounding-mode minimal (bundle auto-discovered), or --grounding-mode full. No automatic full fallback. Terminal completion is not inferred.\n')
    frozen(output/'policy_graph.json',graph)
    return graph


def validate_compiled_policy(graph):
    if graph.get('protocol') != PROTOCOL:raise ValueError('incorrect compiler protocol')
    source=load_frozen(graph['compiler_source_path'])
    if file_hash(graph['compiler_source_path'])!=graph['compiler_source_sha256']:raise ValueError('compiler source changed')
    trajectory=load_frozen(source['seed_trajectory_path'])
    if digest(trajectory)!=source['seed_trajectory_hash']:raise ValueError('seed changed')
    from .graph_policy import validate_replay_graph
    validate_replay_graph(source,trajectory)
    expected=policy_projection(source,trajectory)
    for key in ('nodes','match_nodes','terminal_node_ids','goal','grasp_offset_mm','seed_trajectory_path','seed_trajectory_hash'):
        if graph[key]!=expected[key]:raise ValueError('compiled policy changed '+key)
    if len(graph['edges'])!=len(expected['edges']):raise ValueError('compiled edge coverage changed')
    for a,b in zip(graph['edges'],expected['edges']):
        for key,value in b.items():
            if a.get(key)!=value:raise ValueError('compiled edge changed '+key)
    return trajectory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame-mode',choices=['action_local_contact','action_local','garment'],default='action_local_contact')
    a=p.parse_args();g=compile_policy(a.graph,a.output,frame_mode=a.frame_mode)
    print((a.output/'summary.json').read_text())

if __name__=='__main__':main()
